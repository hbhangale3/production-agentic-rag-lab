"""Request-time PDF acquisition, parsing, and chunking of selected live papers.

Nothing here persists, indexes, or embeds: PDFs live in a temporary directory
that is removed after parsing, and chunks exist only in the returned result.
"""

import asyncio
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from src.exceptions import PDFNoTextError
from src.schemas.arxiv import ArxivPaper
from src.schemas.parsed_pdf import ParsedPDF
from src.services.agent.live_search import LiveArxivPaper, extract_query_terms
from src.services.arxiv.client import ArxivClient
from src.services.chunking import PaperChunk, PaperChunkingService

LIVE_SOURCE_TYPE = "live_arxiv"
TRUSTED_ARXIV_PDF_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
DEFAULT_LIVE_PDF_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_LIVE_PAPER_TIMEOUT_SECONDS = 120.0


class LivePDFAcquisitionError(Exception):
    """A selected live paper's PDF could not be acquired safely."""


def validate_arxiv_pdf_url(url: object) -> str:
    """Return an https arXiv PDF URL, or raise for anything that is not one."""

    if not isinstance(url, str) or not url.strip():
        raise LivePDFAcquisitionError("live PDF URL is blank")
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise LivePDFAcquisitionError("live PDF URL is malformed") from exc
    if parts.scheme not in {"http", "https"}:
        raise LivePDFAcquisitionError("live PDF URL must use https")
    if host not in TRUSTED_ARXIV_PDF_HOSTS or parts.username is not None or parts.password is not None:
        raise LivePDFAcquisitionError("live PDF URL is not an arXiv host")
    if port not in (None, 443 if parts.scheme == "https" else 80):
        raise LivePDFAcquisitionError("live PDF URL uses an unexpected port")
    if not parts.path.startswith("/pdf/") or len(parts.path) <= len("/pdf/"):
        raise LivePDFAcquisitionError("live PDF URL is not an arXiv PDF path")
    # arXiv feeds may still advertise http links; the download itself is always https.
    return urlunsplit(("https", host, parts.path, "", ""))


@dataclass(frozen=True)
class TransientLiveEvidenceChunk:
    """One in-memory full-text chunk of a live arXiv paper with citation provenance."""

    arxiv_id: str
    chunk_id: str
    chunk_index: int
    paper_title: str
    section_title: str | None
    text: str
    word_count: int
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published_date: datetime | None
    pdf_url: str
    source_type: Literal["live_arxiv"] = LIVE_SOURCE_TYPE

    def __post_init__(self) -> None:
        for name in ("arxiv_id", "chunk_id", "paper_title", "text", "pdf_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"transient chunk {name} must be a non-empty string")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("transient chunk chunk_index must be non-negative")
        for name in ("authors", "categories"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"transient chunk {name} must be a tuple")
        if self.source_type != LIVE_SOURCE_TYPE:
            raise ValueError("transient chunk source_type must be live_arxiv")


@dataclass(frozen=True)
class LiveDocumentProcessingResult:
    """Chunks plus content-free per-paper outcome counts."""

    chunks: tuple[TransientLiveEvidenceChunk, ...] = ()
    selected_count: int = 0
    processed_count: int = 0
    unusable_count: int = 0
    failed_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.chunks, tuple) or any(
            not isinstance(chunk, TransientLiveEvidenceChunk) for chunk in self.chunks
        ):
            raise ValueError("chunks must be a tuple of TransientLiveEvidenceChunk")
        counts = (self.selected_count, self.processed_count, self.unusable_count, self.failed_count)
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("processing counts must be non-negative integers")
        if self.processed_count + self.unusable_count + self.failed_count != self.selected_count:
            raise ValueError("processing counts must account for every selected paper")


class LiveDocumentProcessor(Protocol):
    """Boundary the graph depends on for turning selected papers into transient evidence."""

    async def process(
        self,
        papers: Sequence[LiveArxivPaper],
        *,
        question: str,
        max_chunks_per_paper: int,
    ) -> LiveDocumentProcessingResult:
        """Process each paper independently; one paper's failure must not discard another's chunks."""
        ...


class PDFParser(Protocol):
    """The existing parser's interface, declared here so importing the agent does not load Docling."""

    async def parse_pdf(self, pdf_path: str | Path) -> ParsedPDF: ...


class LivePaperAcquisitionService:
    """Download one validated arXiv PDF through the existing client into a caller-owned directory."""

    def __init__(self, *, arxiv_client: ArxivClient, max_bytes: int = DEFAULT_LIVE_PDF_MAX_BYTES) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.arxiv_client = arxiv_client
        self.max_bytes = max_bytes

    async def acquire(self, paper: LiveArxivPaper, *, directory: Path) -> Path:
        pdf_url = validate_arxiv_pdf_url(paper.pdf_url)
        request = ArxivPaper(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=list(paper.authors),
            abstract=paper.abstract,
            categories=list(paper.categories),
            published_date=paper.published_date,
            updated_date=paper.updated_date,
            pdf_url=pdf_url,
        )
        try:
            return await self.arxiv_client.download_pdf(request, cache_dir=directory, max_bytes=self.max_bytes)
        except Exception as exc:
            raise LivePDFAcquisitionError("live PDF download failed") from exc


def _relevance_terms(text: str) -> frozenset[str]:
    return frozenset(term.casefold() for term in extract_query_terms(text))


def select_relevant_chunks(chunks: Sequence[PaperChunk], *, question: str, limit: int) -> tuple[PaperChunk, ...]:
    """Keep the ``limit`` chunks sharing most question terms, returned in document order.

    This is a deterministic lexical cap to bound memory, not a final rerank.
    """

    if len(chunks) <= limit:
        return tuple(chunks)
    terms = _relevance_terms(question)
    ranked = sorted(
        chunks,
        key=lambda chunk: (-len(terms & _relevance_terms(chunk.text)), chunk.chunk_index),
    )
    return tuple(sorted(ranked[:limit], key=lambda chunk: chunk.chunk_index))


@dataclass(frozen=True)
class _ChunkablePaper:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published_date: datetime | None
    raw_text: str
    sections: tuple[dict[str, str], ...]


class LiveDocumentProcessingService:
    """Acquire, parse, and chunk selected papers one at a time with the existing services."""

    def __init__(
        self,
        *,
        acquisition_service: LivePaperAcquisitionService,
        pdf_parser: PDFParser,
        chunking_service: PaperChunkingService,
        paper_timeout_seconds: float = DEFAULT_LIVE_PAPER_TIMEOUT_SECONDS,
    ) -> None:
        if paper_timeout_seconds <= 0:
            raise ValueError("paper_timeout_seconds must be greater than zero")
        self.acquisition_service = acquisition_service
        self.pdf_parser = pdf_parser
        self.chunking_service = chunking_service
        self.paper_timeout_seconds = paper_timeout_seconds

    async def process(
        self,
        papers: Sequence[LiveArxivPaper],
        *,
        question: str,
        max_chunks_per_paper: int,
    ) -> LiveDocumentProcessingResult:
        if isinstance(max_chunks_per_paper, bool) or not isinstance(max_chunks_per_paper, int) or max_chunks_per_paper < 1:
            raise ValueError("max_chunks_per_paper must be a positive integer")
        chunks: list[TransientLiveEvidenceChunk] = []
        processed = unusable = failed = 0
        # Sequential on purpose: Docling is CPU-heavy and arXiv asks for unhurried requests.
        for paper in papers:
            try:
                async with asyncio.timeout(self.paper_timeout_seconds):
                    paper_chunks = await self._process_paper(
                        paper, question=question, max_chunks=max_chunks_per_paper
                    )
            except Exception:
                failed += 1
                continue
            if paper_chunks:
                processed += 1
                chunks.extend(paper_chunks)
            else:
                unusable += 1
        return LiveDocumentProcessingResult(
            chunks=tuple(chunks),
            selected_count=len(papers),
            processed_count=processed,
            unusable_count=unusable,
            failed_count=failed,
        )

    async def _process_paper(
        self,
        paper: LiveArxivPaper,
        *,
        question: str,
        max_chunks: int,
    ) -> tuple[TransientLiveEvidenceChunk, ...]:
        pdf_url = validate_arxiv_pdf_url(paper.pdf_url)
        with tempfile.TemporaryDirectory(prefix="live-arxiv-") as directory:
            pdf_path = await self.acquisition_service.acquire(paper, directory=Path(directory))
            try:
                parsed = await self.pdf_parser.parse_pdf(pdf_path)
            except PDFNoTextError:
                # The PDF was acquired and converted but has no extractable text: unusable, not failed.
                return ()
        result = self.chunking_service.chunk_paper(self._chunkable(paper, parsed))
        selected = select_relevant_chunks(result.chunks, question=question, limit=max_chunks)
        return tuple(
            TransientLiveEvidenceChunk(
                arxiv_id=paper.arxiv_id,
                chunk_id=f"live::{chunk.chunk_id}",
                chunk_index=chunk.chunk_index,
                paper_title=paper.title,
                section_title=chunk.section_title,
                text=chunk.text,
                word_count=chunk.word_count,
                authors=paper.authors,
                categories=paper.categories,
                published_date=paper.published_date,
                pdf_url=pdf_url,
            )
            for chunk in selected
        )

    @staticmethod
    def _chunkable(paper: LiveArxivPaper, parsed: ParsedPDF) -> _ChunkablePaper:
        return _ChunkablePaper(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            categories=paper.categories,
            published_date=paper.published_date,
            raw_text=parsed.raw_text,
            sections=tuple({"title": section.title, "text": section.text} for section in parsed.sections),
        )
