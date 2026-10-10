import asyncio
import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.exceptions import ArxivPDFDownloadError, PDFNoTextError, PDFParserError
from src.schemas.arxiv import ArxivPaper
from src.schemas.parsed_pdf import ParsedPDF, ParsedPDFSection
from src.services.agent import (
    LiveArxivPaper,
    LiveDocumentProcessingResult,
    LiveDocumentProcessingService,
    LivePaperAcquisitionService,
    LivePDFAcquisitionError,
    TransientLiveEvidenceChunk,
)
from src.services.agent.live_documents import (
    DEFAULT_LIVE_PDF_MAX_BYTES,
    select_relevant_chunks,
    validate_arxiv_pdf_url,
)
from src.services.arxiv.client import ArxivClient
from src.services.chunking import PaperChunk, PaperChunkingService
from src.services.pdf_parser.docling_parser import DoclingPDFParser

QUESTION = "How can AI improve healthcare access in India?"


def make_paper(arxiv_id: str = "2501.00001", *, pdf_url: str | None = None) -> LiveArxivPaper:
    return LiveArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Live Paper {arxiv_id}",
        abstract="An abstract.",
        authors=("Ada Lovelace", "Alan Turing"),
        categories=("cs.CY", "cs.AI"),
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        updated_date=datetime(2025, 2, 3, tzinfo=UTC),
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}v2",
    )


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def make_service(
    *,
    parsed: dict[str, ParsedPDF | Exception] | ParsedPDF | None = None,
    download_errors: dict[str, Exception] | None = None,
    target_words: int = 50,
    overlap_words: int = 10,
    min_words: int = 10,
    paper_timeout_seconds: float = 30.0,
):
    """Service wired to a fake client and parser; records every directory and path it is given."""
    seen: dict[str, list] = {"download_dirs": [], "parsed_paths": [], "existed_at_parse": []}

    async def download_pdf(paper: ArxivPaper, *, cache_dir, max_bytes):
        seen["download_dirs"].append(Path(cache_dir))
        if download_errors and paper.arxiv_id in download_errors:
            raise download_errors[paper.arxiv_id]
        path = Path(cache_dir) / ArxivClient.pdf_filename(paper.arxiv_id)
        path.write_bytes(b"%PDF-1.7\nfake")
        return path

    async def parse_pdf(pdf_path):
        path = Path(pdf_path)
        seen["parsed_paths"].append(path)
        seen["existed_at_parse"].append(path.is_file())
        outcome = parsed.get(path.stem) if isinstance(parsed, dict) else parsed
        if outcome is None:
            outcome = ParsedPDF(raw_text=words("body", 120))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = Mock(spec=ArxivClient)
    client.download_pdf = AsyncMock(side_effect=download_pdf)
    parser = Mock(spec=DoclingPDFParser)
    parser.parse_pdf = AsyncMock(side_effect=parse_pdf)
    service = LiveDocumentProcessingService(
        acquisition_service=LivePaperAcquisitionService(arxiv_client=client),
        pdf_parser=parser,
        chunking_service=PaperChunkingService(
            target_words=target_words, overlap_words=overlap_words, min_words=min_words
        ),
        paper_timeout_seconds=paper_timeout_seconds,
    )
    return service, client, parser, seen


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/pdf/2501.00001v2", "https://arxiv.org/pdf/2501.00001v2"),
        ("https://export.arxiv.org/pdf/2501.00001", "https://export.arxiv.org/pdf/2501.00001"),
        ("https://www.arxiv.org/pdf/hep-th/9901001v1", "https://www.arxiv.org/pdf/hep-th/9901001v1"),
        ("https://arxiv.org:443/pdf/2501.00001", "https://arxiv.org/pdf/2501.00001"),
        ("  https://arxiv.org/pdf/2501.00001?download=1#page=2 ", "https://arxiv.org/pdf/2501.00001"),
        ("http://arxiv.org/pdf/2501.00001v1", "https://arxiv.org/pdf/2501.00001v1"),
    ],
)
def test_trusted_arxiv_pdf_urls_are_accepted_and_always_https(url: str, expected: str) -> None:
    assert validate_arxiv_pdf_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        None,
        "not a url",
        "ftp://arxiv.org/pdf/2501.00001",
        "file:///etc/passwd",
        "//arxiv.org/pdf/2501.00001",
        "https://evil.example/pdf/2501.00001",
        "https://arxiv.org.evil.example/pdf/2501.00001",
        "https://evilarxiv.org/pdf/2501.00001",
        "https://arxiv.org@evil.example/pdf/2501.00001",
        "https://user:pass@arxiv.org/pdf/2501.00001",
        "https://arxiv.org:8443/pdf/2501.00001",
        "https://arxiv.org/abs/2501.00001",
        "https://arxiv.org/pdf/",
        "https://arxiv.org/",
        "https://[::1/pdf/2501.00001",
    ],
)
def test_untrusted_or_malformed_pdf_urls_are_rejected(url) -> None:
    with pytest.raises(LivePDFAcquisitionError):
        validate_arxiv_pdf_url(url)


@pytest.mark.anyio
async def test_acquisition_reuses_client_download_with_bound_and_caller_directory(tmp_path) -> None:
    client = Mock(spec=ArxivClient)
    client.download_pdf = AsyncMock(return_value=tmp_path / "2501.00001.pdf")
    service = LivePaperAcquisitionService(arxiv_client=client, max_bytes=1234)

    path = await service.acquire(make_paper(pdf_url="http://arxiv.org/pdf/2501.00001v2"), directory=tmp_path)

    assert path == tmp_path / "2501.00001.pdf"
    request = client.download_pdf.await_args.args[0]
    assert isinstance(request, ArxivPaper)
    assert request.arxiv_id == "2501.00001"
    assert str(request.pdf_url) == "https://arxiv.org/pdf/2501.00001v2"
    assert client.download_pdf.await_args.kwargs == {"cache_dir": tmp_path, "max_bytes": 1234}
    assert [call[0] for call in client.method_calls] == ["download_pdf"]


@pytest.mark.anyio
async def test_acquisition_never_requests_an_untrusted_url(tmp_path) -> None:
    client = Mock(spec=ArxivClient)
    client.download_pdf = AsyncMock()

    with pytest.raises(LivePDFAcquisitionError, match="not an arXiv host"):
        await LivePaperAcquisitionService(arxiv_client=client).acquire(
            make_paper(pdf_url="https://evil.example/pdf/2501.00001"), directory=tmp_path
        )

    client.download_pdf.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [ArxivPDFDownloadError("private exceeds the 1234 byte limit"), TimeoutError("private"), OSError("private")],
)
async def test_acquisition_failures_are_controlled_and_content_free(tmp_path, error: Exception) -> None:
    client = Mock(spec=ArxivClient)
    client.download_pdf = AsyncMock(side_effect=error)

    with pytest.raises(LivePDFAcquisitionError) as caught:
        await LivePaperAcquisitionService(arxiv_client=client).acquire(make_paper(), directory=tmp_path)

    assert str(caught.value) == "live PDF download failed"


def test_acquisition_default_and_invalid_byte_limits() -> None:
    client = Mock(spec=ArxivClient)

    assert LivePaperAcquisitionService(arxiv_client=client).max_bytes == DEFAULT_LIVE_PDF_MAX_BYTES == 20 * 1024 * 1024
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            LivePaperAcquisitionService(arxiv_client=client, max_bytes=invalid)


@pytest.mark.anyio
async def test_structured_sections_become_transient_chunks_with_full_provenance() -> None:
    parsed = ParsedPDF(
        raw_text="ignored when sections exist",
        sections=[
            ParsedPDFSection(title="Introduction", text=words("intro", 30)),
            ParsedPDFSection(title="Methods", text=words("method", 30)),
        ],
    )
    service, client, parser, _ = make_service(parsed=parsed)
    paper = make_paper()

    result = await service.process((paper,), question=QUESTION, max_chunks_per_paper=5)

    assert (result.selected_count, result.processed_count, result.unusable_count, result.failed_count) == (1, 1, 0, 0)
    assert [(chunk.chunk_id, chunk.chunk_index, chunk.section_title) for chunk in result.chunks] == [
        ("live::2501.00001::chunk::000", 0, "Introduction"),
        ("live::2501.00001::chunk::001", 1, "Methods"),
    ]
    first = result.chunks[0]
    assert first == TransientLiveEvidenceChunk(
        arxiv_id="2501.00001",
        chunk_id="live::2501.00001::chunk::000",
        chunk_index=0,
        paper_title="Live Paper 2501.00001",
        section_title="Introduction",
        text=words("intro", 30),
        word_count=30,
        authors=("Ada Lovelace", "Alan Turing"),
        categories=("cs.CY", "cs.AI"),
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2501.00001v2",
    )
    assert first.source_type == "live_arxiv"
    client.download_pdf.assert_awaited_once()
    parser.parse_pdf.assert_awaited_once()


@pytest.mark.anyio
async def test_raw_text_fallback_uses_existing_chunk_size_and_overlap_policy() -> None:
    service, _, _, _ = make_service(parsed=ParsedPDF(raw_text=words("w", 120)))

    result = await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=10)

    expected = PaperChunkingService(target_words=50, overlap_words=10, min_words=10).chunk_paper(
        SimpleNamespace(arxiv_id="2501.00001", raw_text=words("w", 120))
    )
    assert [chunk.text for chunk in result.chunks] == [chunk.text for chunk in expected.chunks]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]
    assert all(chunk.section_title is None for chunk in result.chunks)
    assert all(chunk.word_count <= 50 for chunk in result.chunks)
    assert result.chunks[1].text.split()[:10] == result.chunks[0].text.split()[-10:]


@pytest.mark.anyio
async def test_processing_is_deterministic() -> None:
    service, _, _, _ = make_service()
    papers = (make_paper("2501.00001"), make_paper("2501.00002"))

    first = await service.process(papers, question=QUESTION, max_chunks_per_paper=2)
    second = await service.process(papers, question=QUESTION, max_chunks_per_paper=2)

    assert first == second
    assert [chunk.arxiv_id for chunk in first.chunks] == ["2501.00001"] * 2 + ["2501.00002"] * 2


@pytest.mark.anyio
async def test_chunk_limit_keeps_most_relevant_chunks_in_document_order() -> None:
    sections = [ParsedPDFSection(title=f"S{index}", text=words(f"filler{index}x", 20)) for index in range(6)]
    sections[4] = ParsedPDFSection(title="S4", text="AI improves healthcare access in India " + words("z", 14))
    sections[2] = ParsedPDFSection(title="S2", text="healthcare access " + words("y", 18))
    service, _, _, _ = make_service(parsed=ParsedPDF(raw_text="unused", sections=sections), min_words=5)

    result = await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=3)

    assert [(chunk.chunk_index, chunk.section_title) for chunk in result.chunks] == [(0, "S0"), (2, "S2"), (4, "S4")]
    assert [chunk.chunk_id for chunk in result.chunks] == [
        "live::2501.00001::chunk::000",
        "live::2501.00001::chunk::002",
        "live::2501.00001::chunk::004",
    ]


def test_relevance_cap_is_stable_and_a_no_op_under_the_limit() -> None:
    chunks = [
        PaperChunk(
            chunk_id=f"p::chunk::{index:03d}",
            arxiv_id="p",
            chunk_index=index,
            text=text,
            word_count=len(text.split()),
            has_overlap=False,
        )
        for index, text in enumerate(["alpha beta", "India healthcare", "gamma", "AI India healthcare access"])
    ]

    assert select_relevant_chunks(chunks, question=QUESTION, limit=10) == tuple(chunks)
    assert [chunk.chunk_index for chunk in select_relevant_chunks(chunks, question=QUESTION, limit=2)] == [1, 3]
    assert [chunk.chunk_index for chunk in select_relevant_chunks(chunks, question="???", limit=2)] == [0, 1]


@pytest.mark.anyio
async def test_pdf_lives_in_a_private_temp_directory_that_is_removed_after_parsing(tmp_path) -> None:
    service, client, _, seen = make_service()
    client.pdf_cache_dir = tmp_path / "canonical-cache"

    await service.process((make_paper("2501.00001"), make_paper("../../evil")), question=QUESTION, max_chunks_per_paper=2)

    assert seen["existed_at_parse"] == [True, True]
    assert len(set(seen["download_dirs"])) == 2
    for directory, path in zip(seen["download_dirs"], seen["parsed_paths"], strict=True):
        assert directory.name.startswith("live-arxiv-")
        assert path.parent == directory
        assert not directory.exists()
    assert seen["parsed_paths"][1].name == "evil.pdf"
    assert not (tmp_path / "canonical-cache").exists()


@pytest.mark.anyio
async def test_temp_directory_is_removed_when_parsing_fails() -> None:
    service, _, _, seen = make_service(parsed=PDFParserError("private parser detail"))

    result = await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=2)

    assert result.failed_count == 1
    assert not seen["download_dirs"][0].exists()


@pytest.mark.anyio
async def test_one_failing_paper_does_not_discard_another_papers_chunks() -> None:
    service, client, parser, _ = make_service(
        parsed={"2501.00002": PDFParserError("private parser detail")},
        download_errors={"2501.00003": ArxivPDFDownloadError("private download detail")},
    )
    papers = (make_paper("2501.00001"), make_paper("2501.00002"), make_paper("2501.00003"))

    result = await service.process(papers, question=QUESTION, max_chunks_per_paper=1)

    assert (result.selected_count, result.processed_count, result.unusable_count, result.failed_count) == (3, 1, 0, 2)
    assert [chunk.arxiv_id for chunk in result.chunks] == ["2501.00001"]
    assert client.download_pdf.await_count == 3
    assert parser.parse_pdf.await_count == 2
    assert "private" not in repr(result)


@pytest.mark.anyio
async def test_paper_without_chunkable_text_is_unusable_not_failed() -> None:
    service, _, _, _ = make_service(
        parsed={"2501.00001": ParsedPDF(raw_text="  \n\n  ", sections=[ParsedPDFSection(title="Empty", text=" ")])}
    )

    result = await service.process(
        (make_paper("2501.00001"), make_paper("2501.00002")), question=QUESTION, max_chunks_per_paper=1
    )

    assert (result.processed_count, result.unusable_count, result.failed_count) == (1, 1, 0)
    assert [chunk.arxiv_id for chunk in result.chunks] == ["2501.00002"]


@pytest.mark.anyio
async def test_parser_no_text_signal_classifies_the_paper_as_unusable_not_failed() -> None:
    service, client, parser, seen = make_service(parsed=PDFNoTextError("private no-text detail /tmp/secret.pdf"))

    result = await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=2)

    assert (result.selected_count, result.processed_count, result.unusable_count, result.failed_count) == (1, 0, 1, 0)
    assert result.chunks == ()
    client.download_pdf.assert_awaited_once()
    parser.parse_pdf.assert_awaited_once()
    assert not seen["download_dirs"][0].exists()
    assert "private" not in repr(result)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({"2501.00001": "no_text", "2501.00002": "no_text"}, (0, 2, 0, [])),
        ({"2501.00001": "no_text", "2501.00002": "parser_error"}, (0, 1, 1, [])),
        ({"2501.00001": "no_text", "2501.00002": "download_error"}, (0, 1, 1, [])),
        ({"2501.00001": "parser_error", "2501.00002": "unexpected_error"}, (0, 0, 2, [])),
        ({"2501.00001": "usable", "2501.00002": "no_text"}, (1, 1, 0, ["2501.00001"])),
        ({"2501.00001": "usable", "2501.00002": "parser_error"}, (1, 0, 1, ["2501.00001"])),
    ],
)
async def test_usable_unusable_and_failed_papers_are_counted_separately(outcomes: dict, expected: tuple) -> None:
    parser_outcomes = {
        "no_text": PDFNoTextError("private no-text detail /tmp/secret.pdf"),
        "parser_error": PDFParserError("private parser detail"),
        "unexpected_error": RuntimeError("private unexpected detail"),
    }
    service, _, _, _ = make_service(
        parsed={arxiv_id: parser_outcomes[kind] for arxiv_id, kind in outcomes.items() if kind in parser_outcomes},
        download_errors={
            arxiv_id: ArxivPDFDownloadError("private download detail")
            for arxiv_id, kind in outcomes.items()
            if kind == "download_error"
        },
    )

    result = await service.process(
        tuple(make_paper(arxiv_id) for arxiv_id in outcomes), question=QUESTION, max_chunks_per_paper=1
    )

    processed, unusable, failed, chunk_ids = expected
    assert (result.processed_count, result.unusable_count, result.failed_count) == (processed, unusable, failed)
    assert [chunk.arxiv_id for chunk in result.chunks] == chunk_ids
    assert "private" not in repr(result)


@pytest.mark.anyio
@pytest.mark.parametrize("error", [PDFParserError("produced no text"), RuntimeError("no text"), ValueError("empty")])
async def test_only_the_typed_no_text_signal_is_unusable_other_parser_errors_stay_failures(error: Exception) -> None:
    service, _, _, _ = make_service(parsed=error)

    result = await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=1)

    assert (result.processed_count, result.unusable_count, result.failed_count) == (0, 0, 1)


@pytest.mark.anyio
async def test_untrusted_pdf_url_fails_that_paper_without_any_download() -> None:
    service, client, parser, _ = make_service()

    result = await service.process(
        (make_paper(pdf_url="https://evil.example/pdf/2501.00001"),), question=QUESTION, max_chunks_per_paper=1
    )

    assert (result.processed_count, result.failed_count, result.chunks) == (0, 1, ())
    client.download_pdf.assert_not_awaited()
    parser.parse_pdf.assert_not_awaited()


@pytest.mark.anyio
async def test_per_paper_deadline_fails_only_the_slow_paper() -> None:
    release = asyncio.Event()

    async def slow_download(paper: ArxivPaper, *, cache_dir, max_bytes):
        if paper.arxiv_id == "2501.00001":
            await release.wait()
        path = Path(cache_dir) / "paper.pdf"
        path.write_bytes(b"%PDF-1.7")
        return path

    service, client, _, _ = make_service(paper_timeout_seconds=0.01)
    client.download_pdf = AsyncMock(side_effect=slow_download)

    result = await service.process(
        (make_paper("2501.00001"), make_paper("2501.00002")), question=QUESTION, max_chunks_per_paper=1
    )

    assert (result.processed_count, result.failed_count) == (1, 1)
    assert [chunk.arxiv_id for chunk in result.chunks] == ["2501.00002"]


@pytest.mark.anyio
async def test_papers_are_processed_sequentially_through_the_parsers_own_offload(tmp_path) -> None:
    converter = Mock()
    parser = DoclingPDFParser(converter=converter)
    order: list[str] = []

    async def download_pdf(paper: ArxivPaper, *, cache_dir, max_bytes):
        order.append(f"download:{paper.arxiv_id}")
        path = Path(cache_dir) / f"{paper.arxiv_id}.pdf"
        path.write_bytes(b"%PDF-1.7\nfake")
        return path

    client = Mock(spec=ArxivClient)
    client.download_pdf = AsyncMock(side_effect=download_pdf)
    service = LiveDocumentProcessingService(
        acquisition_service=LivePaperAcquisitionService(arxiv_client=client),
        pdf_parser=parser,
        chunking_service=PaperChunkingService(target_words=50, overlap_words=10, min_words=10),
    )

    async def offload(function, path):
        order.append(f"parse:{Path(path).stem}")
        assert function == parser._convert
        return ParsedPDF(raw_text=words("body", 30))

    with patch("src.services.pdf_parser.docling_parser.asyncio.to_thread", new=AsyncMock(side_effect=offload)) as thread:
        result = await service.process(
            (make_paper("2501.00001"), make_paper("2501.00002")), question=QUESTION, max_chunks_per_paper=1
        )

    assert thread.await_count == 2
    converter.convert.assert_not_called()
    assert order == ["download:2501.00001", "parse:2501.00001", "download:2501.00002", "parse:2501.00002"]
    assert result.processed_count == 2


@pytest.mark.anyio
async def test_only_download_parse_and_chunk_collaborators_are_used() -> None:
    service, client, parser, _ = make_service()

    await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=1)

    assert [call[0] for call in client.method_calls] == ["download_pdf"]
    assert [call[0] for call in parser.method_calls] == ["parse_pdf"]
    client.fetch_papers.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_invalid_chunk_limit_is_a_programming_error(limit) -> None:
    service, _, _, _ = make_service()

    with pytest.raises(ValueError, match="max_chunks_per_paper"):
        await service.process((make_paper(),), question=QUESTION, max_chunks_per_paper=limit)


def test_invalid_paper_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="paper_timeout_seconds"):
        make_service(paper_timeout_seconds=0)


def make_chunk(**overrides) -> TransientLiveEvidenceChunk:
    fields = {
        "arxiv_id": "2501.00001",
        "chunk_id": "live::2501.00001::chunk::000",
        "chunk_index": 0,
        "paper_title": "Title",
        "section_title": None,
        "text": "Chunk text.",
        "word_count": 2,
        "authors": ("Ada",),
        "categories": ("cs.AI",),
        "published_date": None,
        "pdf_url": "https://arxiv.org/pdf/2501.00001",
    }
    return TransientLiveEvidenceChunk(**{**fields, **overrides})


def test_transient_chunk_is_an_immutable_value_type() -> None:
    chunk = make_chunk()

    assert [field.name for field in dataclasses.fields(TransientLiveEvidenceChunk)] == [
        "arxiv_id",
        "chunk_id",
        "chunk_index",
        "paper_title",
        "section_title",
        "text",
        "word_count",
        "authors",
        "categories",
        "published_date",
        "pdf_url",
        "source_type",
    ]
    assert chunk == make_chunk()
    assert hash(chunk) == hash(make_chunk())
    assert chunk.source_type == "live_arxiv"
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"arxiv_id": " "},
        {"chunk_id": ""},
        {"paper_title": ""},
        {"text": "  "},
        {"pdf_url": ""},
        {"chunk_index": -1},
        {"chunk_index": True},
        {"authors": ["Ada"]},
        {"categories": ["cs.AI"]},
        {"source_type": "local"},
    ],
)
def test_transient_chunk_rejects_invalid_fields(overrides: dict) -> None:
    with pytest.raises(ValueError):
        make_chunk(**overrides)


def test_processing_result_counts_must_be_consistent_and_chunks_typed() -> None:
    assert LiveDocumentProcessingResult() == LiveDocumentProcessingResult(chunks=(), selected_count=0)
    with pytest.raises(ValueError, match="account for every selected paper"):
        LiveDocumentProcessingResult(selected_count=2, processed_count=1)
    with pytest.raises(ValueError):
        LiveDocumentProcessingResult(selected_count=1, failed_count=1, chunks=[make_chunk()])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LiveDocumentProcessingResult(selected_count=1, processed_count=1, chunks=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LiveDocumentProcessingResult(selected_count=-1, failed_count=-1)
