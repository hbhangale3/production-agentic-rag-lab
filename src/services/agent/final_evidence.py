"""Source-agnostic normalization, merge, and semantic rerank of local and live evidence.

Local hybrid-search scores and live evidence are never compared with each
other. Every candidate is rescored against the original question with one
embedding model, and only that score orders the final evidence.
"""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from src.schemas.hybrid_search import HybridSearchResult
from src.services.agent.live_documents import LIVE_SOURCE_TYPE, TransientLiveEvidenceChunk
from src.services.agent.live_search import normalize_arxiv_id
from src.services.embeddings.base import EmbeddingProvider
from src.services.evidence import EvidenceInput

LOCAL_SOURCE_TYPE = "local"
DEFAULT_MAX_RERANK_CANDIDATES = 64


class EvidenceRerankError(Exception):
    """Common relevance scores could not be computed."""


def _clean(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


@dataclass(frozen=True)
class AgentEvidenceCandidate:
    """One evidence chunk in the representation shared by local and live sources."""

    source_type: Literal["local", "live_arxiv"]
    arxiv_id: str
    chunk_id: str
    chunk_index: int
    paper_title: str
    section_title: str | None
    text: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published_date: datetime | None
    source_url: str | None
    original_rank: int
    relevance_score: float | None = None

    def __post_init__(self) -> None:
        if self.source_type not in (LOCAL_SOURCE_TYPE, LIVE_SOURCE_TYPE):
            raise ValueError("evidence candidate source_type must be local or live_arxiv")
        for name in ("arxiv_id", "chunk_id", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evidence candidate {name} must be a non-empty string")
        for name in ("authors", "categories"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"evidence candidate {name} must be a tuple")
        if isinstance(self.original_rank, bool) or not isinstance(self.original_rank, int) or self.original_rank < 1:
            raise ValueError("evidence candidate original_rank must be a positive integer")

    @property
    def embedding_text(self) -> str:
        """Title, section, and text only: the same shape for every source."""

        return "\n".join(part for part in (self.paper_title, self.section_title or "", self.text) if part)


def normalize_local_evidence(result: HybridSearchResult | None) -> tuple[AgentEvidenceCandidate, ...]:
    """Convert hybrid-search hits in retrieval order; their retrieval scores are deliberately dropped."""

    if result is None:
        return ()
    candidates: list[AgentEvidenceCandidate] = []
    for rank, hit in enumerate(result.results, start=1):
        text = _clean(hit.chunk_text)
        arxiv_id, chunk_id = _clean(hit.arxiv_id), _clean(hit.chunk_id)
        if not text or not arxiv_id or not chunk_id:
            continue
        candidates.append(
            AgentEvidenceCandidate(
                source_type=LOCAL_SOURCE_TYPE,
                arxiv_id=arxiv_id,
                chunk_id=chunk_id,
                chunk_index=hit.chunk_index,
                paper_title=_clean(hit.paper_title),
                section_title=_clean(hit.section_title) or None,
                text=text,
                authors=tuple(cleaned for author in hit.authors if (cleaned := _clean(author))),
                categories=tuple(cleaned for category in hit.categories if (cleaned := _clean(category))),
                published_date=hit.published_date,
                source_url=None,
                original_rank=rank,
            )
        )
    return tuple(candidates)


def normalize_live_evidence(
    chunks: Sequence[TransientLiveEvidenceChunk] | None,
) -> tuple[AgentEvidenceCandidate, ...]:
    """Convert transient live chunks in their stored order."""

    return tuple(
        AgentEvidenceCandidate(
            source_type=LIVE_SOURCE_TYPE,
            arxiv_id=chunk.arxiv_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            paper_title=_clean(chunk.paper_title),
            section_title=_clean(chunk.section_title) or None,
            text=_clean(chunk.text),
            authors=chunk.authors,
            categories=chunk.categories,
            published_date=chunk.published_date,
            source_url=chunk.pdf_url,
            original_rank=rank,
        )
        for rank, chunk in enumerate(chunks or (), start=1)
    )


def merge_evidence(
    local: Sequence[AgentEvidenceCandidate],
    live: Sequence[AgentEvidenceCandidate],
) -> tuple[AgentEvidenceCandidate, ...]:
    """Concatenate local then live, dropping repeated chunk IDs and repeated text of the same paper.

    Local comes first, so when the same content exists in both the local
    (already indexed) representation is the one kept.
    """

    merged: list[AgentEvidenceCandidate] = []
    seen_chunk_ids: set[str] = set()
    seen_content: set[tuple[str, str]] = set()
    for candidate in (*local, *live):
        content_key = (normalize_arxiv_id(candidate.arxiv_id), _clean(candidate.text).casefold())
        if candidate.chunk_id in seen_chunk_ids or content_key in seen_content:
            continue
        seen_chunk_ids.add(candidate.chunk_id)
        seen_content.add(content_key)
        merged.append(candidate)
    return tuple(merged)


def to_evidence_inputs(candidates: Sequence[AgentEvidenceCandidate]) -> list[EvidenceInput]:
    """Hand final candidates to the existing context builder without losing provenance."""

    return [
        EvidenceInput(
            chunk_id=candidate.chunk_id,
            arxiv_id=candidate.arxiv_id,
            chunk_index=candidate.chunk_index,
            chunk_text=candidate.text,
            paper_title=candidate.paper_title,
            section_title=candidate.section_title,
            authors=candidate.authors,
            categories=candidate.categories,
            published_date=candidate.published_date,
            source_type=candidate.source_type,
            source_url=candidate.source_url,
        )
        for candidate in candidates
    ]


def _bound_candidates(
    candidates: Sequence[AgentEvidenceCandidate],
    limit: int,
) -> list[AgentEvidenceCandidate]:
    """Cap embedding work, alternating sources so neither is starved, then restore merged order."""

    if len(candidates) <= limit:
        return list(candidates)
    queues = [
        [index for index, candidate in enumerate(candidates) if candidate.source_type == source_type]
        for source_type in (LOCAL_SOURCE_TYPE, LIVE_SOURCE_TYPE)
    ]
    kept: list[int] = []
    while len(kept) < limit:
        for queue in queues:
            if queue and len(kept) < limit:
                kept.append(queue.pop(0))
    return [candidates[index] for index in sorted(kept)]


def _cosine(query: Sequence[float], passage: Sequence[float]) -> float:
    if len(query) != len(passage) or not query:
        raise EvidenceRerankError("embedding dimensions do not match")
    query_norm = math.sqrt(sum(value * value for value in query))
    passage_norm = math.sqrt(sum(value * value for value in passage))
    score = sum(a * b for a, b in zip(query, passage, strict=True)) / (query_norm * passage_norm or math.nan)
    if not math.isfinite(score):
        raise EvidenceRerankError("embedding similarity is not finite")
    return score


class FinalEvidenceSelector:
    """Rescore merged candidates against the question with one embedding model and keep the best."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        max_candidates: int = DEFAULT_MAX_RERANK_CANDIDATES,
    ) -> None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        self.embedding_provider = embedding_provider
        self.max_candidates = max_candidates

    async def select(
        self,
        question: str,
        candidates: Sequence[AgentEvidenceCandidate],
        *,
        max_sources: int,
    ) -> tuple[AgentEvidenceCandidate, ...]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be blank")
        if isinstance(max_sources, bool) or not isinstance(max_sources, int) or max_sources < 1:
            raise ValueError("max_sources must be a positive integer")
        bounded = _bound_candidates(candidates, self.max_candidates)
        if not bounded:
            return ()
        texts = [candidate.embedding_text for candidate in bounded]
        try:
            # The provider is synchronous and CPU-bound; one hop covers the query and one passage batch.
            query_vector, passage_vectors = await asyncio.to_thread(self._embed, question.strip(), texts)
        except Exception as exc:
            raise EvidenceRerankError("evidence embedding failed") from exc
        if len(passage_vectors) != len(bounded):
            raise EvidenceRerankError("embedding count does not match candidate count")
        scored = [
            replace(candidate, relevance_score=_cosine(query_vector, vector))
            for candidate, vector in zip(bounded, passage_vectors, strict=True)
        ]
        # sorted() is stable, so equal scores keep merged order (local before live, then source rank).
        ranked = sorted(scored, key=lambda candidate: -candidate.relevance_score)
        return tuple(ranked[:max_sources])

    def _embed(self, question: str, texts: list[str]) -> tuple[list[float], list[list[float]]]:
        return self.embedding_provider.embed_query(question), self.embedding_provider.embed_passages(texts)
