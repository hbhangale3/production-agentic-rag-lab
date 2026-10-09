import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult


class TokenCounter(Protocol):
    """Count tokens for budget enforcement without coupling to an LLM SDK."""

    description: str

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class CharacterTokenEstimator:
    """Deterministic estimate; this is not an exact provider billing count."""

    characters_per_token: int = 4
    description: str = "estimated: ceil(Unicode code points / 4)"

    def __post_init__(self) -> None:
        if (
            isinstance(self.characters_per_token, bool)
            or not isinstance(self.characters_per_token, int)
            or self.characters_per_token < 1
        ):
            raise ValueError("characters_per_token must be a positive integer")

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.characters_per_token) if text else 0


@dataclass(frozen=True)
class EvidenceSource:
    """One selected source and its retrieval provenance."""

    label: str
    retrieval_rank: int
    arxiv_id: str
    chunk_id: str
    chunk_index: int
    paper_title: str
    section_title: str | None
    content: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published_date: datetime | None
    rrf_score: float
    bm25_rank: int | None
    vector_rank: int | None
    bm25_score: float | None
    vector_score: float | None
    truncated: bool


@dataclass(frozen=True)
class EvidenceContext:
    """Citation-ready context and the ordered mapping behind its labels."""

    text: str
    sources: tuple[EvidenceSource, ...]
    estimated_tokens: int
    max_context_tokens: int
    token_counting: str
    query: str
    retrieval_mode: Literal["hybrid", "bm25_fallback", "vector_fallback"]
    degradation_reason: str | None


class EvidenceContextBuilder:
    """Select ranked hybrid-search evidence within an evidence-only budget.

    ``max_context_tokens`` does not include any future system prompt, user
    question, or completion allowance. An orchestrator must reserve those
    independently before choosing this budget.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens < 1
        ):
            raise ValueError("max_context_tokens must be a positive integer")
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter or CharacterTokenEstimator()

    @classmethod
    def from_settings(cls, settings: object) -> "EvidenceContextBuilder":
        return cls(max_context_tokens=getattr(settings, "evidence_context_max_tokens"))

    def build(self, result: HybridSearchResult) -> EvidenceContext:
        """Build context in retrieval order without mutating ``result``."""
        blocks: list[str] = []
        sources: list[EvidenceSource] = []
        seen_chunk_ids: set[str] = set()
        seen_evidence: set[str] = set()

        for retrieval_rank, hit in enumerate(result.results, start=1):
            content = self._clean(hit.chunk_text)
            chunk_id = self._clean(hit.chunk_id)
            if not content or (chunk_id and chunk_id in seen_chunk_ids) or content in seen_evidence:
                continue

            label = f"S{len(sources) + 1}"
            full_block = self._render_block(label, hit, content)
            candidate_text = self._join_blocks(blocks, full_block)
            truncated = False
            selected_content = content

            if self.token_counter.count(candidate_text) > self.max_context_tokens:
                selected_content = self._largest_fitting_prefix(label, hit, content, blocks)
                if not selected_content:
                    break
                truncated = selected_content != content
                full_block = self._render_block(label, hit, selected_content)
                candidate_text = self._join_blocks(blocks, full_block)

            blocks.append(full_block)
            sources.append(
                EvidenceSource(
                    label=f"[{label}]",
                    retrieval_rank=retrieval_rank,
                    arxiv_id=self._clean(hit.arxiv_id),
                    chunk_id=chunk_id,
                    chunk_index=hit.chunk_index,
                    paper_title=self._clean(hit.paper_title),
                    section_title=self._clean(hit.section_title) or None,
                    content=selected_content,
                    authors=self._clean_sequence(hit.authors),
                    categories=self._clean_sequence(hit.categories),
                    published_date=hit.published_date
                    if isinstance(hit.published_date, datetime)
                    else None,
                    rrf_score=hit.rrf_score,
                    bm25_rank=hit.bm25_rank,
                    vector_rank=hit.vector_rank,
                    bm25_score=hit.bm25_score,
                    vector_score=hit.vector_score,
                    truncated=truncated,
                )
            )
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            seen_evidence.add(content)
            if truncated:
                break

        text = "\n\n".join(blocks)
        return EvidenceContext(
            text=text,
            sources=tuple(sources),
            estimated_tokens=self.token_counter.count(text),
            max_context_tokens=self.max_context_tokens,
            token_counting=self.token_counter.description,
            query=result.query,
            retrieval_mode=result.retrieval_mode,
            degradation_reason=result.degradation_reason,
        )

    def _largest_fitting_prefix(
        self,
        label: str,
        hit: HybridSearchHit,
        content: str,
        existing_blocks: list[str],
    ) -> str:
        low, high = 1, len(content)
        best = ""
        while low <= high:
            midpoint = (low + high) // 2
            prefix = content[:midpoint].rstrip()
            if not prefix:
                low = midpoint + 1
                continue
            excerpt = f"{prefix} …"
            block = self._render_block(label, hit, excerpt)
            if self.token_counter.count(self._join_blocks(existing_blocks, block)) <= self.max_context_tokens:
                best = excerpt
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    @classmethod
    def _render_block(cls, label: str, hit: HybridSearchHit, content: str) -> str:
        title = cls._clean(hit.paper_title) or "Unknown title"
        arxiv_id = cls._clean(hit.arxiv_id) or "Unknown"
        chunk_id = cls._clean(hit.chunk_id) or "Unknown"
        lines = [
            f"[{label}] {title}",
            f"arXiv ID: {arxiv_id}",
            f"Chunk ID: {chunk_id}",
        ]
        section = cls._clean(hit.section_title)
        if section:
            lines.append(f"Section: {section}")
        lines.append(f"Evidence: {content}")
        return "\n".join(lines)

    @staticmethod
    def _join_blocks(blocks: list[str], block: str) -> str:
        return "\n\n".join((*blocks, block))

    @staticmethod
    def _clean(value: object) -> str:
        return " ".join(value.split()) if isinstance(value, str) else ""

    @classmethod
    def _clean_sequence(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(cleaned for item in value if (cleaned := cls._clean(item)))
