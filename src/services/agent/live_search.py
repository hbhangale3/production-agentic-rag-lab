"""Read-only live arXiv discovery of metadata and abstracts for the agent graph."""

import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.schemas.arxiv import ArxivPaper
from src.services.arxiv.client import ARXIV_VERSION_SUFFIX, ArxivClient

MAX_LIVE_QUERY_TERMS = 16
_QUERY_TERM = re.compile(r"[^\W_]+")
_IGNORED_QUERY_TERMS = frozenset(
    """a an and andnot are as at be by can could current do does for from how in is it its of on or not
    should that the their them these this those to us was we were what when where which who why will
    with would""".split()
)


class LiveSearchError(Exception):
    """Live research discovery could not be executed."""


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Return the version-independent identifier used to compare papers."""

    return ARXIV_VERSION_SUFFIX.sub("", arxiv_id.strip())


def _clean(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


@dataclass(frozen=True)
class LiveArxivPaper:
    """One live discovery candidate: metadata and abstract only, never PDF content."""

    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published_date: datetime
    updated_date: datetime | None
    pdf_url: str

    def __post_init__(self) -> None:
        for name in ("arxiv_id", "title", "abstract", "pdf_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"live paper {name} must be a non-empty string")
        for name in ("authors", "categories"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"live paper {name} must be a tuple of strings")


@dataclass(frozen=True)
class LiveArxivSearchResult:
    """Ordered, deduplicated candidates for the retrieval query that produced them."""

    query: str
    candidates: tuple[LiveArxivPaper, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, LiveArxivPaper) for candidate in self.candidates
        ):
            raise ValueError("live search candidates must be a tuple of LiveArxivPaper")

    @property
    def count(self) -> int:
        return len(self.candidates)


class LiveResearchSearchService(Protocol):
    """Side-effect-free discovery boundary the graph depends on."""

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        exclude_arxiv_ids: Collection[str] = (),
    ) -> LiveArxivSearchResult:
        """Return at most ``max_results`` candidates not already in ``exclude_arxiv_ids``."""
        ...


def extract_query_terms(text: str) -> list[str]:
    """Return distinct alphanumeric content terms in first-seen order."""

    terms: list[str] = []
    seen: set[str] = set()
    for term in _QUERY_TERM.findall(text):
        key = term.casefold()
        if len(term) < 2 or key in _IGNORED_QUERY_TERMS or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def build_arxiv_search_query(query: str) -> str | None:
    """Turn free text into ``all:term OR ...`` so it cannot alter arXiv query syntax."""

    terms = extract_query_terms(query)[:MAX_LIVE_QUERY_TERMS]
    return " OR ".join(f"all:{term}" for term in terms) or None


class ArxivLiveSearchService:
    """Adapt the existing arXiv API client to bounded, read-only candidate discovery."""

    def __init__(self, *, arxiv_client: ArxivClient) -> None:
        self.arxiv_client = arxiv_client

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        exclude_arxiv_ids: Collection[str] = (),
    ) -> LiveArxivSearchResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        normalized_query = _clean(query)
        excluded = {normalize_arxiv_id(arxiv_id) for arxiv_id in exclude_arxiv_ids if isinstance(arxiv_id, str)}
        excluded.discard("")
        search_query = build_arxiv_search_query(normalized_query)
        if search_query is None:
            return LiveArxivSearchResult(query=normalized_query)
        # Ask for a few extra so already-local papers do not starve the bounded result.
        request_size = min(max_results + len(excluded), 2 * max_results)
        try:
            papers = await self.arxiv_client.fetch_papers(
                search_query=search_query,
                max_results=request_size,
                sort_by="relevance",
                sort_order="descending",
            )
        except Exception as exc:
            raise LiveSearchError("live arXiv search failed") from exc

        candidates: list[LiveArxivPaper] = []
        seen = set(excluded)
        for paper in papers:
            candidate = self._to_candidate(paper)
            if candidate is None or candidate.arxiv_id in seen:
                continue
            seen.add(candidate.arxiv_id)
            candidates.append(candidate)
            if len(candidates) == max_results:
                break
        return LiveArxivSearchResult(query=normalized_query, candidates=tuple(candidates))

    @staticmethod
    def _to_candidate(paper: ArxivPaper) -> LiveArxivPaper | None:
        arxiv_id = normalize_arxiv_id(paper.arxiv_id) if isinstance(paper.arxiv_id, str) else ""
        title = _clean(paper.title)
        abstract = _clean(paper.abstract)
        if not arxiv_id or not title or not abstract:
            return None
        return LiveArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=tuple(cleaned for author in paper.authors if (cleaned := _clean(author))),
            categories=tuple(cleaned for category in paper.categories if (cleaned := _clean(category))),
            published_date=paper.published_date,
            updated_date=paper.updated_date,
            pdf_url=str(paper.pdf_url),
        )
