from datetime import date
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from src.dependencies import PaperSearchServiceDep
from src.schemas.search import PaperSearchResult

router = APIRouter(prefix="/search", tags=["Search"])

NonBlankQuery = Annotated[str, Query(min_length=1, pattern=r".*\S.*")]
ResultSize = Annotated[int, Query(ge=1, le=100, description="Maximum number of results")]


class SearchField(StrEnum):
    title = "title"
    abstract = "abstract"
    raw_text = "raw_text"
    authors = "authors"


class SearchSort(StrEnum):
    relevance = "relevance"
    published_date = "published_date"


class SortOrder(StrEnum):
    asc = "asc"
    desc = "desc"


def _http_result(result: PaperSearchResult) -> PaperSearchResult:
    if not result.successful:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.error or "Search backend unavailable",
        )
    return result


@router.get(
    "/simple",
    response_model=PaperSearchResult,
    summary="Simple BM25 Search",
    description="Weighted BM25 search across title, abstract, authors, and full paper text.",
)
def simple_search(service: PaperSearchServiceDep, q: NonBlankQuery, size: ResultSize = 10) -> PaperSearchResult:
    return _http_result(service.search_simple(q, size=size))


@router.get(
    "/match",
    response_model=PaperSearchResult,
    summary="Single-Field Match Search",
    description="Demonstrates an OpenSearch match query against one safe searchable field.",
)
def match_search(
    service: PaperSearchServiceDep,
    q: NonBlankQuery,
    field: Annotated[SearchField, Query(description="Paper field to search")] = SearchField.abstract,
    size: ResultSize = 10,
) -> PaperSearchResult:
    return _http_result(service.search_match(q, field=field.value, size=size))


@router.get(
    "/multi-match",
    response_model=PaperSearchResult,
    summary="Multi-Field BM25 Search",
    description="Searches selected paper fields with best-fields BM25; defaults use application-controlled weighting.",
)
def multi_match_search(
    service: PaperSearchServiceDep,
    q: NonBlankQuery,
    fields: Annotated[list[SearchField] | None, Query(description="Repeat to search multiple fields")] = None,
    size: ResultSize = 10,
) -> PaperSearchResult:
    selected_fields = [field.value for field in fields] if fields else None
    return _http_result(service.search_multi_match(q, fields=selected_fields, size=size))


@router.get(
    "/boosting",
    response_model=PaperSearchResult,
    summary="Boosting Search",
    description="Ranks positive matches while demoting, rather than excluding, matches to the negative query.",
)
def boosting_search(
    service: PaperSearchServiceDep,
    positive: NonBlankQuery,
    negative: NonBlankQuery,
    negative_boost: Annotated[float, Query(gt=0, le=1, description="Negative-match score multiplier")] = 0.2,
    size: ResultSize = 10,
) -> PaperSearchResult:
    return _http_result(
        service.search_boosting(positive, negative, negative_boost=negative_boost, size=size)
    )


@router.get(
    "/filter",
    response_model=PaperSearchResult,
    summary="Filtered BM25 Search",
    description="Uses BM25 for relevance while category and publication-date filters restrict eligibility.",
)
def filtered_search(
    service: PaperSearchServiceDep,
    q: NonBlankQuery,
    category: Annotated[list[str] | None, Query(description="Repeat to allow multiple categories")] = None,
    published_from: date | None = None,
    published_to: date | None = None,
    size: ResultSize = 10,
) -> PaperSearchResult:
    return _http_result(
        service.search_filtered(
            q,
            categories=category,
            published_from=published_from,
            published_to=published_to,
            size=size,
        )
    )


@router.get(
    "/sort",
    response_model=PaperSearchResult,
    summary="Sorted BM25 Search",
    description="Orders by BM25 relevance or publication date; scores may be null for explicit date sorting.",
)
def sorted_search(
    service: PaperSearchServiceDep,
    q: NonBlankQuery,
    sort_by: SearchSort = SearchSort.relevance,
    sort_order: SortOrder = SortOrder.desc,
    size: ResultSize = 10,
) -> PaperSearchResult:
    return _http_result(
        service.search_sorted(q, sort_by=sort_by.value, sort_order=sort_order.value, size=size)
    )


@router.get(
    "/combined",
    response_model=PaperSearchResult,
    summary="Combined BM25 Search",
    description="Combines multi-field relevance, non-scoring filters, preference boosting, and safe sorting.",
)
def combined_search(
    service: PaperSearchServiceDep,
    q: NonBlankQuery,
    category: Annotated[list[str] | None, Query(description="Repeat to allow multiple categories")] = None,
    published_from: date | None = None,
    published_to: date | None = None,
    preferred_category: str | None = None,
    sort_by: SearchSort = SearchSort.relevance,
    sort_order: SortOrder = SortOrder.desc,
    size: ResultSize = 10,
) -> PaperSearchResult:
    return _http_result(
        service.search_combined(
            q,
            categories=category,
            published_from=published_from,
            published_to=published_to,
            preferred_category=preferred_category,
            sort_by=sort_by.value,
            sort_order=sort_order.value,
            size=size,
        )
    )
