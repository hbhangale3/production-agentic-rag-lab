from datetime import date, datetime
from typing import Any, Literal

SearchSort = Literal["relevance", "published_date"]
SortOrder = Literal["asc", "desc"]

SEARCHABLE_FIELDS = frozenset({"title", "abstract", "raw_text", "authors"})
FILTERABLE_FIELDS = frozenset({"categories", "published_date"})
SORTABLE_FIELDS = frozenset({"_score", "published_date"})
DEFAULT_SEARCH_FIELDS = ("title^3", "abstract^2", "raw_text", "authors")
SEARCH_SOURCE_FIELDS = (
    "arxiv_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published_date",
    "updated_date",
    "page_count",
    "parser_used",
)


def build_simple_query(query: str, *, size: int = 10) -> dict[str, Any]:
    return _body(_multi_match(query, DEFAULT_SEARCH_FIELDS), size)


def build_match_query(query: str, *, field: str = "abstract", size: int = 10) -> dict[str, Any]:
    _validate_query(query)
    _validate_fields([field])
    return _body({"match": {field: query.strip()}}, size)


def build_multi_match_query(
    query: str,
    *,
    fields: list[str] | tuple[str, ...] | None = None,
    size: int = 10,
) -> dict[str, Any]:
    return _body(_multi_match(query, fields or DEFAULT_SEARCH_FIELDS), size)


def build_boosting_query(
    positive_query: str,
    negative_query: str,
    *,
    negative_boost: float = 0.2,
    size: int = 10,
) -> dict[str, Any]:
    _validate_query(negative_query)
    if not 0 < negative_boost <= 1:
        raise ValueError("negative_boost must be greater than 0 and at most 1")
    query = {
        "boosting": {
            "positive": _multi_match(positive_query, DEFAULT_SEARCH_FIELDS),
            "negative": _multi_match(negative_query, DEFAULT_SEARCH_FIELDS),
            "negative_boost": negative_boost,
        }
    }
    return _body(query, size)


def build_filtered_query(
    query: str,
    *,
    categories: list[str] | None = None,
    published_from: date | datetime | None = None,
    published_to: date | datetime | None = None,
    size: int = 10,
) -> dict[str, Any]:
    filters = _filters(categories, published_from, published_to)
    return _body({"bool": {"must": [_multi_match(query, DEFAULT_SEARCH_FIELDS)], "filter": filters}}, size)


def build_sorted_query(
    query: str,
    *,
    sort_by: SearchSort = "relevance",
    sort_order: SortOrder = "desc",
    size: int = 10,
) -> dict[str, Any]:
    body = _body(_multi_match(query, DEFAULT_SEARCH_FIELDS), size)
    body["sort"] = _sort(sort_by, sort_order)
    return body


def build_combined_query(
    query: str,
    *,
    categories: list[str] | None = None,
    published_from: date | datetime | None = None,
    published_to: date | datetime | None = None,
    preferred_category: str | None = None,
    sort_by: SearchSort = "relevance",
    sort_order: SortOrder = "desc",
    size: int = 10,
) -> dict[str, Any]:
    bool_query: dict[str, Any] = {
        "must": [_multi_match(query, DEFAULT_SEARCH_FIELDS)],
        "filter": _filters(categories, published_from, published_to),
    }
    if preferred_category is not None:
        preferred = preferred_category.strip()
        if not preferred:
            raise ValueError("preferred_category cannot be blank")
        bool_query["should"] = [{"term": {"categories": {"value": preferred, "boost": 2.0}}}]

    body = _body({"bool": bool_query}, size)
    body["sort"] = _sort(sort_by, sort_order)
    return body


def _multi_match(query: str, fields: list[str] | tuple[str, ...]) -> dict[str, Any]:
    _validate_query(query)
    validated_fields = _validate_fields(fields)
    return {
        "multi_match": {
            "query": query.strip(),
            "fields": validated_fields,
            "type": "best_fields",
        }
    }


def _filters(
    categories: list[str] | None,
    published_from: date | datetime | None,
    published_to: date | datetime | None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    clean_categories = [category.strip() for category in categories or [] if category.strip()]
    if clean_categories:
        filters.append({"terms": {"categories": clean_categories}})
    if published_from and published_to and published_from > published_to:
        raise ValueError("published_from cannot be later than published_to")
    if published_from or published_to:
        boundaries = {}
        if published_from:
            boundaries["gte"] = published_from.isoformat()
        if published_to:
            boundaries["lte"] = published_to.isoformat()
        filters.append({"range": {"published_date": boundaries}})
    return filters


def _sort(sort_by: SearchSort, sort_order: SortOrder) -> list[dict[str, dict[str, str]]]:
    if sort_by not in ("relevance", "published_date"):
        raise ValueError("sort_by must be 'relevance' or 'published_date'")
    if sort_order not in ("asc", "desc"):
        raise ValueError("sort_order must be 'asc' or 'desc'")
    field = "_score" if sort_by == "relevance" else "published_date"
    return [{field: {"order": sort_order}}]


def _body(query: dict[str, Any], size: int) -> dict[str, Any]:
    if not 1 <= size <= 100:
        raise ValueError("size must be between 1 and 100")
    return {"query": query, "size": size, "_source": list(SEARCH_SOURCE_FIELDS)}


def _validate_query(query: str) -> None:
    if not query.strip():
        raise ValueError("query cannot be blank")


def _validate_fields(fields: list[str] | tuple[str, ...]) -> list[str]:
    if not fields:
        raise ValueError("at least one searchable field is required")
    for field in fields:
        base_field = field.split("^", maxsplit=1)[0]
        if base_field not in SEARCHABLE_FIELDS:
            raise ValueError(f"unsupported searchable field: {base_field}")
    return list(fields)
