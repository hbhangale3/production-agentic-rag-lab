from datetime import date, datetime
from typing import Any

from src.services.opensearch.vector_query_builder import VECTOR_SOURCE_FIELDS

CHUNK_BM25_FIELDS = ("paper_title^3", "section_title^2", "chunk_text")


def build_chunk_bm25_query(
    query: str,
    *,
    size: int,
    categories: list[str] | None = None,
    published_from: date | datetime | None = None,
    published_to: date | datetime | None = None,
) -> dict[str, Any]:
    """Build a safe chunk-level BM25 query with optional metadata filters."""
    filters: list[dict[str, Any]] = []
    if categories:
        filters.append({"terms": {"categories": categories}})
    if published_from is not None or published_to is not None:
        bounds = {}
        if published_from is not None:
            bounds["gte"] = published_from.isoformat()
        if published_to is not None:
            bounds["lte"] = published_to.isoformat()
        filters.append({"range": {"published_date": bounds}})

    return {
        "size": size,
        "_source": list(VECTOR_SOURCE_FIELDS),
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": list(CHUNK_BM25_FIELDS)}}],
                "filter": filters,
            }
        },
    }
