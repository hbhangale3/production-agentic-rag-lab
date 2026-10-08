from datetime import date, datetime
from typing import Any

VECTOR_SOURCE_FIELDS = (
    "chunk_id",
    "arxiv_id",
    "chunk_index",
    "section_title",
    "chunk_text",
    "word_count",
    "paper_title",
    "authors",
    "categories",
    "published_date",
)


def build_vector_query(
    vector: list[float],
    *,
    size: int,
    categories: list[str] | None = None,
    published_from: date | datetime | None = None,
    published_to: date | datetime | None = None,
) -> dict[str, Any]:
    """Build OpenSearch 2.x Lucene k-NN DSL for the chunk embedding field."""
    knn_parameters: dict[str, Any] = {"vector": vector, "k": size}
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
    if filters:
        knn_parameters["filter"] = {"bool": {"filter": filters}}

    return {
        "size": size,
        "_source": list(VECTOR_SOURCE_FIELDS),
        "query": {"knn": {"embedding": knn_parameters}},
    }
