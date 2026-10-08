from datetime import date

from src.services.opensearch.chunk_bm25_query_builder import (
    CHUNK_BM25_FIELDS,
    build_chunk_bm25_query,
)


def test_builds_boosted_chunk_query_without_embedding() -> None:
    body = build_chunk_bm25_query("clinical fairness", size=20)

    assert body["size"] == 20
    assert body["query"]["bool"]["must"][0]["multi_match"] == {
        "query": "clinical fairness",
        "fields": list(CHUNK_BM25_FIELDS),
    }
    assert set(CHUNK_BM25_FIELDS) == {"paper_title^3", "section_title^2", "chunk_text"}
    assert "embedding" not in body["_source"]


def test_applies_exact_category_and_date_filters() -> None:
    body = build_chunk_bm25_query(
        "robots",
        size=10,
        categories=["cs.RO"],
        published_from=date(2026, 1, 1),
        published_to=date(2026, 12, 31),
    )

    assert body["query"]["bool"]["filter"] == [
        {"terms": {"categories": ["cs.RO"]}},
        {"range": {"published_date": {"gte": "2026-01-01", "lte": "2026-12-31"}}},
    ]
