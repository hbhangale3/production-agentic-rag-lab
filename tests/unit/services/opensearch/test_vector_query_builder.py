from datetime import date

from src.services.opensearch.vector_query_builder import VECTOR_SOURCE_FIELDS, build_vector_query


def test_build_vector_query_uses_lucene_knn_shape_and_requested_size() -> None:
    vector = [0.25] * 384

    body = build_vector_query(vector, size=7)

    assert body["size"] == 7
    assert body["query"] == {"knn": {"embedding": {"vector": vector, "k": 7}}}
    assert body["_source"] == list(VECTOR_SOURCE_FIELDS)


def test_vector_query_never_requests_stored_embedding() -> None:
    body = build_vector_query([0.0] * 384, size=5)

    assert "embedding" not in body["_source"]


def test_vector_query_places_exact_metadata_filters_inside_lucene_knn() -> None:
    body = build_vector_query(
        [0.0] * 384,
        size=5,
        categories=["cs.AI", "cs.RO"],
        published_from=date(2025, 1, 1),
        published_to=date(2025, 12, 31),
    )

    assert body["query"]["knn"]["embedding"]["filter"] == {
        "bool": {
            "filter": [
                {"terms": {"categories": ["cs.AI", "cs.RO"]}},
                {"range": {"published_date": {"gte": "2025-01-01", "lte": "2025-12-31"}}},
            ]
        }
    }
