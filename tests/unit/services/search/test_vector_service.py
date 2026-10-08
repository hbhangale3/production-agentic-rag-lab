from datetime import date, datetime
from unittest.mock import Mock, call

import pytest
from src.exceptions import ChunkIndexError, EmbeddingEncodingError, VectorSearchError
from src.services.search.vector_service import VectorSearchService

DIMENSION = 384


def response(*, hits: list[dict] | None = None, total: int | dict = 1) -> dict:
    items = hits if hits is not None else [raw_hit()]
    return {"took": 4, "hits": {"total": total, "hits": items}}


def raw_hit(
    *, chunk_id: str = "2610.01963::chunk::000", chunk_index: int = 0, score: float = 1.73
) -> dict:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "arxiv_id": "2610.01963",
            "chunk_index": chunk_index,
            "section_title": "Introduction",
            "chunk_text": "Semantic retrieval for clinical access.",
            "word_count": 5,
            "paper_title": "Clinical Access",
            "authors": ["Ada Lovelace"],
            "categories": ["cs.AI", "cs.CY"],
            "published_date": "2026-10-02T12:30:00Z",
            "embedding": [0.99],
        },
    }


def make_service(*, vector: list[float] | None = None, max_results: int = 100):
    provider = Mock()
    provider.embed_query.return_value = vector if vector is not None else [0.1] * DIMENSION
    chunk_index = Mock()
    chunk_index.search_chunks.return_value = response()
    service = VectorSearchService(
        embedding_provider=provider,
        chunk_index=chunk_index,
        embedding_dimension=DIMENSION,
        max_results=max_results,
    )
    return service, provider, chunk_index


def test_query_is_embedded_then_sent_to_chunk_knn_search() -> None:
    service, provider, chunk_index = make_service()

    result = service.search("  clinical access  ", size=5)

    chunk_index.ensure_compatible_index.assert_called_once_with()
    provider.embed_query.assert_called_once_with("clinical access")
    body = chunk_index.search_chunks.call_args.args[0]
    assert body["size"] == 5
    assert body["query"]["knn"]["embedding"]["k"] == 5
    assert len(body["query"]["knn"]["embedding"]["vector"]) == 384
    assert result.query == "clinical access"


def test_result_fields_are_normalized_without_embedding() -> None:
    service, _, _ = make_service()

    result = service.search("clinical access")

    assert result.total == 1
    assert result.took_ms == 4
    assert result.hits[0].model_dump() == {
        "chunk_id": "2610.01963::chunk::000",
        "arxiv_id": "2610.01963",
        "chunk_index": 0,
        "section_title": "Introduction",
        "chunk_text": "Semantic retrieval for clinical access.",
        "word_count": 5,
        "paper_title": "Clinical Access",
        "authors": ["Ada Lovelace"],
        "categories": ["cs.AI", "cs.CY"],
        "published_date": result.hits[0].published_date,
        "score": 1.73,
    }
    assert "embedding" not in result.hits[0].model_dump()


def test_result_order_follows_opensearch_order() -> None:
    service, _, chunk_index = make_service()
    chunk_index.search_chunks.return_value = response(
        hits=[
            raw_hit(chunk_id="paper::chunk::002", chunk_index=2, score=1.9),
            raw_hit(chunk_id="paper::chunk::000", chunk_index=0, score=1.2),
        ],
        total={"value": 2, "relation": "eq"},
    )

    result = service.search("physical world")

    assert [hit.chunk_id for hit in result.hits] == ["paper::chunk::002", "paper::chunk::000"]
    assert [hit.score for hit in result.hits] == [1.9, 1.2]


@pytest.mark.parametrize("query", ["", "   ", "\n\t", None])
def test_blank_or_non_string_query_is_rejected_without_calls(query) -> None:
    service, provider, chunk_index = make_service()

    with pytest.raises(ValueError, match="non-empty"):
        service.search(query)

    provider.embed_query.assert_not_called()
    chunk_index.ensure_compatible_index.assert_not_called()
    chunk_index.search_chunks.assert_not_called()


@pytest.mark.parametrize("size", [0, -1, 101, 1.5, True])
def test_invalid_size_is_rejected_without_calls(size) -> None:
    service, provider, chunk_index = make_service()

    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("valid", size=size)

    provider.embed_query.assert_not_called()
    chunk_index.ensure_compatible_index.assert_not_called()


def test_custom_maximum_is_enforced() -> None:
    service, _, _ = make_service(max_results=3)

    with pytest.raises(ValueError, match="between 1 and 3"):
        service.search("valid", size=4)


def test_embedding_failure_is_clear_and_search_is_not_called() -> None:
    service, provider, chunk_index = make_service()
    provider.embed_query.side_effect = EmbeddingEncodingError("encoder failed")

    with pytest.raises(VectorSearchError, match="Could not embed.*encoder failed"):
        service.search("valid")

    chunk_index.search_chunks.assert_not_called()


def test_wrong_query_vector_dimension_is_rejected_before_search() -> None:
    service, _, chunk_index = make_service(vector=[0.1] * 383)

    with pytest.raises(VectorSearchError, match="dimension 383; expected 384"):
        service.search("valid")

    chunk_index.search_chunks.assert_not_called()


def test_malformed_query_vector_is_rejected_before_search() -> None:
    service, _, chunk_index = make_service(vector="not-a-vector")

    with pytest.raises(VectorSearchError, match="dimension invalid; expected 384"):
        service.search("valid")

    chunk_index.search_chunks.assert_not_called()


def test_valid_filters_are_forwarded_to_lucene_knn_query() -> None:
    service, _, chunk_index = make_service()

    service.search(
        "clinical fairness",
        categories=[" cs.AI "],
        published_from=date(2025, 1, 1),
        published_to=date(2025, 12, 31),
    )

    parameters = chunk_index.search_chunks.call_args.args[0]["query"]["knn"]["embedding"]
    assert parameters["filter"] == {
        "bool": {
            "filter": [
                {"terms": {"categories": ["cs.AI"]}},
                {"range": {"published_date": {"gte": "2025-01-01", "lte": "2025-12-31"}}},
            ]
        }
    }


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"categories": []}, "categories must be a non-empty list"),
        ({"categories": ["  "]}, "each category"),
        ({"published_from": "2025-01-01"}, "published_from must be"),
        (
            {"published_from": date(2025, 2, 1), "published_to": date(2025, 1, 1)},
            "published_from must not be after",
        ),
        (
            {"published_from": date(2025, 1, 1), "published_to": datetime(2025, 1, 2)},
            "matching date precision",
        ),
    ],
)
def test_invalid_filters_are_rejected_before_external_calls(filters: dict, message: str) -> None:
    service, provider, chunk_index = make_service()

    with pytest.raises(ValueError, match=message):
        service.search("valid", **filters)

    provider.embed_query.assert_not_called()
    chunk_index.ensure_compatible_index.assert_not_called()


@pytest.mark.parametrize("message", ["does not exist", "embedding dimension is 768"])
def test_missing_or_incompatible_index_is_clear_and_embedding_is_not_called(message: str) -> None:
    service, provider, chunk_index = make_service()
    chunk_index.ensure_compatible_index.side_effect = ChunkIndexError(message)

    with pytest.raises(VectorSearchError, match="unavailable or incompatible"):
        service.search("valid")

    provider.embed_query.assert_not_called()
    chunk_index.search_chunks.assert_not_called()


def test_opensearch_search_failure_is_clear() -> None:
    service, _, chunk_index = make_service()
    chunk_index.search_chunks.side_effect = ChunkIndexError("transport failure")

    with pytest.raises(VectorSearchError, match="vector search failed.*transport failure"):
        service.search("valid")


def test_empty_result_set_is_normalized() -> None:
    service, _, chunk_index = make_service()
    chunk_index.search_chunks.return_value = response(hits=[], total=0)

    result = service.search("nothing")

    assert result.total == 0
    assert result.hits == []


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {"hits": []},
        {"hits": {"hits": {}}},
        {"hits": {"total": -1, "hits": []}},
        {"hits": {"total": 1, "hits": [{}]}},
        {"hits": {"total": 1, "hits": [{"_source": {}}]}},
    ],
)
def test_malformed_response_or_hit_raises_clear_error(malformed: dict) -> None:
    service, _, chunk_index = make_service()
    chunk_index.search_chunks.return_value = malformed

    with pytest.raises(VectorSearchError, match="malformed vector-search response"):
        service.search("valid")


def test_same_provider_instance_is_reused_across_searches() -> None:
    service, provider, chunk_index = make_service()
    chunk_index.search_chunks.side_effect = [response(), response()]

    service.search("first")
    service.search("second")

    assert provider.embed_query.call_count == 2
    assert service.embedding_provider is provider


def test_service_performs_no_index_writes_and_never_targets_paper_index() -> None:
    service, _, chunk_index = make_service()

    service.search("read only")

    assert chunk_index.method_calls == [
        call.ensure_compatible_index(),
        call.search_chunks(chunk_index.search_chunks.call_args.args[0]),
    ]
    assert "arxiv-papers" not in str(chunk_index.method_calls)
