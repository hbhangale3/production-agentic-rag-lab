from datetime import date
from unittest.mock import Mock

import pytest
from src.exceptions import ChunkBM25SearchError, HybridSearchError, VectorSearchError
from src.schemas.vector_search import ChunkSearchHit, VectorSearchResult
from src.services.search.hybrid_service import HybridSearchService


def hit(chunk_id: str, score: float = 1.0) -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id, arxiv_id="p", chunk_index=0, chunk_text="text", word_count=1, score=score
    )


def result(*ids: str) -> VectorSearchResult:
    return VectorSearchResult(query="q", hits=[hit(value) for value in ids], total=len(ids))


def service(bm25=None, vector=None, **kwargs):
    b = Mock(); v = Mock()
    b.search.return_value = bm25 if bm25 is not None else result("both", "lexical")
    v.search.return_value = vector if vector is not None else result("both", "semantic")
    return HybridSearchService(bm25_service=b, vector_service=v, **kwargs), b, v


def test_candidate_depth_filters_and_hybrid_results() -> None:
    instance, bm25, vector = service(candidate_multiplier=4, max_results=100)
    answer = instance.search(
        " query ", size=5, categories=[" cs.AI "], published_from=date(2026, 1, 1)
    )

    assert answer.retrieval_mode == "hybrid"
    assert answer.count == 3
    for retriever in (bm25, vector):
        retriever.search.assert_called_once_with(
            "query", size=20, categories=["cs.AI"],
            published_from=date(2026, 1, 1), published_to=None,
        )


def test_candidate_depth_is_capped() -> None:
    instance, bm25, vector = service(candidate_multiplier=10, max_results=25)
    instance.search("q", size=5)
    assert bm25.search.call_args.kwargs["size"] == 25
    assert vector.search.call_args.kwargs["size"] == 25


@pytest.mark.parametrize("name", ["bm25", "vector"])
def test_single_failure_returns_explicit_fallback(name: str) -> None:
    instance, bm25, vector = service()
    if name == "bm25":
        bm25.search.side_effect = ChunkBM25SearchError("lexical down")
        expected = "vector_fallback"
    else:
        vector.search.side_effect = VectorSearchError("embedding down")
        expected = "bm25_fallback"

    answer = instance.search("q")
    assert answer.retrieval_mode == expected
    assert answer.degradation_reason
    assert all(
        (item.bm25_rank is None if name == "bm25" else item.vector_rank is None)
        for item in answer.results
    )


def test_both_fail_raises() -> None:
    instance, bm25, vector = service()
    bm25.search.side_effect = ChunkBM25SearchError("down")
    vector.search.side_effect = VectorSearchError("down")
    with pytest.raises(HybridSearchError, match="Both hybrid retrievers failed"):
        instance.search("q")


def test_both_empty_is_successful_hybrid() -> None:
    instance, _, _ = service(bm25=result(), vector=result())
    answer = instance.search("q")
    assert answer.retrieval_mode == "hybrid" and answer.results == []


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"rrf_k": 0}, "rrf_k"), ({"candidate_multiplier": 0}, "candidate_multiplier"),
])
def test_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        service(**kwargs)


@pytest.mark.parametrize(("query", "size"), [(" ", 5), ("q", 0), ("q", 101)])
def test_invalid_request_does_not_call_retrievers(query, size) -> None:
    instance, bm25, vector = service()
    with pytest.raises(ValueError):
        instance.search(query, size=size)
    bm25.search.assert_not_called(); vector.search.assert_not_called()
