import pytest
from src.schemas.vector_search import ChunkSearchHit
from src.services.search.rrf import reciprocal_rank_fusion


def hit(chunk_id: str, score: float, *, title: str = "canonical") -> ChunkSearchHit:
    return ChunkSearchHit(
        chunk_id=chunk_id,
        arxiv_id="paper",
        chunk_index=0,
        chunk_text="text",
        word_count=1,
        paper_title=title,
        score=score,
    )


def test_rrf_uses_one_based_ranks_and_merges_by_chunk_id() -> None:
    results = reciprocal_rank_fusion(
        [hit("both", 500), hit("bm25", 999)],
        [hit("both", 0.1, title="conflict"), hit("vector", 0.9)],
        k=60,
        size=3,
    )

    both = results[0]
    assert both.rrf_score == pytest.approx(1 / 61 + 1 / 61)
    assert (both.bm25_rank, both.vector_rank) == (1, 1)
    assert (both.bm25_score, both.vector_score) == (500, 0.1)
    assert both.paper_title == "canonical"
    assert len({item.chunk_id for item in results}) == 3
    assert next(item for item in results if item.chunk_id == "bm25").vector_rank is None
    assert next(item for item in results if item.chunk_id == "vector").bm25_rank is None


def test_raw_scores_do_not_affect_fusion_and_ties_are_deterministic() -> None:
    first = reciprocal_rank_fusion(
        [hit("z", 1_000_000)], [hit("a", -50)], k=60, size=2
    )
    second = reciprocal_rank_fusion(
        [hit("z", -1)], [hit("a", 999_999)], k=60, size=2
    )

    assert [item.chunk_id for item in first] == ["a", "z"]
    assert [item.chunk_id for item in second] == ["a", "z"]
    assert first[0].rrf_score == first[1].rrf_score == pytest.approx(1 / 61)


def test_final_size_truncates_and_invalid_parameters_fail() -> None:
    assert len(reciprocal_rank_fusion([hit("a", 1), hit("b", 1)], [], k=60, size=1)) == 1
    with pytest.raises(ValueError, match="k must"):
        reciprocal_rank_fusion([], [], k=0, size=1)
    with pytest.raises(ValueError, match="size must"):
        reciprocal_rank_fusion([], [], k=60, size=0)
