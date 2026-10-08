from dataclasses import dataclass

from src.schemas.hybrid_search import HybridSearchHit
from src.schemas.vector_search import ChunkSearchHit


@dataclass
class _FusedCandidate:
    hit: ChunkSearchHit
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None


def reciprocal_rank_fusion(
    bm25_hits: list[ChunkSearchHit],
    vector_hits: list[ChunkSearchHit],
    *,
    k: int,
    size: int,
) -> list[HybridSearchHit]:
    """Fuse rankings by chunk ID; BM25 metadata wins deterministic conflicts."""
    if k < 1:
        raise ValueError("RRF k must be greater than zero")
    if size < 1:
        raise ValueError("size must be greater than zero")
    candidates: dict[str, _FusedCandidate] = {}
    for rank, hit in enumerate(bm25_hits, 1):
        candidate = candidates.setdefault(hit.chunk_id, _FusedCandidate(hit=hit))
        if candidate.bm25_rank is None:
            candidate.bm25_rank = rank
            candidate.bm25_score = hit.score
    for rank, hit in enumerate(vector_hits, 1):
        candidate = candidates.setdefault(hit.chunk_id, _FusedCandidate(hit=hit))
        if candidate.vector_rank is None:
            candidate.vector_rank = rank
            candidate.vector_score = hit.score

    def score(candidate: _FusedCandidate) -> float:
        return sum(
            1 / (k + rank)
            for rank in (candidate.bm25_rank, candidate.vector_rank)
            if rank is not None
        )

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -score(candidate),
            min(rank for rank in (candidate.bm25_rank, candidate.vector_rank) if rank is not None),
            candidate.hit.chunk_id,
        ),
    )[:size]
    return [
        HybridSearchHit(
            **candidate.hit.model_dump(exclude={"score"}),
            rrf_score=score(candidate),
            bm25_rank=candidate.bm25_rank,
            vector_rank=candidate.vector_rank,
            bm25_score=candidate.bm25_score,
            vector_score=candidate.vector_score,
        )
        for candidate in ordered
    ]
