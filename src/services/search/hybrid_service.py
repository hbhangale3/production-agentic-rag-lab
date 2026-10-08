from datetime import date
from typing import Protocol

from src.exceptions import ChunkBM25SearchError, HybridSearchError, VectorSearchError
from src.schemas.hybrid_search import HybridSearchResult
from src.schemas.vector_search import VectorSearchResult
from src.services.search.rrf import reciprocal_rank_fusion
from src.services.search.vector_service import VectorSearchService


class ChunkRetriever(Protocol):
    def search(self, query: str, **kwargs: object) -> VectorSearchResult: ...


class HybridSearchService:
    """Retrieve chunk candidates independently and combine only their ranks."""

    def __init__(
        self,
        *,
        bm25_service: ChunkRetriever,
        vector_service: ChunkRetriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        max_results: int = 100,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be greater than zero")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be greater than zero")
        if max_results < 1:
            raise ValueError("max_results must be greater than zero")
        self.bm25_service = bm25_service
        self.vector_service = vector_service
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.max_results = max_results

    def search(
        self,
        query: str,
        *,
        size: int = 5,
        categories: list[str] | None = None,
        published_from: date | None = None,
        published_to: date | None = None,
    ) -> HybridSearchResult:
        normalized_query = VectorSearchService._validate_query(query)
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= self.max_results:
            raise ValueError(f"size must be an integer between 1 and {self.max_results}")
        normalized_categories = VectorSearchService._validate_filters(
            categories, published_from, published_to
        )
        candidate_size = min(self.max_results, size * self.candidate_multiplier)
        kwargs = {
            "size": candidate_size,
            "categories": normalized_categories,
            "published_from": published_from,
            "published_to": published_to,
        }
        bm25_result = vector_result = None
        bm25_error = vector_error = None
        try:
            bm25_result = self.bm25_service.search(normalized_query, **kwargs)
        except ChunkBM25SearchError as exc:
            bm25_error = exc
        try:
            vector_result = self.vector_service.search(normalized_query, **kwargs)
        except VectorSearchError as exc:
            vector_error = exc
        if bm25_result is None and vector_result is None:
            raise HybridSearchError(
                f"Both hybrid retrievers failed: BM25={bm25_error}; vector={vector_error}"
            )
        mode = "hybrid"
        reason = None
        if vector_result is None:
            mode, reason = "bm25_fallback", str(vector_error)
        elif bm25_result is None:
            mode, reason = "vector_fallback", str(bm25_error)
        results = reciprocal_rank_fusion(
            bm25_result.hits if bm25_result else [],
            vector_result.hits if vector_result else [],
            k=self.rrf_k,
            size=size,
        )
        return HybridSearchResult(
            query=normalized_query,
            retrieval_mode=mode,
            count=len(results),
            results=results,
            degradation_reason=reason,
        )
