from datetime import date, datetime

from src.exceptions import ChunkBM25SearchError, ChunkIndexError, VectorSearchError
from src.schemas.vector_search import VectorSearchResult
from src.services.opensearch.chunk_bm25_query_builder import build_chunk_bm25_query
from src.services.opensearch.chunk_index import ChunkIndexManager
from src.services.search.vector_service import VectorSearchService


class ChunkBM25SearchService:
    """Run BM25 against chunk documents and return the shared chunk result model."""

    def __init__(self, *, chunk_index: ChunkIndexManager, max_results: int = 100) -> None:
        if max_results < 1:
            raise ValueError("max_results must be greater than zero")
        self.chunk_index = chunk_index
        self.max_results = max_results

    def search(
        self,
        query: str,
        *,
        size: int = 10,
        categories: list[str] | None = None,
        published_from: date | datetime | None = None,
        published_to: date | datetime | None = None,
    ) -> VectorSearchResult:
        normalized_query = VectorSearchService._validate_query(query)
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= self.max_results:
            raise ValueError(f"size must be an integer between 1 and {self.max_results}")
        normalized_categories = VectorSearchService._validate_filters(
            categories, published_from, published_to
        )
        try:
            self.chunk_index.ensure_compatible_index()
            response = self.chunk_index.search_chunks(
                build_chunk_bm25_query(
                    normalized_query,
                    size=size,
                    categories=normalized_categories,
                    published_from=published_from,
                    published_to=published_to,
                )
            )
            return VectorSearchService._normalize_response(normalized_query, response)
        except (ChunkIndexError, VectorSearchError) as exc:
            raise ChunkBM25SearchError(f"Chunk BM25 search failed: {exc}") from exc
