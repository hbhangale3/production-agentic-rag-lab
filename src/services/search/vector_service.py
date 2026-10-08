from datetime import date, datetime
from typing import Any

from pydantic import ValidationError
from src.exceptions import ChunkIndexError, EmbeddingError, VectorSearchError
from src.schemas.vector_search import ChunkSearchHit, VectorSearchResult
from src.services.embeddings.base import EmbeddingProvider
from src.services.opensearch.chunk_index import ChunkIndexManager
from src.services.opensearch.vector_query_builder import build_vector_query

MAX_VECTOR_RESULTS = 100


class VectorSearchService:
    """Orchestrate query embedding and read-only chunk-level k-NN retrieval."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        chunk_index: ChunkIndexManager,
        embedding_dimension: int,
        max_results: int = MAX_VECTOR_RESULTS,
    ) -> None:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be greater than zero")
        if max_results < 1:
            raise ValueError("max_results must be greater than zero")
        self.embedding_provider = embedding_provider
        self.chunk_index = chunk_index
        self.embedding_dimension = embedding_dimension
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
        normalized_query = self._validate_query(query)
        self._validate_size(size)
        normalized_categories = self._validate_filters(categories, published_from, published_to)

        try:
            self.chunk_index.ensure_compatible_index()
        except ChunkIndexError as exc:
            raise VectorSearchError(f"Chunk index is unavailable or incompatible: {exc}") from exc

        try:
            vector = self.embedding_provider.embed_query(normalized_query)
        except EmbeddingError as exc:
            raise VectorSearchError(f"Could not embed vector-search query: {exc}") from exc
        except Exception as exc:
            raise VectorSearchError(f"Embedding provider failed unexpectedly: {exc}") from exc

        if not isinstance(vector, list) or len(vector) != self.embedding_dimension:
            actual_dimension = len(vector) if isinstance(vector, list) else "invalid"
            raise VectorSearchError(
                f"Query embedding has dimension {actual_dimension}; expected {self.embedding_dimension}"
            )

        try:
            response = self.chunk_index.search_chunks(
                build_vector_query(
                    vector,
                    size=size,
                    categories=normalized_categories,
                    published_from=published_from,
                    published_to=published_to,
                )
            )
        except ChunkIndexError as exc:
            raise VectorSearchError(f"OpenSearch vector search failed: {exc}") from exc
        return self._normalize_response(normalized_query, response)

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return query.strip()

    def _validate_size(self, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= self.max_results:
            raise ValueError(f"size must be an integer between 1 and {self.max_results}")

    @staticmethod
    def _validate_filters(
        categories: list[str] | None,
        published_from: date | datetime | None,
        published_to: date | datetime | None,
    ) -> list[str] | None:
        if categories is not None:
            if not isinstance(categories, list) or not categories:
                raise ValueError("categories must be a non-empty list when provided")
            if any(not isinstance(category, str) or not category.strip() for category in categories):
                raise ValueError("each category must be a non-empty string")
            categories = [category.strip() for category in categories]
        for field_name, value in (("published_from", published_from), ("published_to", published_to)):
            if value is not None and not isinstance(value, (date, datetime)):
                raise ValueError(f"{field_name} must be a date or datetime")
        if published_from is not None and published_to is not None:
            try:
                invalid_range = published_from > published_to
            except TypeError as exc:
                raise ValueError(
                    "published_from and published_to must use matching date precision"
                ) from exc
            if invalid_range:
                raise ValueError("published_from must not be after published_to")
        return categories

    @classmethod
    def _normalize_response(cls, query: str, response: dict[str, Any]) -> VectorSearchResult:
        try:
            raw_hits = response["hits"]
            hit_items = raw_hits["hits"]
            if not isinstance(raw_hits, dict) or not isinstance(hit_items, list):
                raise TypeError("hits must contain a list")
            total_value = raw_hits.get("total", len(hit_items))
            total = total_value.get("value", 0) if isinstance(total_value, dict) else total_value
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                raise TypeError("total must be a non-negative integer")
            hits = [cls._normalize_hit(hit) for hit in hit_items]
            took = response.get("took")
            if took is not None and (isinstance(took, bool) or not isinstance(took, int) or took < 0):
                raise TypeError("took must be a non-negative integer")
            return VectorSearchResult(query=query, total=total, hits=hits, took_ms=took)
        except (KeyError, TypeError, ValidationError) as exc:
            raise VectorSearchError(f"OpenSearch returned a malformed vector-search response: {exc}") from exc

    @staticmethod
    def _normalize_hit(hit: Any) -> ChunkSearchHit:
        if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
            raise TypeError("each search hit must contain an object _source")
        source = hit["_source"]
        return ChunkSearchHit(
            chunk_id=source["chunk_id"],
            arxiv_id=source["arxiv_id"],
            chunk_index=source["chunk_index"],
            section_title=source.get("section_title"),
            chunk_text=source["chunk_text"],
            word_count=source["word_count"],
            paper_title=source.get("paper_title", ""),
            authors=source.get("authors") or [],
            categories=source.get("categories") or [],
            published_date=source.get("published_date"),
            score=hit.get("_score"),
        )
