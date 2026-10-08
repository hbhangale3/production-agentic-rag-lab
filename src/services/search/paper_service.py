from datetime import date, datetime

from src.schemas.search import PaperSearchHit, PaperSearchResult
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.query_builder import (
    SearchSort,
    SortOrder,
    build_boosting_query,
    build_combined_query,
    build_filtered_query,
    build_match_query,
    build_multi_match_query,
    build_simple_query,
    build_sorted_query,
)


class PaperSearchService:
    """Expose named BM25 strategies and normalize OpenSearch responses."""

    def __init__(self, opensearch_client: OpenSearchClient) -> None:
        self.opensearch_client = opensearch_client

    def search_simple(self, query: str, *, size: int = 10) -> PaperSearchResult:
        return self._execute(query, build_simple_query(query, size=size))

    def search_match(self, query: str, *, field: str = "abstract", size: int = 10) -> PaperSearchResult:
        return self._execute(query, build_match_query(query, field=field, size=size))

    def search_multi_match(
        self, query: str, *, fields: list[str] | None = None, size: int = 10
    ) -> PaperSearchResult:
        return self._execute(query, build_multi_match_query(query, fields=fields, size=size))

    def search_boosting(
        self,
        positive_query: str,
        negative_query: str,
        *,
        negative_boost: float = 0.2,
        size: int = 10,
    ) -> PaperSearchResult:
        body = build_boosting_query(
            positive_query,
            negative_query,
            negative_boost=negative_boost,
            size=size,
        )
        return self._execute(positive_query, body)

    def search_filtered(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        published_from: date | datetime | None = None,
        published_to: date | datetime | None = None,
        size: int = 10,
    ) -> PaperSearchResult:
        body = build_filtered_query(
            query,
            categories=categories,
            published_from=published_from,
            published_to=published_to,
            size=size,
        )
        return self._execute(query, body)

    def search_sorted(
        self,
        query: str,
        *,
        sort_by: SearchSort = "relevance",
        sort_order: SortOrder = "desc",
        size: int = 10,
    ) -> PaperSearchResult:
        body = build_sorted_query(query, sort_by=sort_by, sort_order=sort_order, size=size)
        return self._execute(query, body)

    def search_combined(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        published_from: date | datetime | None = None,
        published_to: date | datetime | None = None,
        preferred_category: str | None = None,
        sort_by: SearchSort = "relevance",
        sort_order: SortOrder = "desc",
        size: int = 10,
    ) -> PaperSearchResult:
        body = build_combined_query(
            query,
            categories=categories,
            published_from=published_from,
            published_to=published_to,
            preferred_category=preferred_category,
            sort_by=sort_by,
            sort_order=sort_order,
            size=size,
        )
        return self._execute(query, body)

    def _execute(self, query: str, body: dict) -> PaperSearchResult:
        response = self.opensearch_client.search(body)
        if response is None:
            return PaperSearchResult(query=query, successful=False, error="OpenSearch search failed")

        raw_hits = response.get("hits", {})
        total_value = raw_hits.get("total", 0)
        total = total_value.get("value", 0) if isinstance(total_value, dict) else total_value
        hits = [self._normalize_hit(hit) for hit in raw_hits.get("hits", [])]
        return PaperSearchResult(query=query, total=total, hits=hits, took_ms=response.get("took"))

    @staticmethod
    def _normalize_hit(hit: dict) -> PaperSearchHit:
        source = hit.get("_source", {})
        return PaperSearchHit(
            arxiv_id=source.get("arxiv_id") or hit.get("_id", ""),
            title=source.get("title", ""),
            authors=source.get("authors") or [],
            abstract=source.get("abstract", ""),
            categories=source.get("categories") or [],
            published_date=source.get("published_date"),
            updated_date=source.get("updated_date"),
            page_count=source.get("page_count"),
            parser_used=source.get("parser_used"),
            score=hit.get("_score"),
        )
