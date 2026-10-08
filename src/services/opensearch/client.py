import logging
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException
from src.services.opensearch.index_config import ARXIV_PAPERS_MAPPING

logger = logging.getLogger(__name__)

HEALTHY_CLUSTER_STATUSES = {"green", "yellow"}


class OpenSearchClient:
    """Small application wrapper around the OpenSearch Python client."""

    def __init__(self, *, client: OpenSearch, index_name: str) -> None:
        if not index_name.strip():
            raise ValueError("index_name cannot be empty")

        self._client = client
        self.index_name = index_name.strip()

    def health_check(self) -> bool:
        """Return whether the cluster is reachable and usable by this application."""
        try:
            health = self._client.cluster.health()
        except OpenSearchException as exc:
            logger.warning("OpenSearch health check failed: %s", exc)
            return False

        status = health.get("status")
        healthy = status in HEALTHY_CLUSTER_STATUSES
        if not healthy:
            logger.warning("OpenSearch reported an unhealthy cluster status: %s", status or "missing")
        return healthy

    def index_exists(self) -> bool:
        """Return whether the configured index exists and is reachable."""
        try:
            return self._index_exists()
        except OpenSearchException as exc:
            logger.warning("Could not check OpenSearch index %s: %s", self.index_name, exc)
            return False

    def create_index(self) -> bool:
        """Create the configured index if absent without modifying an existing index."""
        try:
            if self._index_exists():
                logger.info("OpenSearch index %s already exists; leaving it unchanged", self.index_name)
                return True
            self._client.indices.create(index=self.index_name, body=ARXIV_PAPERS_MAPPING)
            logger.info("Created OpenSearch index %s", self.index_name)
            return True
        except OpenSearchException as exc:
            logger.warning("Could not create OpenSearch index %s: %s", self.index_name, exc)
            return False

    def get_index_stats(self) -> dict[str, int] | None:
        """Return normalized document-count and storage statistics for the index."""
        try:
            stats = self._client.indices.stats(index=self.index_name)
        except OpenSearchException as exc:
            logger.warning("Could not read OpenSearch index stats for %s: %s", self.index_name, exc)
            return None

        totals = stats["_all"]["total"]
        return {
            "document_count": totals["docs"]["count"],
            "size_in_bytes": totals["store"]["size_in_bytes"],
        }

    def index_document(self, document_id: str, document: dict[str, Any]) -> bool:
        """Create or replace one document using a stable caller-provided ID."""
        try:
            self._client.index(index=self.index_name, id=document_id, body=document)
            return True
        except OpenSearchException as exc:
            logger.warning("Could not index OpenSearch document %s: %s", document_id, exc)
            return False

    def search(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Execute caller-built search DSL against the configured index."""
        try:
            return self._client.search(index=self.index_name, body=body)
        except OpenSearchException as exc:
            logger.warning("OpenSearch search failed for index %s: %s", self.index_name, exc)
            return None

    def _index_exists(self) -> bool:
        return bool(self._client.indices.exists(index=self.index_name))

    def close(self) -> None:
        """Release transport resources owned by the underlying client."""
        self._client.close()
