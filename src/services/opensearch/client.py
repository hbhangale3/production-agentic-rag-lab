import logging

from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException

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

    def close(self) -> None:
        """Release transport resources owned by the underlying client."""
        self._client.close()
