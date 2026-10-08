from opensearchpy import OpenSearch
from src.config import Settings, get_settings
from src.services.opensearch.client import OpenSearchClient


def make_opensearch_client(settings: Settings | None = None) -> OpenSearchClient:
    """Create the application OpenSearch client from application settings."""
    config = settings or get_settings()
    raw_client = OpenSearch(hosts=[config.opensearch_host])
    return OpenSearchClient(client=raw_client, index_name=config.opensearch_index_name)
