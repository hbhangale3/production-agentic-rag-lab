from opensearchpy import OpenSearch
from src.config import Settings, get_settings
from src.services.opensearch.chunk_index import ChunkIndexManager
from src.services.opensearch.client import OpenSearchClient


def make_opensearch_client(settings: Settings | None = None) -> OpenSearchClient:
    """Create the application OpenSearch client from application settings."""
    config = settings or get_settings()
    raw_client = OpenSearch(hosts=[config.opensearch_host])
    return OpenSearchClient(client=raw_client, index_name=config.opensearch_index_name)


def make_chunk_index_manager(settings: Settings | None = None) -> ChunkIndexManager:
    """Create the configured chunk-index manager without creating the index."""
    config = settings or get_settings()
    raw_client = OpenSearch(hosts=[config.opensearch_host])
    return ChunkIndexManager(
        client=raw_client,
        index_name=config.opensearch_chunk_index_name,
        embedding_dimension=config.embedding_dimension,
    )
