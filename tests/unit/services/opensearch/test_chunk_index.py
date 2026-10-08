from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from opensearchpy.exceptions import OpenSearchException
from src.exceptions import ChunkIndexError
from src.services.opensearch.chunk_index import ChunkIndexManager
from src.services.opensearch.factory import make_chunk_index_manager
from src.services.opensearch.index_config import CHUNK_FIELD_TYPES, build_arxiv_chunks_mapping

INDEX_NAME = "configured-chunks"
DIMENSION = 384


def live_mapping(*, dimension: int = DIMENSION) -> dict:
    return {INDEX_NAME: {"mappings": build_arxiv_chunks_mapping(dimension)["mappings"]}}


def manager(raw_client: Mock | None = None) -> tuple[ChunkIndexManager, Mock]:
    client = raw_client or Mock()
    return ChunkIndexManager(client=client, index_name=INDEX_NAME, embedding_dimension=DIMENSION), client


def test_chunk_mapping_contains_required_bm25_metadata_and_vector_fields() -> None:
    definition = build_arxiv_chunks_mapping(DIMENSION)
    properties = definition["mappings"]["properties"]

    assert {field: value["type"] for field, value in properties.items()} == CHUNK_FIELD_TYPES
    assert properties["embedding"]["dimension"] == 384
    assert properties["embedding"]["method"] == {
        "name": "hnsw",
        "space_type": "cosinesimil",
        "engine": "lucene",
    }
    assert definition["settings"]["index"]["knn"] is True


def test_factory_uses_configured_host_index_name_and_dimension_without_creation() -> None:
    settings = SimpleNamespace(
        opensearch_host="http://search.internal:9200",
        opensearch_chunk_index_name="custom-chunks",
        embedding_dimension=768,
    )
    raw_client = Mock()

    with patch("src.services.opensearch.factory.OpenSearch", return_value=raw_client) as constructor:
        result = make_chunk_index_manager(settings)

    constructor.assert_called_once_with(hosts=["http://search.internal:9200"])
    assert result.index_name == "custom-chunks"
    assert result.embedding_dimension == 768
    raw_client.indices.create.assert_not_called()


def test_create_chunk_index_when_missing_uses_only_chunk_index() -> None:
    instance, raw_client = manager()
    raw_client.indices.exists.return_value = False

    assert instance.create_chunk_index() is True

    raw_client.indices.create.assert_called_once_with(
        index=INDEX_NAME,
        body=build_arxiv_chunks_mapping(DIMENSION),
    )
    assert "arxiv-papers" not in str(raw_client.method_calls)


def test_create_chunk_index_is_idempotent_for_compatible_existing_index() -> None:
    instance, raw_client = manager()
    raw_client.indices.exists.return_value = True
    raw_client.indices.get_mapping.return_value = live_mapping()

    assert instance.create_chunk_index() is True

    raw_client.indices.create.assert_not_called()
    raw_client.indices.get_mapping.assert_called_once_with(index=INDEX_NAME)


def test_incompatible_existing_dimension_is_rejected_without_recreation() -> None:
    instance, raw_client = manager()
    raw_client.indices.exists.return_value = True
    raw_client.indices.get_mapping.return_value = live_mapping(dimension=1024)

    with pytest.raises(ChunkIndexError, match="dimension is 1024; expected 384"):
        instance.create_chunk_index()

    raw_client.indices.create.assert_not_called()


def test_incompatible_core_field_is_rejected() -> None:
    instance, _ = manager()
    mapping = live_mapping()
    mapping[INDEX_NAME]["mappings"]["properties"]["chunk_text"] = {"type": "keyword"}

    with pytest.raises(ChunkIndexError, match="chunk_text.*keyword.*text"):
        instance.validate_chunk_index_mapping(mapping)


def test_mapping_retrieval_uses_configured_index() -> None:
    instance, raw_client = manager()
    raw_client.indices.get_mapping.return_value = live_mapping()

    assert instance.get_chunk_index_mapping() == live_mapping()
    raw_client.indices.get_mapping.assert_called_once_with(index=INDEX_NAME)


def test_stats_are_normalized() -> None:
    instance, raw_client = manager()
    raw_client.indices.stats.return_value = {
        "_all": {"total": {"docs": {"count": 0}, "store": {"size_in_bytes": 512}}}
    }

    assert instance.get_chunk_index_stats() == {"document_count": 0, "size_in_bytes": 512}
    raw_client.indices.stats.assert_called_once_with(index=INDEX_NAME)


@pytest.mark.parametrize("operation", ["exists", "create", "mapping", "stats"])
def test_opensearch_failures_are_clear(operation: str) -> None:
    instance, raw_client = manager()
    error = OpenSearchException("backend unavailable")
    if operation == "exists":
        raw_client.indices.exists.side_effect = error
        call = instance.chunk_index_exists
    elif operation == "create":
        raw_client.indices.exists.return_value = False
        raw_client.indices.create.side_effect = error
        call = instance.create_chunk_index
    elif operation == "mapping":
        raw_client.indices.get_mapping.side_effect = error
        call = instance.get_chunk_index_mapping
    else:
        raw_client.indices.stats.side_effect = error
        call = instance.get_chunk_index_stats

    with pytest.raises(ChunkIndexError, match="backend unavailable"):
        call()


def test_close_releases_transport() -> None:
    instance, raw_client = manager()

    instance.close()

    raw_client.close.assert_called_once_with()
