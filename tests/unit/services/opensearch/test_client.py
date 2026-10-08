from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from opensearchpy.exceptions import OpenSearchException
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.factory import make_opensearch_client
from src.services.opensearch.index_config import ARXIV_PAPERS_MAPPING


@pytest.mark.parametrize("status", ["green", "yellow"])
def test_health_check_accepts_reachable_cluster(status: str) -> None:
    raw_client = Mock()
    raw_client.cluster.health.return_value = {"status": status}
    client = OpenSearchClient(client=raw_client, index_name="papers")

    assert client.health_check() is True
    raw_client.cluster.health.assert_called_once_with()


def test_health_check_rejects_red_cluster() -> None:
    raw_client = Mock()
    raw_client.cluster.health.return_value = {"status": "red"}

    assert OpenSearchClient(client=raw_client, index_name="papers").health_check() is False


def test_health_check_returns_false_for_client_exception() -> None:
    raw_client = Mock()
    raw_client.cluster.health.side_effect = OpenSearchException("connection unavailable")

    assert OpenSearchClient(client=raw_client, index_name="papers").health_check() is False


def test_close_delegates_to_underlying_client() -> None:
    raw_client = Mock()
    client = OpenSearchClient(client=raw_client, index_name="papers")

    client.close()

    raw_client.close.assert_called_once_with()


def test_constructor_rejects_empty_index_name() -> None:
    with pytest.raises(ValueError, match="index_name cannot be empty"):
        OpenSearchClient(client=Mock(), index_name="  ")


def test_factory_uses_configured_host_and_retains_index_name() -> None:
    settings = SimpleNamespace(
        opensearch_host="http://search.internal:9200",
        opensearch_index_name="research-papers",
    )
    raw_client = Mock()

    with patch("src.services.opensearch.factory.OpenSearch", return_value=raw_client) as constructor:
        client = make_opensearch_client(settings)

    constructor.assert_called_once_with(hosts=["http://search.internal:9200"])
    assert client.index_name == "research-papers"
    assert client._client is raw_client


@pytest.mark.parametrize(("exists", "expected"), [(True, True), (False, False)])
def test_index_exists_uses_configured_index(exists: bool, expected: bool) -> None:
    raw_client = Mock()
    raw_client.indices.exists.return_value = exists
    client = OpenSearchClient(client=raw_client, index_name="configured-papers")

    assert client.index_exists() is expected
    raw_client.indices.exists.assert_called_once_with(index="configured-papers")


def test_index_exists_returns_false_for_client_exception() -> None:
    raw_client = Mock()
    raw_client.indices.exists.side_effect = OpenSearchException("connection unavailable")

    assert OpenSearchClient(client=raw_client, index_name="papers").index_exists() is False


def test_create_index_uses_central_mapping_and_configured_name() -> None:
    raw_client = Mock()
    raw_client.indices.exists.return_value = False
    client = OpenSearchClient(client=raw_client, index_name="configured-papers")

    assert client.create_index() is True
    raw_client.indices.create.assert_called_once_with(index="configured-papers", body=ARXIV_PAPERS_MAPPING)


def test_create_index_is_noop_when_index_exists() -> None:
    raw_client = Mock()
    raw_client.indices.exists.return_value = True

    assert OpenSearchClient(client=raw_client, index_name="papers").create_index() is True
    raw_client.indices.create.assert_not_called()


@pytest.mark.parametrize("operation", ["exists", "create"])
def test_create_index_returns_false_for_client_exception(operation: str) -> None:
    raw_client = Mock()
    if operation == "exists":
        raw_client.indices.exists.side_effect = OpenSearchException("connection unavailable")
    else:
        raw_client.indices.exists.return_value = False
        raw_client.indices.create.side_effect = OpenSearchException("creation failed")

    assert OpenSearchClient(client=raw_client, index_name="papers").create_index() is False


def test_get_index_stats_returns_normalized_values_for_configured_index() -> None:
    raw_client = Mock()
    raw_client.indices.stats.return_value = {
        "_all": {"total": {"docs": {"count": 17}, "store": {"size_in_bytes": 4096}}}
    }
    client = OpenSearchClient(client=raw_client, index_name="configured-papers")

    assert client.get_index_stats() == {"document_count": 17, "size_in_bytes": 4096}
    raw_client.indices.stats.assert_called_once_with(index="configured-papers")


def test_get_index_stats_returns_none_for_client_exception() -> None:
    raw_client = Mock()
    raw_client.indices.stats.side_effect = OpenSearchException("stats unavailable")

    assert OpenSearchClient(client=raw_client, index_name="papers").get_index_stats() is None


def test_mapping_matches_persisted_searchable_fields_without_vectors() -> None:
    properties = ARXIV_PAPERS_MAPPING["mappings"]["properties"]

    assert properties["arxiv_id"]["type"] == "keyword"
    assert properties["title"]["type"] == "text"
    assert properties["authors"]["type"] == "text"
    assert properties["abstract"]["type"] == "text"
    assert properties["categories"]["type"] == "keyword"
    assert properties["published_date"]["type"] == "date"
    assert properties["updated_date"]["type"] == "date"
    assert properties["raw_text"]["type"] == "text"
    assert properties["local_pdf_path"]["type"] == "keyword"
    assert properties["parser_used"]["type"] == "keyword"
    assert properties["page_count"]["type"] == "integer"
    assert "vector" not in str(ARXIV_PAPERS_MAPPING).lower()
