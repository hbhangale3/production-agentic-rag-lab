from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from opensearchpy.exceptions import OpenSearchException
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.factory import make_opensearch_client


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
