from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr
from src.config import Settings
from src.exceptions import LLMConfigurationError
from src.services.llm.factory import make_llm_provider
from src.services.llm.groq_provider import GroqLLMProvider


def settings(**overrides) -> Settings:
    values = {"debug": False, "groq_api_key": SecretStr("test-key"), **overrides}
    return Settings(_env_file=None, **values)


def test_factory_builds_owned_client_with_explicit_transport_settings() -> None:
    with patch("src.services.llm.factory.AsyncGroq") as client_type:
        result = make_llm_provider(settings(llm_timeout_seconds=12.5, llm_max_retries=4))

    assert isinstance(result, GroqLLMProvider)
    client_type.assert_called_once_with(api_key="test-key", timeout=12.5, max_retries=4)
    assert result._owns_client is True


def test_factory_reuses_injected_client_without_requiring_credentials() -> None:
    client = Mock()
    config = settings(groq_api_key=None)

    first = make_llm_provider(config, client=client)
    second = make_llm_provider(config, client=client)

    assert first._client is second._client is client
    assert first._owns_client is second._owns_client is False


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("   ")])
def test_factory_rejects_missing_or_blank_credentials(key) -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY is required"):
        make_llm_provider(settings(groq_api_key=key))


def test_factory_rejects_unsupported_provider() -> None:
    config = SimpleNamespace(llm_provider="ollama")

    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider: ollama"):
        make_llm_provider(config)
