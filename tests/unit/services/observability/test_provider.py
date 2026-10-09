from unittest.mock import Mock, patch

import pytest
from src.config import Settings
from src.services.observability import (
    LangfuseObservabilityProvider,
    NoOpObservabilityProvider,
    ObservabilityStatus,
    make_observability_provider,
)


def enabled_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        debug=False,
        langfuse_enabled=True,
        langfuse_public_key="pk-test-only",
        langfuse_secret_key="obvious-fake-secret",
        **overrides,
    )


@pytest.mark.anyio
async def test_noop_provider_and_nested_observations_are_harmless() -> None:
    provider = NoOpObservabilityProvider()
    trace = provider.start_trace(name="rag-request", metadata={"safe": True})
    child = trace.start_span(name="retrieval", metadata={"count": 2})
    child.update(metadata={"status": "ok"}, error_type="SafeErrorType")
    child.end()
    trace.end()

    assert provider.status is ObservabilityStatus.DISABLED
    assert provider.capture_content is False
    await provider.flush()
    await provider.close()
    await provider.close()


def test_disabled_factory_does_not_construct_sdk() -> None:
    with patch("src.services.observability.factory.Langfuse") as sdk:
        provider = make_observability_provider(Settings(_env_file=None, langfuse_enabled=False))

    assert isinstance(provider, NoOpObservabilityProvider)
    assert provider.status is ObservabilityStatus.DISABLED
    sdk.assert_not_called()


def test_enabled_factory_constructs_current_sdk_with_safe_configuration() -> None:
    client = Mock()
    settings = enabled_settings(
        langfuse_host="https://observe.internal.example",
        langfuse_timeout_seconds=7,
        environment="testing",
    )
    with patch("src.services.observability.factory.Langfuse", return_value=client) as sdk:
        provider = make_observability_provider(settings)

    assert isinstance(provider, LangfuseObservabilityProvider)
    assert provider.status is ObservabilityStatus.CONFIGURED
    assert provider.capture_content is False
    sdk.assert_called_once_with(
        public_key="pk-test-only",
        secret_key="obvious-fake-secret",
        base_url="https://observe.internal.example",
        timeout=7,
        environment="testing",
        tracing_enabled=True,
    )


def test_construction_failure_degrades_without_leaking_details(caplog) -> None:
    with patch(
        "src.services.observability.factory.Langfuse",
        side_effect=RuntimeError("obvious-fake-secret private host detail"),
    ):
        provider = make_observability_provider(enabled_settings())

    assert isinstance(provider, NoOpObservabilityProvider)
    assert provider.status is ObservabilityStatus.UNAVAILABLE
    assert "obvious-fake-secret" not in caplog.text
    assert "private host detail" not in caplog.text


def test_adapter_wraps_trace_and_nested_span_without_exposing_sdk_types() -> None:
    child = Mock()
    root = Mock()
    root.start_observation.return_value = child
    client = Mock()
    client.start_observation.return_value = root
    provider = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=1)

    trace = provider.start_trace(name="rag-request", metadata={"retrieval_mode": "hybrid"})
    span = trace.start_span(name="retrieval", metadata={"result_count": 2})
    span.update(metadata={"status": "ok"}, error_type="SafeFailure")
    span.end()
    span.end()

    client.start_observation.assert_called_once_with(
        name="rag-request", as_type="span", metadata={"retrieval_mode": "hybrid"}
    )
    root.start_observation.assert_called_once_with(
        name="retrieval", as_type="span", metadata={"result_count": 2}
    )
    child.update.assert_called_once_with(
        metadata={"status": "ok"}, level="ERROR", status_message="SafeFailure"
    )
    child.end.assert_called_once_with()
    assert trace is not root
    assert span is not child


@pytest.mark.anyio
async def test_flush_and_owned_shutdown_delegate_and_are_idempotent() -> None:
    client = Mock()
    provider = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=1)

    await provider.flush()
    await provider.close()
    await provider.close()

    client.flush.assert_called_once_with()
    client.shutdown.assert_called_once_with()
    assert provider.status is ObservabilityStatus.UNAVAILABLE


@pytest.mark.anyio
async def test_flush_and_shutdown_failures_are_best_effort(caplog) -> None:
    client = Mock()
    client.flush.side_effect = RuntimeError("private flush payload")
    client.shutdown.side_effect = RuntimeError("private shutdown payload")
    provider = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=1)

    await provider.flush()
    await provider.close()

    assert "private flush payload" not in caplog.text
    assert "private shutdown payload" not in caplog.text


@pytest.mark.anyio
async def test_caller_owned_client_is_not_shutdown() -> None:
    client = Mock()
    provider = LangfuseObservabilityProvider(
        client=client,
        shutdown_timeout_seconds=1,
        owns_client=False,
    )

    await provider.close()

    client.shutdown.assert_not_called()
