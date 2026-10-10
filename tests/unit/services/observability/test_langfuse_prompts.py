import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.config import Settings
from src.services.observability import (
    LangfuseObservabilityProvider,
    LangfusePromptResolver,
    NoOpObservabilityProvider,
    ObservabilityStatus,
    make_prompt_resolver,
)
from src.services.prompts import LocalPromptResolver, PromptDefinition, fingerprint_prompt

DEFINITION = PromptDefinition(
    name="agent-guardrail",
    fallback_content='Local template. Return {"score": <0-100>}.',
    required_markers=('{"score"',),
)
MANAGED = 'Managed template v7. Return {"score": <0-100>}.'


def managed_prompt(content: object = MANAGED, version: object = 7, **extra) -> SimpleNamespace:
    return SimpleNamespace(prompt=content, version=version, labels=["production"], is_fallback=False, **extra)


def resolver_for(result=None, *, error: Exception | None = None, timeout_seconds: float = 5) -> tuple:
    client = Mock()
    client.get_prompt = Mock(return_value=result, side_effect=error)
    return LangfusePromptResolver(client=client, timeout_seconds=timeout_seconds, cache_ttl_seconds=30), client


@pytest.mark.anyio
async def test_managed_prompt_is_used_with_its_version_label_and_our_fingerprint() -> None:
    resolver, client = resolver_for(managed_prompt())

    prompt = await resolver.resolve(DEFINITION, label="production")

    assert (prompt.name, prompt.content, prompt.source) == ("agent-guardrail", MANAGED, "langfuse")
    assert (prompt.version, prompt.label) == ("langfuse-v7", "production")
    assert prompt.fingerprint == fingerprint_prompt(MANAGED)
    assert prompt.identity.fingerprint != fingerprint_prompt(DEFINITION.fallback_content)
    client.get_prompt.assert_called_once_with(
        "agent-guardrail",
        label="production",
        type="text",
        cache_ttl_seconds=30,
        max_retries=0,
        fetch_timeout_seconds=5,
    )


@pytest.mark.anyio
async def test_configured_label_is_requested_and_retained() -> None:
    resolver, client = resolver_for(managed_prompt())

    prompt = await resolver.resolve(DEFINITION, label="staging")

    assert prompt.label == "staging"
    assert client.get_prompt.call_args.kwargs["label"] == "staging"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [RuntimeError("private sdk detail"), TimeoutError("private"), ConnectionError("private"), KeyError("missing")],
)
async def test_fetch_failure_falls_back_to_local_without_raising(error: Exception, caplog) -> None:
    resolver, _ = resolver_for(error=error)

    prompt = await resolver.resolve(DEFINITION)

    assert (prompt.content, prompt.version, prompt.source) == (DEFINITION.fallback_content, "local-v1", "local")
    assert prompt.label == "production"
    assert "private" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "managed",
    [
        None,
        managed_prompt(content=""),
        managed_prompt(content="   "),
        managed_prompt(content=[{"role": "system", "content": MANAGED}]),
        managed_prompt(content=None),
        managed_prompt(version=None),
        managed_prompt(version="7"),
        managed_prompt(version=0),
        managed_prompt(version=True),
        managed_prompt(content="A managed template that dropped the output contract."),
        SimpleNamespace(prompt=MANAGED, version=3, is_fallback=True),
        SimpleNamespace(),
    ],
)
async def test_malformed_managed_prompt_falls_back_to_local(managed) -> None:
    resolver, _ = resolver_for(managed)

    prompt = await resolver.resolve(DEFINITION)

    assert prompt.source == "local"
    assert prompt.content == DEFINITION.fallback_content


@pytest.mark.anyio
async def test_slow_fetch_is_bounded_and_falls_back() -> None:
    client = Mock()
    client.get_prompt = Mock(side_effect=lambda *args, **kwargs: time.sleep(0.2) or managed_prompt())
    resolver = LangfusePromptResolver(client=client, timeout_seconds=0.01)

    prompt = await resolver.resolve(DEFINITION)

    assert prompt.source == "local"


@pytest.mark.parametrize(("timeout", "ttl"), [(0, 60), (-1, 60), (5, -1), (5, True), (5, 1.5)])
def test_invalid_resolver_bounds_are_rejected(timeout, ttl) -> None:
    with pytest.raises(ValueError):
        LangfusePromptResolver(client=Mock(), timeout_seconds=timeout, cache_ttl_seconds=ttl)


def langfuse_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        debug=False,
        langfuse_enabled=True,
        langfuse_public_key="pk-test-only",
        langfuse_secret_key="obvious-fake-secret",
        **overrides,
    )


def test_prompt_management_is_off_by_default_and_uses_local_prompts() -> None:
    provider = LangfuseObservabilityProvider(client=Mock(), shutdown_timeout_seconds=5)

    assert Settings(_env_file=None).langfuse_prompt_management_enabled is False
    assert Settings(_env_file=None).langfuse_prompt_label == "production"
    assert isinstance(make_prompt_resolver(langfuse_settings(), provider), LocalPromptResolver)
    assert isinstance(
        make_prompt_resolver(
            Settings(_env_file=None, langfuse_prompt_management_enabled=True), NoOpObservabilityProvider()
        ),
        LocalPromptResolver,
    )
    unavailable = NoOpObservabilityProvider(status=ObservabilityStatus.UNAVAILABLE)
    assert isinstance(
        make_prompt_resolver(langfuse_settings(langfuse_prompt_management_enabled=True), unavailable),
        LocalPromptResolver,
    )


@pytest.mark.anyio
async def test_enabled_prompt_management_shares_the_provider_client_and_timeout() -> None:
    client = Mock()
    client.get_prompt = Mock(return_value=managed_prompt())
    provider = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=7)
    settings = langfuse_settings(langfuse_prompt_management_enabled=True, langfuse_prompt_cache_ttl_seconds=120)

    resolver = make_prompt_resolver(settings, provider)
    prompt = await resolver.resolve(DEFINITION, label=settings.langfuse_prompt_label)

    assert isinstance(resolver, LangfusePromptResolver)
    assert prompt.source == "langfuse"
    assert client.get_prompt.call_args.kwargs["cache_ttl_seconds"] == 120
    assert client.get_prompt.call_args.kwargs["fetch_timeout_seconds"] == 7


def test_resolver_construction_failure_falls_back_to_local() -> None:
    provider = Mock()
    provider.status = ObservabilityStatus.CONFIGURED
    provider.make_prompt_resolver.side_effect = RuntimeError("down")

    resolver = make_prompt_resolver(langfuse_settings(langfuse_prompt_management_enabled=True), provider)

    assert isinstance(resolver, LocalPromptResolver)


@pytest.mark.parametrize("label", ["", "   "])
def test_blank_prompt_label_setting_is_rejected(label: str) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, langfuse_prompt_label=label)
