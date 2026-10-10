import logging
from typing import Any

from langfuse import Langfuse
from src.config import Settings
from src.services.observability.base import ObservabilityProvider, ObservabilityStatus
from src.services.observability.langfuse_provider import LangfuseObservabilityProvider
from src.services.observability.noop import NoOpObservabilityProvider
from src.services.prompts.base import LocalPromptResolver, PromptResolver

logger = logging.getLogger(__name__)


def make_observability_provider(
    settings: Settings,
    *,
    client: Any | None = None,
) -> ObservabilityProvider:
    """Create optional worker-scoped telemetry without testing remote reachability."""
    if not settings.langfuse_enabled:
        logger.info("Langfuse observability disabled")
        return NoOpObservabilityProvider()
    try:
        owned_client = (
            client
            if client is not None
            else Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_host,
                timeout=settings.langfuse_timeout_seconds,
                environment=settings.environment,
                tracing_enabled=True,
            )
        )
    except Exception:
        logger.warning("Langfuse client construction failed; continuing without observability")
        return NoOpObservabilityProvider(status=ObservabilityStatus.UNAVAILABLE)
    logger.info("Langfuse observability configured")
    return LangfuseObservabilityProvider(
        client=owned_client,
        shutdown_timeout_seconds=settings.langfuse_timeout_seconds,
        capture_content=settings.langfuse_capture_content,
        owns_client=client is None,
    )


def make_prompt_resolver(settings: Settings, provider: ObservabilityProvider) -> PromptResolver:
    """Managed prompts only when explicitly enabled and Langfuse is configured; otherwise local."""
    make_resolver = getattr(provider, "make_prompt_resolver", None)
    if (
        not settings.langfuse_enabled
        or not settings.langfuse_prompt_management_enabled
        or provider.status is not ObservabilityStatus.CONFIGURED
        or make_resolver is None
    ):
        return LocalPromptResolver()
    try:
        return make_resolver(cache_ttl_seconds=settings.langfuse_prompt_cache_ttl_seconds)
    except Exception:
        logger.warning("Managed prompt resolver construction failed; using local prompts")
        return LocalPromptResolver()
