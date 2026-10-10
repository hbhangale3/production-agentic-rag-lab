import asyncio
import logging
from typing import Any

from src.services.observability.base import GenerationCost, ObservabilityStatus, Observation
from src.services.observability.langfuse_prompts import LangfusePromptResolver

logger = logging.getLogger(__name__)


class LangfuseObservation:
    """Thin fail-open wrapper that prevents SDK types escaping the adapter."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation
        self._ended = False

    def start_span(self, *, name: str, metadata: dict[str, object] | None = None) -> Observation:
        try:
            child = self._observation.start_observation(name=name, as_type="span", metadata=metadata)
        except Exception:
            logger.warning("Observability span creation failed; continuing without child telemetry")
            return _INERT_OBSERVATION
        return LangfuseObservation(child)

    def start_generation(
        self,
        *,
        name: str,
        model: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Observation:
        try:
            child = self._observation.start_observation(
                name=name,
                as_type="generation",
                model=model,
                metadata=metadata,
            )
        except Exception:
            logger.warning("Observability generation creation failed; continuing without child telemetry")
            return _INERT_OBSERVATION
        return LangfuseObservation(child)

    def record_generation(
        self,
        *,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: GenerationCost | None = None,
    ) -> None:
        try:
            kwargs: dict[str, object] = {}
            if model is not None:
                kwargs["model"] = model
            usage = {key: value for key, value in (("input", input_tokens), ("output", output_tokens)) if value is not None}
            if input_tokens is not None and output_tokens is not None:
                usage["total"] = input_tokens + output_tokens
            if usage:
                kwargs["usage_details"] = usage
            if cost is not None:
                kwargs["cost_details"] = {
                    "input": cost.input_cost,
                    "output": cost.output_cost,
                    "total": cost.total_cost,
                }
            if kwargs:
                self._observation.update(**kwargs)
        except Exception:
            logger.warning("Observability generation update failed; continuing without usage telemetry")

    def update(self, *, metadata: dict[str, object] | None = None, error_type: str | None = None) -> None:
        try:
            kwargs: dict[str, object] = {"metadata": metadata}
            if error_type is not None:
                kwargs.update(level="ERROR", status_message=error_type)
            self._observation.update(**kwargs)
        except Exception:
            logger.warning("Observability update failed; continuing without telemetry update")

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self._observation.end()
        except Exception:
            logger.warning("Observability completion failed; continuing without telemetry completion")


class _InertObservation:
    def start_span(self, *, name: str, metadata: dict[str, object] | None = None) -> Observation:
        return self

    def start_generation(
        self,
        *,
        name: str,
        model: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Observation:
        return self

    def record_generation(self, **kwargs: object) -> None:
        return None

    def update(self, *, metadata: dict[str, object] | None = None, error_type: str | None = None) -> None:
        return None

    def end(self) -> None:
        return None


_INERT_OBSERVATION = _InertObservation()


class LangfuseObservabilityProvider:
    """Worker-scoped adapter around the Langfuse v4 SDK and its batch worker."""

    def __init__(
        self,
        *,
        client: Any,
        shutdown_timeout_seconds: float,
        capture_content: bool = False,
        owns_client: bool = True,
    ) -> None:
        self._client = client
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._owns_client = owns_client
        self._capture_content = capture_content
        self._closed = False

    @property
    def status(self) -> ObservabilityStatus:
        return ObservabilityStatus.UNAVAILABLE if self._closed else ObservabilityStatus.CONFIGURED

    @property
    def capture_content(self) -> bool:
        return self._capture_content

    def start_trace(self, *, name: str, metadata: dict[str, object] | None = None) -> Observation:
        try:
            observation = self._client.start_observation(name=name, as_type="span", metadata=metadata)
        except Exception:
            logger.warning("Observability trace creation failed; continuing without telemetry")
            return _INERT_OBSERVATION
        return LangfuseObservation(observation)

    def make_prompt_resolver(self, *, cache_ttl_seconds: int = 60) -> LangfusePromptResolver:
        """Managed-prompt resolver sharing this provider's client and timeout."""
        return LangfusePromptResolver(
            client=self._client,
            timeout_seconds=self._shutdown_timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def flush(self) -> None:
        if self._closed:
            return
        await self._best_effort(self._client.flush, "flush")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._owns_client:
            return
        await self._best_effort(self._client.shutdown, "shutdown")

    async def _best_effort(self, operation: Any, label: str) -> None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(operation),
                timeout=self._shutdown_timeout_seconds,
            )
        except Exception:
            logger.warning("Observability %s failed; application lifecycle will continue", label)
