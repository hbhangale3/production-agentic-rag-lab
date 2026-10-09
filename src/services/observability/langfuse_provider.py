import asyncio
import logging
from typing import Any

from src.services.observability.base import ObservabilityStatus, Observation

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
