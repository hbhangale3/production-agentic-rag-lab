from enum import StrEnum
from typing import Protocol


class ObservabilityStatus(StrEnum):
    DISABLED = "disabled"
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"


class Observation(Protocol):
    """Provider-neutral handle for one trace or nested operation."""

    def start_span(self, *, name: str, metadata: dict[str, object] | None = None) -> "Observation": ...

    def update(self, *, metadata: dict[str, object] | None = None, error_type: str | None = None) -> None: ...

    def end(self) -> None: ...


class ObservabilityProvider(Protocol):
    """Small explicit tracing boundary for future RAG instrumentation."""

    @property
    def status(self) -> ObservabilityStatus: ...

    @property
    def capture_content(self) -> bool: ...

    def start_trace(self, *, name: str, metadata: dict[str, object] | None = None) -> Observation: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...
