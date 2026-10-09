from src.services.observability.base import ObservabilityStatus, Observation


class NoOpObservation:
    """Deterministic inert observation used when telemetry is disabled."""

    def start_span(self, *, name: str, metadata: dict[str, object] | None = None) -> "NoOpObservation":
        return self

    def update(self, *, metadata: dict[str, object] | None = None, error_type: str | None = None) -> None:
        return None

    def end(self) -> None:
        return None


class NoOpObservabilityProvider:
    def __init__(self, *, status: ObservabilityStatus = ObservabilityStatus.DISABLED) -> None:
        self._status = status
        self._observation = NoOpObservation()

    @property
    def status(self) -> ObservabilityStatus:
        return self._status

    @property
    def capture_content(self) -> bool:
        return False

    def start_trace(self, *, name: str, metadata: dict[str, object] | None = None) -> Observation:
        return self._observation

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None
