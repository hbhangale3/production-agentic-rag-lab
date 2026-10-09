from src.services.observability.base import ObservabilityProvider, Observation
from src.services.observability.noop import NoOpObservation


class SafeObservation:
    """Fail-open guard around any provider implementation or test double."""

    def __init__(self, observation: Observation) -> None:
        self._observation = observation
        self._ended = False

    def start_span(self, *, name: str, metadata: dict[str, object] | None = None) -> "SafeObservation":
        try:
            return SafeObservation(self._observation.start_span(name=name, metadata=metadata))
        except Exception:
            return SafeObservation(NoOpObservation())

    def update(self, *, metadata: dict[str, object] | None = None, error_type: str | None = None) -> None:
        try:
            self._observation.update(metadata=metadata, error_type=error_type)
        except Exception:
            return None

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self._observation.end()
        except Exception:
            return None


def safe_start_trace(
    provider: ObservabilityProvider,
    *,
    name: str,
    metadata: dict[str, object] | None = None,
) -> SafeObservation:
    try:
        return SafeObservation(provider.start_trace(name=name, metadata=metadata))
    except Exception:
        return SafeObservation(NoOpObservation())


def safe_content_capture(provider: ObservabilityProvider) -> bool:
    try:
        return bool(provider.capture_content)
    except Exception:
        return False
