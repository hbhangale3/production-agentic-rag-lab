from src.services.observability.base import GenerationCost, ObservabilityProvider, Observation
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

    def start_generation(
        self,
        *,
        name: str,
        model: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "SafeObservation":
        try:
            start_generation = getattr(self._observation, "start_generation", None)
            if start_generation is None:
                # An observation without model-call support still records the operation as a span.
                return SafeObservation(self._observation.start_span(name=name, metadata=metadata))
            return SafeObservation(start_generation(name=name, model=model, metadata=metadata))
        except Exception:
            return SafeObservation(NoOpObservation())

    def record_generation(
        self,
        *,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: GenerationCost | None = None,
    ) -> None:
        try:
            self._observation.record_generation(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        except Exception:
            return None

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
