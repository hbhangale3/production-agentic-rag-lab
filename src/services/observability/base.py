from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ObservabilityStatus(StrEnum):
    DISABLED = "disabled"
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GenerationCost:
    """Cost of one model call in US dollars, present only when it is actually known."""

    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass(frozen=True)
class TokenCostRates:
    """Explicitly configured prices per million tokens; never a guessed default."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        for value in (self.input_per_million, self.output_per_million):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("token cost rates must be non-negative numbers")

    def cost(self, *, input_tokens: int | None, output_tokens: int | None) -> GenerationCost | None:
        """Return a cost only when the provider reported both token counts."""

        if input_tokens is None or output_tokens is None:
            return None
        return GenerationCost(
            input_cost=input_tokens * self.input_per_million / 1_000_000,
            output_cost=output_tokens * self.output_per_million / 1_000_000,
        )


class Observation(Protocol):
    """Provider-neutral handle for one trace or nested operation."""

    def start_span(self, *, name: str, metadata: dict[str, object] | None = None) -> "Observation": ...

    def start_generation(
        self,
        *,
        name: str,
        model: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "Observation":
        """Start a child observation that represents one model call."""
        ...

    def record_generation(
        self,
        *,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: GenerationCost | None = None,
    ) -> None:
        """Attach provider-reported model, usage, and known cost to a generation observation."""
        ...

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
