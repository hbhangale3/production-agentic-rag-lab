"""One model call observed as a generation: model, provider-reported usage, latency, known cost."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.services.observability.base import Observation, TokenCostRates
from src.services.observability.safe import SafeObservation
from src.services.prompts.base import PromptIdentity


@dataclass(frozen=True)
class LLMTelemetry:
    """Static facts about the configured model that telemetry cannot learn from a completion."""

    provider_name: str | None = None
    cost_rates: TokenCostRates | None = None
    clock: Callable[[], float] = field(default=time.perf_counter, compare=False)


def start_generation_observation(
    observation: Observation | None,
    *,
    name: str,
    prompt: PromptIdentity | None,
    telemetry: LLMTelemetry,
    metadata: dict[str, object] | None = None,
) -> SafeObservation | None:
    """Start a fail-open generation child carrying prompt identity and no content."""

    if observation is None:
        return None
    details: dict[str, object] = dict(metadata or {})
    if telemetry.provider_name:
        details["provider"] = telemetry.provider_name
    if prompt is not None:
        details.update(prompt.as_metadata())
    return SafeObservation(observation).start_generation(name=name, metadata=details)


def finish_generation_observation(
    generation: SafeObservation | None,
    *,
    telemetry: LLMTelemetry,
    started_at: float,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    """Record provider-reported usage, measured latency, and cost only when rates are configured."""

    if generation is None:
        return
    cost = (
        telemetry.cost_rates.cost(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        if telemetry.cost_rates is not None
        else None
    )
    generation.record_generation(
        model=model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cost=cost,
    )
    metadata: dict[str, object] = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": elapsed_ms(telemetry, started_at),
    }
    if cost is not None:
        metadata["cost_usd"] = cost.total_cost
    generation.update(metadata=metadata)
    generation.end()


def fail_generation_observation(
    generation: SafeObservation | None,
    *,
    telemetry: LLMTelemetry,
    started_at: float,
    error_type: str,
) -> None:
    if generation is None:
        return
    generation.update(metadata={"latency_ms": elapsed_ms(telemetry, started_at)}, error_type=error_type)
    generation.end()


def elapsed_ms(telemetry: LLMTelemetry, started_at: float) -> float:
    try:
        return max(0.0, round((telemetry.clock() - started_at) * 1000, 3))
    except Exception:
        return 0.0


async def observed_completion(
    llm_provider: Any,
    messages: Sequence[Any],
    *,
    temperature: float | None,
    max_tokens: int | None,
    observation: Observation | None,
    name: str,
    prompt: PromptIdentity | None = None,
    telemetry: LLMTelemetry | None = None,
    metadata: dict[str, object] | None = None,
) -> Any:
    """Call ``llm_provider.complete`` unchanged; telemetry problems never affect the call."""

    telemetry = telemetry or LLMTelemetry()
    generation = start_generation_observation(
        observation, name=name, prompt=prompt, telemetry=telemetry, metadata=metadata
    )
    started_at = telemetry.clock() if generation is not None else 0.0
    try:
        completion = await llm_provider.complete(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        fail_generation_observation(
            generation, telemetry=telemetry, started_at=started_at, error_type=type(exc).__name__
        )
        raise
    finish_generation_observation(
        generation,
        telemetry=telemetry,
        started_at=started_at,
        model=getattr(completion, "model", None),
        prompt_tokens=getattr(completion, "prompt_tokens", None),
        completion_tokens=getattr(completion, "completion_tokens", None),
    )
    return completion
