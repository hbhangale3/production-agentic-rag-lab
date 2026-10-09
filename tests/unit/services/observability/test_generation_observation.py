from unittest.mock import Mock

import pytest
from src.services.llm.base import LLMCompletion
from src.services.observability import (
    GenerationCost,
    LangfuseObservabilityProvider,
    LLMTelemetry,
    NoOpObservabilityProvider,
    SafeObservation,
    TokenCostRates,
    observed_completion,
)
from src.services.prompts import PromptIdentity

PROMPT = PromptIdentity(name="agent-guardrail", version="local-v1", label="production", fingerprint="f" * 64)
PRIVATE = "PRIVATE_QUESTION_SENTINEL"


class RecordingObservation:
    def __init__(self, name: str = "root", kind: str = "span", metadata=None, model=None) -> None:
        self.name, self.kind, self.model = name, kind, model
        self.metadata = [metadata or {}]
        self.error_types: list[str] = []
        self.generation_records: list[dict] = []
        self.children: list[RecordingObservation] = []
        self.end_count = 0

    def start_span(self, *, name, metadata=None):
        child = RecordingObservation(name, "span", metadata)
        self.children.append(child)
        return child

    def start_generation(self, *, name, model=None, metadata=None):
        child = RecordingObservation(name, "generation", metadata, model)
        self.children.append(child)
        return child

    def record_generation(self, **kwargs) -> None:
        self.generation_records.append(kwargs)

    def update(self, *, metadata=None, error_type=None) -> None:
        if metadata:
            self.metadata.append(metadata)
        if error_type:
            self.error_types.append(error_type)

    def end(self) -> None:
        self.end_count += 1


class FakeLLM:
    def __init__(self, completion=None, error: Exception | None = None) -> None:
        self.completion = completion or LLMCompletion(
            content="PRIVATE_ANSWER_SENTINEL", model="test-model", prompt_tokens=100, completion_tokens=25
        )
        self.error = error
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return self.completion


def fake_clock(*ticks: float):
    values = iter(ticks)
    return lambda: next(values)


def test_langfuse_adapter_creates_a_true_generation_observation() -> None:
    client, root, generation = Mock(), Mock(), Mock()
    client.start_observation.return_value = root
    root.start_observation.return_value = generation
    provider = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=5)

    trace = provider.start_trace(name="rag.request")
    trace.start_span(name="rag.retrieval", metadata={"count": 2})
    child = trace.start_generation(name="rag.generation", model="test-model", metadata={"prompt_name": "x"})

    assert [call.kwargs["as_type"] for call in root.start_observation.call_args_list] == ["span", "generation"]
    assert root.start_observation.call_args_list[1].kwargs == {
        "name": "rag.generation",
        "as_type": "generation",
        "model": "test-model",
        "metadata": {"prompt_name": "x"},
    }
    assert "input" not in root.start_observation.call_args_list[1].kwargs
    assert "output" not in root.start_observation.call_args_list[1].kwargs
    child.end()
    generation.end.assert_called_once()


def test_langfuse_adapter_maps_usage_and_known_cost_without_content() -> None:
    client, root, generation = Mock(), Mock(), Mock()
    client.start_observation.return_value = root
    root.start_observation.return_value = generation
    child = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=5).start_trace(
        name="rag.request"
    ).start_generation(name="rag.generation")

    child.record_generation(
        model="test-model", input_tokens=100, output_tokens=25, cost=GenerationCost(input_cost=0.001, output_cost=0.002)
    )

    generation.update.assert_called_once_with(
        model="test-model",
        usage_details={"input": 100, "output": 25, "total": 125},
        cost_details={"input": 0.001, "output": 0.002, "total": 0.003},
    )


def test_langfuse_adapter_omits_unknown_usage_and_cost() -> None:
    client, root, generation = Mock(), Mock(), Mock()
    client.start_observation.return_value = root
    root.start_observation.return_value = generation
    child = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=5).start_trace(
        name="rag.request"
    ).start_generation(name="rag.generation")

    child.record_generation(model="test-model", input_tokens=None, output_tokens=None, cost=None)
    child.record_generation(model=None, input_tokens=100, output_tokens=None, cost=None)
    child.record_generation()

    assert [call.kwargs for call in generation.update.call_args_list] == [
        {"model": "test-model"},
        {"usage_details": {"input": 100}},
    ]


def test_langfuse_generation_failures_are_fail_open() -> None:
    client, root = Mock(), Mock()
    client.start_observation.return_value = root
    root.start_observation.side_effect = RuntimeError("generation unavailable")
    trace = LangfuseObservabilityProvider(client=client, shutdown_timeout_seconds=5).start_trace(name="rag.request")

    inert = trace.start_generation(name="rag.generation")
    inert.record_generation(model="m", input_tokens=1, output_tokens=1)
    inert.update(metadata={"ok": True})
    inert.end()

    broken = Mock()
    broken.update.side_effect = RuntimeError("update unavailable")
    root.start_observation.side_effect = None
    root.start_observation.return_value = broken
    trace.start_generation(name="rag.generation").record_generation(model="m", input_tokens=1, output_tokens=1)


def test_noop_and_safe_observations_support_generations() -> None:
    trace = NoOpObservabilityProvider().start_trace(name="rag.request")
    generation = trace.start_generation(name="rag.generation", model="m", metadata={"a": 1})
    generation.record_generation(model="m", input_tokens=1, output_tokens=2, cost=None)
    generation.end()

    class SpanOnly:
        def __init__(self) -> None:
            self.spans = []

        def start_span(self, *, name, metadata=None):
            self.spans.append((name, metadata))
            return self

        def update(self, **kwargs) -> None:
            return None

        def end(self) -> None:
            return None

    span_only = SpanOnly()
    safe = SafeObservation(span_only).start_generation(name="rag.generation", metadata={"a": 1})
    safe.record_generation(model="m", input_tokens=1, output_tokens=2)
    assert span_only.spans == [("rag.generation", {"a": 1})]

    exploding = Mock()
    exploding.start_generation.side_effect = RuntimeError("down")
    SafeObservation(exploding).start_generation(name="rag.generation").record_generation(model="m")


@pytest.mark.anyio
async def test_observed_completion_records_model_provider_usage_latency_and_prompt_identity() -> None:
    root, llm = RecordingObservation(), FakeLLM()
    telemetry = LLMTelemetry(provider_name="test-provider", clock=fake_clock(10.0, 10.25))

    completion = await observed_completion(
        llm,
        [f"system {PRIVATE}", f"user {PRIVATE}"],
        temperature=0.0,
        max_tokens=32,
        observation=root,
        name="agent.guardrail",
        prompt=PROMPT,
        telemetry=telemetry,
        metadata={"grounding_attempt": 2},
    )

    assert completion is llm.completion
    assert llm.calls == [([f"system {PRIVATE}", f"user {PRIVATE}"], {"temperature": 0.0, "max_tokens": 32})]
    (generation,) = root.children
    assert (generation.name, generation.kind, generation.end_count) == ("agent.guardrail", "generation", 1)
    assert generation.metadata[0] == {
        "grounding_attempt": 2,
        "provider": "test-provider",
        "prompt_name": "agent-guardrail",
        "prompt_version": "local-v1",
        "prompt_label": "production",
        "prompt_fingerprint": "f" * 64,
    }
    assert generation.generation_records == [
        {"model": "test-model", "input_tokens": 100, "output_tokens": 25, "cost": None}
    ]
    assert generation.metadata[1] == {
        "model": "test-model",
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "latency_ms": 250.0,
    }
    recorded = repr((generation.metadata, generation.generation_records))
    assert PRIVATE not in recorded and "PRIVATE_ANSWER_SENTINEL" not in recorded


@pytest.mark.anyio
async def test_cost_is_recorded_only_from_explicitly_configured_rates() -> None:
    rates = TokenCostRates(input_per_million=2.0, output_per_million=8.0)
    priced, unpriced = RecordingObservation(), RecordingObservation()

    await observed_completion(
        FakeLLM(), [], temperature=0, max_tokens=1, observation=priced, name="g",
        telemetry=LLMTelemetry(cost_rates=rates, clock=fake_clock(0.0, 0.0)),
    )
    await observed_completion(
        FakeLLM(), [], temperature=0, max_tokens=1, observation=unpriced, name="g",
        telemetry=LLMTelemetry(clock=fake_clock(0.0, 0.0)),
    )

    cost = priced.children[0].generation_records[0]["cost"]
    assert cost == GenerationCost(input_cost=0.0002, output_cost=0.0002)
    assert cost.total_cost == pytest.approx(0.0004)
    assert priced.children[0].metadata[1]["cost_usd"] == pytest.approx(0.0004)
    assert unpriced.children[0].generation_records[0]["cost"] is None
    assert "cost_usd" not in unpriced.children[0].metadata[1]
    assert LLMTelemetry().cost_rates is None


@pytest.mark.anyio
async def test_cost_is_not_invented_when_the_provider_reports_no_usage() -> None:
    root = RecordingObservation()
    llm = FakeLLM(LLMCompletion(content="x", model="test-model"))
    telemetry = LLMTelemetry(
        cost_rates=TokenCostRates(input_per_million=2.0, output_per_million=8.0), clock=fake_clock(0.0, 0.0)
    )

    await observed_completion(llm, [], temperature=0, max_tokens=1, observation=root, name="g", telemetry=telemetry)

    assert root.children[0].generation_records == [
        {"model": "test-model", "input_tokens": None, "output_tokens": None, "cost": None}
    ]
    assert "cost_usd" not in root.children[0].metadata[1]
    assert TokenCostRates(input_per_million=1, output_per_million=1).cost(input_tokens=5, output_tokens=None) is None


@pytest.mark.parametrize("rates", [(-1, 1), (1, -1), (True, 1), ("1", 1)])
def test_invalid_cost_rates_are_rejected(rates) -> None:
    with pytest.raises(ValueError):
        TokenCostRates(input_per_million=rates[0], output_per_million=rates[1])


@pytest.mark.anyio
async def test_provider_failure_is_recorded_and_reraised_unchanged() -> None:
    root, error = RecordingObservation(), RuntimeError("private provider detail")

    with pytest.raises(RuntimeError) as caught:
        await observed_completion(
            FakeLLM(error=error), [], temperature=0, max_tokens=1, observation=root, name="agent.guardrail",
            telemetry=LLMTelemetry(clock=fake_clock(1.0, 1.5)),
        )

    assert caught.value is error
    generation = root.children[0]
    assert generation.error_types == ["RuntimeError"]
    assert generation.metadata[1] == {"latency_ms": 500.0}
    assert generation.generation_records == []
    assert generation.end_count == 1
    assert "private provider detail" not in repr(generation.metadata)


@pytest.mark.anyio
async def test_no_observation_means_no_telemetry_work_and_an_unchanged_call() -> None:
    llm, clock = FakeLLM(), Mock(side_effect=AssertionError("clock must not be read"))

    completion = await observed_completion(
        llm, ["m"], temperature=0.1, max_tokens=64, observation=None, name="g", telemetry=LLMTelemetry(clock=clock)
    )

    assert completion is llm.completion
    assert llm.calls == [(["m"], {"temperature": 0.1, "max_tokens": 64})]


@pytest.mark.anyio
async def test_broken_observation_and_clock_never_affect_the_completion() -> None:
    broken = Mock()
    broken.start_generation.side_effect = RuntimeError("down")
    broken.start_span.side_effect = RuntimeError("down")
    half_broken = Mock()
    half_broken.start_generation.return_value.record_generation.side_effect = RuntimeError("down")
    half_broken.start_generation.return_value.update.side_effect = RuntimeError("down")
    half_broken.start_generation.return_value.end.side_effect = RuntimeError("down")
    ticks = iter([1.0])

    def clock() -> float:
        try:
            return next(ticks)
        except StopIteration:
            raise RuntimeError("clock down") from None

    for observation in (broken, half_broken):
        llm = FakeLLM()
        completion = await observed_completion(
            llm, [], temperature=0, max_tokens=1, observation=observation, name="g"
        )
        assert completion is llm.completion
    llm = FakeLLM()
    assert await observed_completion(
        llm, [], temperature=0, max_tokens=1, observation=RecordingObservation(), name="g",
        telemetry=LLMTelemetry(clock=clock),
    ) is llm.completion
