import asyncio
import dataclasses
import json

import pytest
from src.config import Settings
from src.schemas.agent import AgentSource
from src.services.agent import (
    AGENT_PROMPT_DEFINITIONS,
    AgentGraphConfig,
    TerminalReason,
    build_agent_graph,
    create_initial_agent_state,
    local_agent_prompt_bundle,
)
from src.services.agent.response_cache import (
    AgentCachedAnswer,
    AgentResponseCache,
    deserialize_agent_answer,
    serialize_agent_answer,
)
from src.services.agent.service import (
    AgentExecutionFailure,
    AgentFailureKind,
    AgentService,
    is_cacheable,
    map_agent_state,
)
from src.services.agent.status import (
    ANSWER_NOT_VALIDATED_MESSAGE,
    GROUNDING_FAILED_MESSAGE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
)
from src.services.cache.base import CacheHealth, CacheReadResult, CacheReadStatus
from src.services.prompts import LocalPromptResolver, ResolvedPrompt
from src.services.rag import RAG_SYSTEM_PROMPT

from tests.unit.services.agent.test_graph import (
    ANSWER,
    EVIDENCE_TEXT,
    GROUNDING_SYSTEM_PROMPT,
    HIGH,
    INDIA_QUESTION,
    LOW,
    REGENERATED_ANSWER,
    REWRITE_JSON,
    SELECT_TWO,
    FakeLLMProvider,
    RecordingObservation,
    Stopped,
    Truncated,
    dependencies,
    exhausted_dependencies,
    live_result,
    make_hit,
    retrieval_result,
)

QUESTION = "PRIVATE_QUESTION_SENTINEL how is NLP used in clinical decision support?"
LOCAL_ONLY = AgentGraphConfig(live_fallback_enabled=False, max_local_retrieval_attempts=1)


class RepeatingLLM(FakeLLMProvider):
    """The same script for every graph run, so one service can answer several requests."""

    def __init__(self, *responses, **kwargs) -> None:
        super().__init__(*responses, **kwargs)
        self._script = list(self.responses)

    async def complete(self, messages, **kwargs):
        system = messages[0].content
        is_classifier = not system.startswith(RAG_SYSTEM_PROMPT) and not system.startswith(GROUNDING_SYSTEM_PROMPT)
        if is_classifier and len(self.calls) >= len(self.responses):
            self.responses = self.responses + self._script
        return await super().complete(messages, **kwargs)


class FakeCache:
    """In-memory AsyncCache with switchable read and write failures."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.get_keys: list[str] = []
        self.set_calls: list[tuple[str, str, int]] = []
        self.read_error: Exception | None = None
        self.read_failure = False
        self.write_error: Exception | None = None
        self.write_result = True

    async def get(self, key: str) -> str | None:
        return (await self.get_result(key)).value

    async def get_result(self, key: str) -> CacheReadResult:
        self.get_keys.append(key)
        if self.read_error:
            raise self.read_error
        if self.read_failure:
            return CacheReadResult(status=CacheReadStatus.FAILURE)
        value = self.values.get(key)
        return CacheReadResult(status=CacheReadStatus.MISS if value is None else CacheReadStatus.HIT, value=value)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        self.set_calls.append((key, value, ttl_seconds))
        if self.write_error:
            raise self.write_error
        if self.write_result:
            self.values[key] = value
        return self.write_result

    async def delete(self, key: str) -> bool:
        return self.values.pop(key, None) is not None

    async def ping(self) -> bool:
        return True

    async def health(self) -> CacheHealth:
        return CacheHealth.HEALTHY

    async def close(self) -> None:
        return None


class CountingResolver(LocalPromptResolver):
    def __init__(self, overrides: dict[str, ResolvedPrompt] | None = None) -> None:
        self.calls = 0
        self.overrides = overrides or {}

    async def resolve(self, definition, *, label="production"):
        self.calls += 1
        return self.overrides.get(definition.name) or await super().resolve(definition, label=label)


def make_service(
    deps,
    *,
    cache: FakeCache | None = None,
    config: AgentGraphConfig | None = None,
    redis_enabled: bool = True,
    resolver=None,
    request_timeout_seconds: float = 30.0,
    clock=None,
    **settings_overrides,
) -> tuple[AgentService, FakeCache]:
    cache = cache or FakeCache()
    graph_config = config or AgentGraphConfig()
    settings = Settings(_env_file=None, debug=False, redis_enabled=redis_enabled, **settings_overrides)
    kwargs = {"clock": clock} if clock is not None else {}
    service = AgentService(
        dependencies=deps,
        graph_config=graph_config,
        prompt_resolver=resolver or LocalPromptResolver(),
        response_cache=AgentResponseCache(cache=cache, settings=settings, graph_config=graph_config),
        request_timeout_seconds=request_timeout_seconds,
        **kwargs,
    )
    return service, cache


def stored_envelope(cache: FakeCache) -> dict:
    ((_, value, _),) = cache.set_calls
    return json.loads(value)


@pytest.mark.anyio
async def test_grounded_miss_is_cached_and_exact_repeat_is_a_hit_that_skips_the_graph() -> None:
    deps, provider, retrieval = dependencies()
    service, cache = make_service(deps)

    miss = await service.answer(QUESTION)
    calls_after_miss = (len(provider.calls), len(provider.generation_calls), len(provider.grounding_calls))
    hit = await service.answer(QUESTION)

    assert (miss.cache_status, miss.outcome, miss.grounding_passed, miss.terminal_reason) == ("miss", "answered", True, None)
    assert miss.answer == ANSWER
    assert (miss.model, miss.prompt_tokens, miss.completion_tokens) == ("fake-model", 111, 22)
    assert miss.pipeline_version == "v1"
    assert miss.execution.model_dump() == {
        "retrieval_attempts": 1,
        "query_rewritten": False,
        "live_fallback_used": False,
        "live_papers_selected": 0,
        "generation_attempts": 1,
        "grounding_attempts": 1,
        "source_count": 1,
    }
    assert [status.code for status in miss.statuses][:3] == ["graph_started", "guardrail_started", "guardrail_passed"]
    assert miss.statuses[-1].code == "graph_completed"
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0][0].startswith("rag:agent-response:v1:")
    assert cache.set_calls[0][2] == 86_400

    assert hit.cache_status == "hit"
    assert (hit.answer, hit.sources, hit.model, hit.prompt_tokens, hit.completion_tokens) == (
        miss.answer,
        miss.sources,
        miss.model,
        miss.prompt_tokens,
        miss.completion_tokens,
    )
    assert (hit.outcome, hit.grounding_passed, hit.terminal_reason, hit.execution) == ("answered", True, None, miss.execution)
    assert [(status.code, status.message) for status in hit.statuses] == [("cache_hit", "Validated cached answer found")]
    assert (len(provider.calls), len(provider.generation_calls), len(provider.grounding_calls)) == calls_after_miss
    retrieval.search.assert_called_once()
    assert len(deps.embedding_provider.passage_batches) == 1
    deps.live_search_service.search.assert_not_awaited()
    deps.live_document_processor.process.assert_not_awaited()
    assert cache.get_keys[0] == cache.get_keys[1] == cache.set_calls[0][0]
    assert len(cache.set_calls) == 1
    assert service.response_cache.stats_snapshot().hits == 1
    assert service.response_cache.stats_snapshot().writes == 1


@pytest.mark.anyio
async def test_cache_key_and_payload_contain_no_question_prompt_or_hidden_state() -> None:
    deps, _, _ = dependencies()
    service, cache = make_service(deps)

    await service.answer(QUESTION)

    key, value, _ = cache.set_calls[0]
    assert "PRIVATE_QUESTION_SENTINEL" not in key and "PRIVATE_QUESTION_SENTINEL" not in value
    envelope = stored_envelope(cache)
    assert set(envelope) == {"schema_version", "pipeline_version", "response"}
    assert (envelope["schema_version"], envelope["pipeline_version"]) == ("v1", "v1")
    assert set(envelope["response"]) == {
        "answer",
        "sources",
        "grounding_passed",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "execution",
    }
    assert envelope["response"]["grounding_passed"] is True
    for forbidden in ("You are", "UNTRUSTED_", "relevance_score", "embedding", "rrf_score", "reasoning"):
        assert forbidden not in value


@pytest.mark.anyio
async def test_sources_are_exactly_the_generation_sources_not_all_final_evidence() -> None:
    hits = [
        make_hit("c1", "first " + "alpha " * 40),
        make_hit("c2", "second " + "beta " * 40),
        make_hit("c3", "third " + "omega " * 400),
        make_hit("c4", "fourth passage never sent"),
    ]
    deps, _, _ = dependencies(result=retrieval_result(hits=hits), max_context_tokens=260)
    service, _ = make_service(deps)

    response = await service.answer(QUESTION)

    assert [source.citation for source in response.sources] == ["[S1]", "[S2]", "[S3]"]
    assert [source.chunk_id for source in response.sources] == ["c1", "c2", "c3"]
    assert response.sources[2].truncated is True
    assert all(source.source_type == "local" and source.source_url is None for source in response.sources)
    assert "never sent" not in response.model_dump_json()
    assert response.execution.source_count == 3


@pytest.mark.anyio
async def test_live_sources_keep_provenance_and_the_whole_path_is_summarised() -> None:
    deps, _, _, _ = exhausted_dependencies(live=live_result("2501.00001", "2501.00002"), selection=SELECT_TWO)
    service, cache = make_service(deps)

    response = await service.answer(INDIA_QUESTION)

    assert response.outcome == "answered"
    assert [(source.citation, source.source_type) for source in response.sources] == [
        ("[S1]", "local"),
        ("[S2]", "live_arxiv"),
        ("[S3]", "live_arxiv"),
    ]
    assert response.sources[1].source_url == "https://arxiv.org/pdf/2501.00001"
    assert response.sources[1].chunk_id == "live::2501.00001::chunk::000"
    assert response.execution.model_dump() == {
        "retrieval_attempts": 2,
        "query_rewritten": True,
        "live_fallback_used": True,
        "live_papers_selected": 2,
        "generation_attempts": 1,
        "grounding_attempts": 1,
        "source_count": 3,
    }
    messages = [status.message for status in response.statuses]
    for expected in (
        "Guardrail passed",
        "Local retrieval completed",
        "Evidence insufficient",
        "Query rewritten",
        "Local retry completed",
        "Live arXiv fallback used",
        "Live arXiv search found 2 candidate papers",
        "2 live papers selected",
        "Live document processing completed (2 papers usable)",
        "Evidence reranked (3 sources selected)",
        "Answer generated",
        "Grounding validation passed",
    ):
        assert expected in messages
    assert len(cache.set_calls) == 1


@pytest.mark.anyio
async def test_cache_disabled_runs_the_graph_and_reports_bypass() -> None:
    deps, provider, _ = dependencies(llm=RepeatingLLM())
    service, cache = make_service(deps, redis_enabled=False)

    first, second = await service.answer(QUESTION), await service.answer(QUESTION)

    assert (first.cache_status, second.cache_status) == ("bypass", "bypass")
    assert first.outcome == second.outcome == "answered"
    assert len(provider.generation_calls) == 2
    assert cache.get_keys == [] and cache.set_calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("llm", "outcome", "message", "grounding_passed"),
    [
        (FakeLLMProvider('{"score":10}'), "out_of_scope", OUT_OF_SCOPE_MESSAGE, None),
        (FakeLLMProvider('{"score":90}', '{"score":20}'), "insufficient_evidence", INSUFFICIENT_EVIDENCE_MESSAGE, None),
        (FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=LOW), "grounding_failed", GROUNDING_FAILED_MESSAGE, False),
        (FakeLLMProvider(answer="No citation at all."), "generation_failed", ANSWER_NOT_VALIDATED_MESSAGE, None),
    ],
)
async def test_semantic_outcomes_return_a_fixed_safe_message_and_are_never_cached(
    llm: FakeLLMProvider, outcome: str, message: str, grounding_passed
) -> None:
    deps, provider, _ = dependencies(llm=RepeatingLLM(*llm.responses, answer=llm.answers, grounding=llm.groundings))
    service, cache = make_service(deps, config=LOCAL_ONLY)

    response = await service.answer(QUESTION)

    assert (response.outcome, response.terminal_reason, response.answer) == (outcome, outcome, message)
    assert (response.grounding_passed, response.sources, response.cache_status) == (grounding_passed, [], "miss")
    assert (response.model, response.prompt_tokens, response.completion_tokens) == (None, None, None)
    assert response.execution.source_count == 0
    serialized = response.model_dump_json()
    assert ANSWER not in serialized and REGENERATED_ANSWER not in serialized
    assert "Private generated" not in serialized and "Private regenerated" not in serialized
    assert cache.set_calls == []
    await service.answer(QUESTION)
    assert cache.set_calls == [] and len(cache.get_keys) == 2


@pytest.mark.anyio
async def test_out_of_scope_does_no_retrieval_rerank_live_generation_or_grounding() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":10}'))
    service, _ = make_service(deps)

    response = await service.answer("What is a dog?")

    assert response.outcome == "out_of_scope"
    assert [status.code for status in response.statuses] == [
        "graph_started",
        "guardrail_started",
        "guardrail_rejected",
        "graph_completed",
    ]
    retrieval.search.assert_not_called()
    assert deps.embedding_provider.queries == []
    deps.live_search_service.search.assert_not_awaited()
    assert (len(provider.calls), provider.generation_calls, provider.grounding_calls) == (1, [], [])
    assert response.execution.model_dump()["retrieval_attempts"] == 0


@pytest.mark.anyio
async def test_regenerated_answer_is_the_public_answer_and_the_cached_one() -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH)))
    service, cache = make_service(deps)

    response = await service.answer(QUESTION)
    hit = await service.answer(QUESTION)

    assert response.answer == hit.answer == REGENERATED_ANSWER
    assert (response.execution.generation_attempts, response.execution.grounding_attempts) == (2, 2)
    assert ANSWER not in response.model_dump_json()
    assert ANSWER not in cache.set_calls[0][1]
    assert hit.cache_status == "hit"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("deps_factory", "kind"),
    [
        (lambda: dependencies(llm=FakeLLMProvider(RuntimeError("private failure detail"))), AgentFailureKind.INTERNAL_ERROR),
        (lambda: dependencies(retrieval_error=RuntimeError("private failure detail")), AgentFailureKind.RETRIEVAL_FAILED),
        (lambda: dependencies(llm=FakeLLMProvider(answer=RuntimeError("private failure detail"))), AgentFailureKind.GENERATION_FAILED),
        (lambda: dependencies(llm=FakeLLMProvider(grounding="not json")), AgentFailureKind.INTERNAL_ERROR),
    ],
)
async def test_execution_failures_raise_a_constrained_kind_and_cache_nothing(deps_factory, kind) -> None:
    deps, _, _ = deps_factory()
    service, cache = make_service(deps)

    with pytest.raises(AgentExecutionFailure) as caught:
        await service.answer(QUESTION)

    assert caught.value.kind is kind
    assert str(caught.value) == kind.value
    assert "private failure detail" not in repr(caught.value)
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_unexpected_graph_exception_becomes_internal_error() -> None:
    deps, _, _ = dependencies()
    broken = dataclasses.replace(deps, evidence_context_builder=None)
    service, cache = make_service(broken)

    with pytest.raises(AgentExecutionFailure) as caught:
        await service.answer(QUESTION)

    assert caught.value.kind is AgentFailureKind.INTERNAL_ERROR
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_request_deadline_is_a_controlled_timeout_without_partial_answer() -> None:
    class SlowLLM(FakeLLMProvider):
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(5)
            return await super().complete(messages, **kwargs)

    deps, _, _ = dependencies(llm=SlowLLM())
    service, cache = make_service(deps, request_timeout_seconds=0.05)
    seen = []

    async def on_status(status) -> None:
        seen.append(status.code)

    with pytest.raises(AgentExecutionFailure) as caught:
        await asyncio.wait_for(service.answer(QUESTION, on_status=on_status), timeout=5)

    assert caught.value.kind is AgentFailureKind.TIMEOUT
    assert seen == ["graph_started"]
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_deadline_covers_prompt_resolution_time() -> None:
    ticks = iter([0.0, 100.0, 100.0, 100.0])
    deps, provider, _ = dependencies()
    service, _ = make_service(deps, request_timeout_seconds=50.0, clock=lambda: next(ticks, 100.0))

    with pytest.raises(AgentExecutionFailure) as caught:
        await service.answer(QUESTION)

    assert caught.value.kind is AgentFailureKind.TIMEOUT
    assert provider.generation_calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "corrupt",
    [
        "not json",
        "[]",
        '{"schema_version":"v1"}',
        '{"schema_version":"v0","pipeline_version":"v1","response":{}}',
        "GROUNDING_FALSE",
        "BLANK_ANSWER",
        "NO_SOURCES",
        "EXTRA_FIELD",
        "WRONG_PIPELINE",
    ],
)
async def test_invalid_cache_payload_is_ignored_and_the_graph_runs(corrupt: str) -> None:
    deps, provider, _ = dependencies(llm=RepeatingLLM())
    service, cache = make_service(deps)
    await service.answer(QUESTION)
    key, good, _ = cache.set_calls[0]
    envelope = json.loads(good)
    mutations = {
        "GROUNDING_FALSE": lambda e: e["response"].update(grounding_passed=False),
        "BLANK_ANSWER": lambda e: e["response"].update(answer="   "),
        "NO_SOURCES": lambda e: e["response"].update(sources=[]),
        "EXTRA_FIELD": lambda e: e["response"].update(previous_answer="ungrounded"),
        "WRONG_PIPELINE": lambda e: e.update(pipeline_version="v0"),
    }
    if corrupt in mutations:
        mutations[corrupt](envelope)
        corrupt = json.dumps(envelope)
    cache.values[key] = corrupt

    response = await service.answer(QUESTION)

    assert (response.cache_status, response.outcome, response.answer) == ("miss", "answered", ANSWER)
    assert len(provider.generation_calls) == 2
    assert service.response_cache.stats_snapshot().invalid_entries == 1
    assert deserialize_agent_answer(cache.values[key]) is not None


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["raises", "reports_failure"])
async def test_cache_read_failure_is_fail_open(mode: str) -> None:
    deps, provider, _ = dependencies()
    service, cache = make_service(deps)
    if mode == "raises":
        cache.read_error = RuntimeError("private redis detail")
    else:
        cache.read_failure = True

    response = await service.answer(QUESTION)

    assert (response.cache_status, response.outcome, response.answer) == ("failure", "answered", ANSWER)
    assert len(provider.generation_calls) == 1
    assert service.response_cache.stats_snapshot().read_failures == 1


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["raises", "reports_failure"])
async def test_cache_write_failure_still_returns_the_grounded_answer(mode: str) -> None:
    deps, _, _ = dependencies()
    service, cache = make_service(deps)
    if mode == "raises":
        cache.write_error = RuntimeError("private redis detail")
    else:
        cache.write_result = False

    response = await service.answer(QUESTION)

    assert (response.cache_status, response.outcome, response.answer) == ("miss", "answered", ANSWER)
    assert service.response_cache.stats_snapshot().write_failures == 1
    assert cache.values == {}


@pytest.mark.anyio
async def test_citation_free_insufficiency_answer_is_returned_but_not_cached() -> None:
    answer = "The available evidence is insufficient to answer the question."
    deps, _, _ = dependencies(llm=FakeLLMProvider(answer=answer))
    service, cache = make_service(deps)

    response = await service.answer(QUESTION)

    assert (response.outcome, response.answer, response.grounding_passed) == ("answered", answer, True)
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_prompts_are_resolved_once_per_request_and_the_same_bundle_keys_and_runs() -> None:
    managed = ResolvedPrompt(
        name="agent-guardrail",
        content='MANAGED guardrail. Return {"score": <0-100>}.',
        version="langfuse-v7",
        label="production",
        source="langfuse",
    )
    resolver = CountingResolver({"agent-guardrail": managed})
    system_prompts: list[str] = []

    class RecordingLLM(FakeLLMProvider):
        async def complete(self, messages, **kwargs):
            system_prompts.append(messages[0].content)
            return await super().complete(messages, **kwargs)

    deps, _, _ = dependencies(llm=RecordingLLM())
    service, cache = make_service(deps, resolver=resolver)
    local_service, local_cache = make_service(dependencies()[0])

    prepared = await service.prepare(QUESTION)
    response = await service.execute(prepared)
    await local_service.answer(QUESTION)

    assert resolver.calls == 7
    assert prepared.prompts.guardrail == managed
    assert system_prompts[0] == managed.content
    assert response.outcome == "answered"
    assert cache.set_calls[0][0] == prepared.lookup.key
    assert cache.set_calls[0][0] != local_cache.set_calls[0][0]
    await service.answer(QUESTION)
    assert resolver.calls == 14


@pytest.mark.anyio
@pytest.mark.parametrize("prompt_name", ["rag-answer-generation", "agent-answer-grounding", "agent-query-rewrite"])
async def test_prompt_change_gives_a_different_key_and_a_miss(prompt_name: str) -> None:
    deps, provider, _ = dependencies(llm=RepeatingLLM())
    cache = FakeCache()
    service, _ = make_service(deps, cache=cache)
    await service.answer(QUESTION)
    key = {definition.name: key for key, definition in AGENT_PROMPT_DEFINITIONS.items()}[prompt_name]
    local = getattr(local_agent_prompt_bundle(), key)
    changed = dataclasses.replace(local, content=local.content + "\nAn additional managed instruction.", version="langfuse-v2", source="langfuse")
    changed_service, _ = make_service(deps, cache=cache, resolver=CountingResolver({prompt_name: changed}))

    response = await changed_service.answer(QUESTION)

    assert response.cache_status == "miss"
    assert len({key for key, _, _ in cache.set_calls}) == 2
    assert len(provider.generation_calls) == 2
    assert (await service.answer(QUESTION)).cache_status == "hit"
    assert (await changed_service.answer(QUESTION)).cache_status == "hit"


@pytest.mark.anyio
async def test_corpus_generation_change_gives_a_different_key_and_a_miss() -> None:
    deps, _, _ = dependencies(llm=RepeatingLLM())
    cache = FakeCache()
    first, _ = make_service(deps, cache=cache, rag_corpus_generation="corpus-v1")
    second, _ = make_service(deps, cache=cache, rag_corpus_generation="corpus-v2")

    await first.answer(QUESTION)
    response = await second.answer(QUESTION)

    assert response.cache_status == "miss"
    assert cache.get_keys[0] != cache.get_keys[1]
    assert (await first.answer(QUESTION)).cache_status == "hit"


@pytest.mark.anyio
async def test_compiled_graphs_are_reused_per_prompt_identity_within_a_small_bound() -> None:
    deps, _, _ = dependencies()
    service, _ = make_service(deps, redis_enabled=False)
    local = local_agent_prompt_bundle()

    first = service._graph_for(local)
    assert service._graph_for(local_agent_prompt_bundle()) is first
    for index in range(6):
        variant = dataclasses.replace(
            local, generation=dataclasses.replace(local.generation, content=f"Variant {index}", version=f"langfuse-v{index + 1}")
        )
        service._graph_for(variant)

    assert len(service._graphs) == 4
    assert service._graph_for(local) is not first


@pytest.mark.anyio
async def test_statuses_stream_in_order_as_they_happen_and_contain_no_content() -> None:
    deps, _, _, _ = exhausted_dependencies(live=live_result("2501.00001", "2501.00002"), selection=SELECT_TWO)
    service, _ = make_service(deps)
    seen = []

    async def on_status(status) -> None:
        seen.append(status)

    response = await service.answer(INDIA_QUESTION, on_status=on_status)

    assert seen == response.statuses
    serialized = json.dumps([status.model_dump() for status in seen])
    for forbidden in (
        INDIA_QUESTION,
        "India",
        "healthcare challenges",
        EVIDENCE_TEXT,
        "Private",
        "private",
        ANSWER,
        "2501.0000",
        "arxiv.org",
        "UNTRUSTED_",
        '{"score"',
        "rag:agent-response",
        "Error",
    ):
        assert forbidden not in serialized


@pytest.mark.anyio
async def test_request_observation_gets_prompt_and_cache_spans_and_generations_only_on_a_miss() -> None:
    deps, _, _ = dependencies()
    service, _ = make_service(deps)
    miss_root, hit_root = RecordingObservation(), RecordingObservation()

    miss = await service.answer(QUESTION, observation=miss_root)
    hit = await service.answer(QUESTION, observation=hit_root)

    assert [child.name for child in miss_root.children if child.kind == "span"][:2] == [
        "agent.prompt_resolution",
        "agent.cache.lookup",
    ]
    assert "agent.cache.write" in [child.name for child in miss_root.children]
    assert [g.name for g in miss_root.generations] == [
        "agent.guardrail",
        "agent.evidence_grader",
        "rag.generation",
        "agent.answer_grounding",
    ]
    assert [(child.name, child.kind) for child in hit_root.children] == [
        ("agent.prompt_resolution", "span"),
        ("agent.cache.lookup", "span"),
    ]
    assert hit_root.generations == []
    assert hit_root.children[1].metadata[1] == {"outcome": "hit"}
    assert miss_root.children[0].metadata[1]["prompt_versions"]["generation"] == "local-v1"
    recorded = repr([(c.name, c.metadata, c.generation_records) for root in (miss_root, hit_root) for c in root.children])
    assert "PRIVATE_QUESTION_SENTINEL" not in recorded and ANSWER not in recorded and EVIDENCE_TEXT not in recorded
    miss_meta, hit_meta = service.trace_metadata(miss), service.trace_metadata(hit)
    assert (miss_meta["cache_outcome"], miss_meta["model"], miss_meta["generation_attempts"]) == ("miss", "fake-model", 1)
    assert (hit_meta["cache_outcome"], hit_meta["artifact_model"]) == ("hit", "fake-model")
    assert "model" not in hit_meta
    assert ANSWER not in repr((miss_meta, hit_meta))


@pytest.mark.anyio
async def test_exhausted_ungrounded_answer_left_in_state_is_never_mapped_to_a_success() -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=LOW))
    state = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state(QUESTION))

    assert state["generated_answer"] == REGENERATED_ANSWER
    assert (state["grounding_passed"], state["terminal_reason"]) == (False, TerminalReason.GROUNDING_FAILED)
    response = map_agent_state(state, cache_status="miss")
    assert (response.outcome, response.answer, response.sources) == ("grounding_failed", GROUNDING_FAILED_MESSAGE, [])
    assert not is_cacheable(state, response)
    for tampered in (
        {**state, "terminal_reason": None},
        {**state, "terminal_reason": None, "grounding_passed": None},
        {**state, "terminal_reason": None, "grounding_passed": True, "generated_answer": "different text"},
        {**state, "terminal_reason": None, "grounding_passed": True, "generation_result": None},
    ):
        with pytest.raises(AgentExecutionFailure) as caught:
            map_agent_state(tampered, cache_status="miss")
        assert caught.value.kind is AgentFailureKind.INTERNAL_ERROR


def test_cached_answer_schema_round_trips_and_rejects_non_grounded_payloads() -> None:
    source = AgentSource(
        citation="[S1]", source_type="live_arxiv", arxiv_id="2501.00001", chunk_id="live::2501.00001::chunk::000",
        chunk_index=0, title="T", content="C", truncated=False, source_url="https://arxiv.org/pdf/2501.00001",
    )
    answer = AgentCachedAnswer(answer="Grounded [S1].", sources=[source], model="m", prompt_tokens=1, completion_tokens=2)

    assert deserialize_agent_answer(serialize_agent_answer(answer)) == answer
    assert serialize_agent_answer(answer) == serialize_agent_answer(answer)
    for invalid in (None, 5, "", "   ", "null", '"text"'):
        assert deserialize_agent_answer(invalid) is None
    for bad in ({"answer": " "}, {"sources": []}, {"grounding_passed": False}, {"prompt_tokens": -1}):
        with pytest.raises(ValueError):
            AgentCachedAnswer(**{**answer.model_dump(), **bad})


def test_invalid_service_bounds_are_rejected() -> None:
    deps, _, _ = dependencies()
    for kwargs in ({"request_timeout_seconds": 0}, {"compiled_graph_cache_size": 0}):
        with pytest.raises(ValueError):
            AgentService(
                dependencies=deps,
                graph_config=AgentGraphConfig(),
                prompt_resolver=LocalPromptResolver(),
                response_cache=AgentResponseCache(cache=FakeCache(), settings=Settings(_env_file=None)),
                **kwargs,
            )


@pytest.mark.anyio
async def test_rewrite_path_stays_within_configured_local_attempts() -> None:
    first = retrieval_result(hits=[make_hit("a1", "attempt one evidence")])
    second = retrieval_result(hits=[make_hit("b1", "attempt two evidence")])
    llm = FakeLLMProvider('{"score":90}', '{"score":25}', REWRITE_JSON, '{"score":88}')
    deps, _, retrieval = dependencies(llm=llm, results=[first, second])
    service, _ = make_service(deps)

    response = await service.answer("Tell me about ML stuff")

    assert response.outcome == "answered"
    assert (response.execution.retrieval_attempts, response.execution.query_rewritten) == (2, True)
    assert response.execution.live_fallback_used is False
    assert retrieval.search.call_count == 2
    assert [status.message for status in response.statuses].count("Query rewritten") == 1


PARTIAL = "Private partial synthesis cut off [S1] and then"
RECOVERED = "Private recovered complete synthesis [S1]."


@pytest.mark.anyio
async def test_recovered_complete_answer_is_returned_cached_and_hit_on_repeat() -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=(Truncated(PARTIAL), Stopped(RECOVERED))))
    service, cache = make_service(deps)

    response = await service.answer(QUESTION)
    hit = await service.answer(QUESTION)

    assert (response.outcome, response.answer, response.grounding_passed) == ("answered", RECOVERED, True)
    # One generation cycle: the length recovery is not a second answer attempt or grounding attempt.
    assert (response.execution.generation_attempts, response.execution.grounding_attempts) == (1, 1)
    assert "generation_length_recovery_started" in [status.code for status in response.statuses]
    assert len(cache.set_calls) == 1
    assert RECOVERED in cache.set_calls[0][1] and "rivate partial" not in cache.set_calls[0][1]
    assert "rivate partial" not in response.model_dump_json()
    assert (hit.cache_status, hit.answer) == ("hit", RECOVERED)
    assert len(provider.generation_calls) == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("answers", "generation_calls"),
    [
        ((Truncated(PARTIAL), Truncated(PARTIAL + " more")), 2),
        ((Truncated(PARTIAL), Stopped("Private recovery without a citation.")), 2),
        ((Stopped("Private answer without a citation."),), 1),
    ],
)
async def test_rejected_model_answers_become_a_safe_generation_failed_outcome_and_are_never_cached(
    answers, generation_calls
) -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=answers))
    service, cache = make_service(deps)

    response = await service.answer(QUESTION)

    assert (response.outcome, response.terminal_reason) == ("generation_failed", "generation_failed")
    assert (response.answer, response.sources, response.grounding_passed) == (ANSWER_NOT_VALIDATED_MESSAGE, [], None)
    assert (response.model, response.prompt_tokens, response.completion_tokens) == (None, None, None)
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (generation_calls, 0)
    assert cache.set_calls == []
    assert "rivate" not in response.model_dump_json()


@pytest.mark.anyio
async def test_a_truncated_result_can_never_be_exposed_or_cached_even_if_it_reaches_the_mapper() -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(answer=Stopped(ANSWER)))
    state = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state(QUESTION))
    complete = map_agent_state(state, cache_status="miss")
    truncated_state = {
        **state,
        "generation_result": dataclasses.replace(state["generation_result"], finish_reason="length"),
    }

    assert is_cacheable(state, complete) is True
    assert is_cacheable(truncated_state, complete) is False
    with pytest.raises(AgentExecutionFailure) as caught:
        map_agent_state(truncated_state, cache_status="miss")
    assert caught.value.kind is AgentFailureKind.INTERNAL_ERROR
