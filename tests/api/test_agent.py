import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from src.dependencies import get_agent_service, get_observability_provider
from src.main import app
from src.routers.agent import FAILURE_RESPONSES
from src.services.agent import AgentGraphConfig
from src.services.agent.service import AgentFailureKind
from src.services.agent.status import (
    ANSWER_NOT_VALIDATED_MESSAGE,
    GROUNDING_FAILED_MESSAGE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
)

from tests.api.test_ask_stream import parse_sse
from tests.unit.services.agent.test_graph import (
    ANSWER,
    EVIDENCE_TEXT,
    HIGH,
    INDIA_QUESTION,
    LOW,
    REGENERATED_ANSWER,
    SELECT_TWO,
    FakeLLMProvider,
    RecordingObservation,
    Stopped,
    Truncated,
    dependencies,
    exhausted_dependencies,
    live_result,
)
from tests.unit.services.agent.test_service import LOCAL_ONLY, FakeCache, RepeatingLLM, make_service

QUESTION = "PRIVATE_QUESTION_SENTINEL how is NLP used in clinical decision support?"
ASK, STREAM = "/api/v1/agent/ask", "/api/v1/agent/ask/stream"


class RecordingProvider:
    capture_content = False

    def __init__(self) -> None:
        self.roots: list[RecordingObservation] = []

    def start_trace(self, *, name, metadata=None):
        root = RecordingObservation(name, "span", metadata)
        self.roots.append(root)
        return root


@pytest.fixture
def observability():
    provider = RecordingProvider()
    app.dependency_overrides[get_observability_provider] = lambda: provider
    yield provider
    app.dependency_overrides.pop(get_observability_provider, None)
    app.dependency_overrides.pop(get_agent_service, None)


def use_service(deps, **kwargs):
    service, cache = make_service(deps, **kwargs)
    app.dependency_overrides[get_agent_service] = lambda: service
    return service, cache


async def post(path: str, question: str = QUESTION, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json={"question": question}, **kwargs)


async def stream(question: str = QUESTION) -> list[tuple[str, dict]]:
    response = await post(STREAM, question)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def names(events) -> list[str]:
    return [name for name, _ in events]


@pytest.mark.anyio
async def test_ask_miss_then_hit_returns_the_same_grounded_answer(observability) -> None:
    deps, provider, retrieval = dependencies()
    use_service(deps)

    miss = await post(ASK)
    hit = await post(ASK)

    assert miss.status_code == hit.status_code == 200
    first, second = miss.json(), hit.json()
    assert (first["cache_status"], second["cache_status"]) == ("miss", "hit")
    for body in (first, second):
        assert (body["answer"], body["outcome"], body["grounding_passed"], body["terminal_reason"]) == (
            ANSWER,
            "answered",
            True,
            None,
        )
        assert (body["model"], body["prompt_tokens"], body["completion_tokens"]) == ("fake-model", 111, 22)
        assert body["pipeline_version"] == "v1"
        assert body["latency_ms"] >= 0
    assert first["sources"] == second["sources"]
    assert first["sources"][0]["citation"] == "[S1]"
    assert first["sources"][0]["source_type"] == "local"
    assert set(first["sources"][0]) == {
        "citation",
        "source_type",
        "arxiv_id",
        "chunk_id",
        "chunk_index",
        "title",
        "section",
        "content",
        "truncated",
        "authors",
        "categories",
        "published_date",
        "source_url",
    }
    assert second["statuses"] == [{"code": "cache_hit", "message": "Validated cached answer found"}]
    assert len(provider.generation_calls) == 1 and len(provider.grounding_calls) == 1
    retrieval.search.assert_called_once()
    assert set(first) == {
        "answer",
        "outcome",
        "sources",
        "cache_status",
        "terminal_reason",
        "grounding_passed",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "execution",
        "statuses",
        "pipeline_version",
        "latency_ms",
    }
    assert "rag:agent-response" not in miss.text and "PRIVATE_QUESTION_SENTINEL" not in miss.text


@pytest.mark.anyio
async def test_ask_with_cache_disabled_reports_bypass(observability) -> None:
    deps, _, _ = dependencies()
    use_service(deps, redis_enabled=False)

    body = (await post(ASK)).json()

    assert (body["cache_status"], body["outcome"], body["answer"]) == ("bypass", "answered", ANSWER)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("llm", "outcome", "message", "grounding_passed"),
    [
        (FakeLLMProvider('{"score":10}'), "out_of_scope", OUT_OF_SCOPE_MESSAGE, None),
        (FakeLLMProvider('{"score":90}', '{"score":20}'), "insufficient_evidence", INSUFFICIENT_EVIDENCE_MESSAGE, None),
        (FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=LOW), "grounding_failed", GROUNDING_FAILED_MESSAGE, False),
    ],
)
async def test_ask_semantic_outcomes_are_200_with_a_safe_message_and_no_model_text(
    observability, llm, outcome, message, grounding_passed
) -> None:
    deps, _, _ = dependencies(llm=RepeatingLLM(*llm.responses, answer=llm.answers, grounding=llm.groundings))
    _, cache = use_service(deps, config=LOCAL_ONLY)

    response = await post(ASK)

    assert response.status_code == 200
    body = response.json()
    assert (body["outcome"], body["terminal_reason"], body["answer"]) == (outcome, outcome, message)
    assert (body["grounding_passed"], body["sources"], body["model"]) == (grounding_passed, [], None)
    assert "Private generated" not in response.text and "Private regenerated" not in response.text
    assert cache.set_calls == []
    assert observability.roots[0].metadata[-1]["outcome"] == outcome


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("deps_factory", "kind"),
    [
        (lambda: dependencies(llm=FakeLLMProvider(RuntimeError("private failure detail"))), AgentFailureKind.INTERNAL_ERROR),
        (lambda: dependencies(retrieval_error=RuntimeError("private failure detail")), AgentFailureKind.RETRIEVAL_FAILED),
        (lambda: dependencies(llm=FakeLLMProvider(answer=RuntimeError("private failure detail"))), AgentFailureKind.GENERATION_FAILED),
    ],
)
async def test_ask_execution_failures_map_to_safe_http_errors(observability, deps_factory, kind) -> None:
    deps, _, _ = deps_factory()
    use_service(deps)
    status_code, code, detail = FAILURE_RESPONSES[kind]

    response = await post(ASK)

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "private failure detail" not in response.text
    root = observability.roots[0]
    assert root.error_types == [code] and root.end_count == 1
    assert root.metadata[-1] == {"status": "error", "failure_classification": code}


@pytest.mark.anyio
async def test_ask_timeout_is_504_without_partial_answer(observability) -> None:
    class SlowLLM(FakeLLMProvider):
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(5)
            return await super().complete(messages, **kwargs)

    deps, _, _ = dependencies(llm=SlowLLM())
    use_service(deps, request_timeout_seconds=0.05)

    response = await post(ASK)

    assert response.status_code == 504
    assert response.json() == {"detail": "The research assistant took too long to answer this question."}
    assert {code for code, _, _ in FAILURE_RESPONSES.values()} == {500, 502, 503, 504}


@pytest.mark.anyio
async def test_ask_with_invalid_cache_payload_or_cache_failures_still_answers(observability) -> None:
    deps, provider, _ = dependencies(llm=RepeatingLLM())
    service, cache = use_service(deps)
    await post(ASK)
    key = cache.set_calls[0][0]

    cache.values[key] = '{"schema_version":"v1","response":{"answer":"tampered"}}'
    invalid = (await post(ASK)).json()
    cache.read_error = RuntimeError("private redis detail")
    read_failure = (await post(ASK)).json()
    cache.read_error, cache.write_error = None, RuntimeError("private redis detail")
    cache.values.clear()
    write_failure = await post(ASK)

    assert (invalid["cache_status"], invalid["answer"]) == ("miss", ANSWER)
    assert (read_failure["cache_status"], read_failure["answer"]) == ("failure", ANSWER)
    assert write_failure.status_code == 200 and write_failure.json()["answer"] == ANSWER
    assert "private redis detail" not in write_failure.text
    assert len(provider.generation_calls) == 4
    stats = service.response_cache.stats_snapshot()
    assert (stats.invalid_entries, stats.read_failures, stats.write_failures) == (1, 1, 1)


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}, {"question": "ok", "live_fallback_enabled": False}])
async def test_ask_and_stream_validate_the_request_and_ignore_config_overrides(observability, payload) -> None:
    deps, _, _ = dependencies(llm=RepeatingLLM())
    use_service(deps, redis_enabled=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        ask, streamed = await client.post(ASK, json=payload), await client.post(STREAM, json=payload)

    if payload.get("question", "").strip():
        assert ask.status_code == streamed.status_code == 200
        assert ask.json()["execution"]["live_fallback_used"] is False
    else:
        assert ask.status_code == streamed.status_code == 422
        assert observability.roots == []


@pytest.mark.anyio
async def test_endpoints_are_unavailable_without_a_configured_agent_service() -> None:
    app.dependency_overrides.pop(get_agent_service, None)
    previous = getattr(app.state, "agent_service", None)
    app.state.agent_service = None
    try:
        for path in (ASK, STREAM):
            response = await post(path)
            assert response.status_code == 503
            assert response.json() == {"detail": "Language model service is not configured."}
    finally:
        app.state.agent_service = previous


@pytest.mark.anyio
async def test_stream_miss_emits_statuses_then_answer_sources_done(observability) -> None:
    deps, _, _ = dependencies()
    use_service(deps)

    events = await stream()

    assert names(events)[0] == "metadata"
    assert events[0][1] == {"cache_status": "miss", "pipeline_version": "v1"}
    assert names(events)[-3:] == ["answer", "sources", "done"]
    assert set(names(events)[1:-3]) == {"status"}
    assert [data["code"] for name, data in events if name == "status"] == [
        "graph_started",
        "guardrail_started",
        "guardrail_passed",
        "local_retrieval_started",
        "local_retrieval_completed",
        "evidence_grading_started",
        "evidence_sufficient",
        "evidence_rerank_started",
        "evidence_rerank_completed",
        "generation_started",
        "generation_completed",
        "grounding_started",
        "grounding_passed",
        "graph_completed",
    ]
    answer_index = names(events).index("answer")
    grounding_index = [i for i, (name, data) in enumerate(events) if data.get("code") == "grounding_passed"][0]
    assert grounding_index < answer_index
    assert names(events).count("answer") == 1
    assert events[answer_index][1] == {"text": ANSWER}
    assert [source["citation"] for source in events[-2][1]["sources"]] == ["[S1]"]
    done = events[-1][1]
    assert (done["outcome"], done["grounding_passed"], done["cache_status"], done["terminal_reason"]) == (
        "answered",
        True,
        "miss",
        None,
    )
    assert (done["model"], done["prompt_tokens"], done["completion_tokens"], done["citations"]) == (
        "fake-model",
        111,
        22,
        ["[S1]"],
    )
    assert done["execution"]["generation_attempts"] == 1 and done["latency_ms"] >= 0
    root = observability.roots[0]
    assert (root.name, root.end_count, root.error_types) == ("agent.request", 1, [])
    assert root.metadata[0] == {"transport": "stream", "content_capture": False, "pipeline_version": "v1"}
    assert root.metadata[-1]["cache_outcome"] == "miss"


@pytest.mark.anyio
async def test_stream_hit_has_one_cache_status_and_no_fake_execution_statuses(observability) -> None:
    deps, provider, _ = dependencies()
    use_service(deps)
    miss = await stream()

    hit = await stream()

    assert names(hit) == ["metadata", "status", "answer", "sources", "done"]
    assert hit[0][1]["cache_status"] == "hit"
    assert hit[1][1] == {"code": "cache_hit", "message": "Validated cached answer found"}
    assert hit[2][1] == miss[-3][1]
    assert hit[3][1] == miss[-2][1]
    assert hit[4][1]["cache_status"] == "hit" and hit[4][1]["grounding_passed"] is True
    assert len(provider.generation_calls) == 1
    hit_root = observability.roots[1]
    assert hit_root.generations == []
    assert hit_root.metadata[-1]["cache_outcome"] == "hit"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("llm", "outcome", "message", "last_status"),
    [
        (FakeLLMProvider('{"score":10}'), "out_of_scope", OUT_OF_SCOPE_MESSAGE, "guardrail_rejected"),
        (FakeLLMProvider('{"score":90}', '{"score":20}'), "insufficient_evidence", INSUFFICIENT_EVIDENCE_MESSAGE, "evidence_insufficient"),
        (FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=LOW), "grounding_failed", GROUNDING_FAILED_MESSAGE, "grounding_failed"),
    ],
)
async def test_stream_semantic_outcomes_end_with_outcome_then_done_and_no_answer(
    observability, llm, outcome, message, last_status
) -> None:
    deps, _, _ = dependencies(llm=llm)
    use_service(deps, config=LOCAL_ONLY)

    events = await stream()

    assert names(events)[0] == "metadata" and names(events)[-2:] == ["outcome", "done"]
    assert "answer" not in names(events) and "sources" not in names(events)
    assert events[-2][1] == {"outcome": outcome, "message": message}
    assert (events[-1][1]["outcome"], events[-1][1]["terminal_reason"], events[-1][1]["citations"]) == (outcome, outcome, [])
    codes = [data["code"] for name, data in events if name == "status"]
    assert codes[-2:] == [last_status, "graph_completed"]
    body = json.dumps(events)
    assert "Private generated" not in body and "Private regenerated" not in body
    if outcome == "grounding_failed":
        assert codes.count("generation_completed") == 2 and codes.count("grounding_failed") == 2
        assert events[-1][1]["grounding_passed"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("deps_factory", "code"),
    [
        (lambda: dependencies(llm=FakeLLMProvider(grounding=RuntimeError("private failure detail"))), "internal_error"),
        (lambda: dependencies(llm=FakeLLMProvider(answer=(ANSWER, RuntimeError("private failure detail")), grounding=LOW)), "generation_failed"),
        (lambda: dependencies(retrieval_error=RuntimeError("private failure detail")), "retrieval_failed"),
    ],
)
async def test_stream_failure_after_start_ends_with_one_error_and_no_answer(observability, deps_factory, code) -> None:
    deps, _, _ = deps_factory()
    use_service(deps)

    events = await stream()

    assert names(events)[0] == "metadata" and names(events)[-1] == "error"
    assert not {"answer", "sources", "outcome", "done"} & set(names(events))
    assert events[-1][1]["code"] == code
    assert events[-1][1]["message"] in {detail for _, _, detail in FAILURE_RESPONSES.values()}
    body = json.dumps(events)
    assert "private failure detail" not in body and ANSWER not in body
    assert [data["code"] for name, data in events if name == "status"][-1] == "graph_failed"
    assert observability.roots[0].error_types == [code]


@pytest.mark.anyio
async def test_stream_timeout_is_an_error_event(observability) -> None:
    class SlowLLM(FakeLLMProvider):
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(5)
            return await super().complete(messages, **kwargs)

    deps, _, _ = dependencies(llm=SlowLLM())
    use_service(deps, request_timeout_seconds=0.05)

    events = await stream()

    assert names(events) == ["metadata", "status", "error"]
    assert events[-1][1] == {"code": "timeout", "message": "The research assistant took too long to answer this question."}


@pytest.mark.anyio
async def test_stream_regeneration_emits_only_the_grounded_second_answer(observability) -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH)))
    use_service(deps)

    events = await stream()

    assert names(events).count("answer") == 1
    assert events[names(events).index("answer")][1] == {"text": REGENERATED_ANSWER}
    assert ANSWER not in json.dumps(events)
    messages = [data["message"] for name, data in events if name == "status"]
    assert messages[-6:] == [
        "Grounding validation failed",
        "Regenerating the answer",
        "Answer regenerated",
        "Validating answer grounding",
        "Grounding validation passed",
        "Completed",
    ]


@pytest.mark.anyio
async def test_stream_live_path_statuses_and_sources_are_safe_and_truthful(observability) -> None:
    deps, _, _, _ = exhausted_dependencies(live=live_result("2501.00001", "2501.00002"), selection=SELECT_TWO)
    use_service(deps)

    events = await stream(INDIA_QUESTION)

    statuses = json.dumps([data for name, data in events if name in {"metadata", "status"}])
    for forbidden in (
        "India",
        EVIDENCE_TEXT,
        "Private",
        "private",
        "2501.0000",
        "arxiv.org",
        "UNTRUSTED_",
        '{"score"',
        "rag:agent-response",
        "embedding",
        "reasoning",
    ):
        assert forbidden not in statuses
    sources = events[-2][1]["sources"]
    assert [(source["citation"], source["source_type"]) for source in sources] == [
        ("[S1]", "local"),
        ("[S2]", "live_arxiv"),
        ("[S3]", "live_arxiv"),
    ]
    assert sources[1]["source_url"] == "https://arxiv.org/pdf/2501.00001"
    assert events[-1][1]["execution"] == {
        "retrieval_attempts": 2,
        "query_rewritten": True,
        "live_fallback_used": True,
        "live_papers_selected": 2,
        "generation_attempts": 1,
        "grounding_attempts": 1,
        "source_count": 3,
    }


@pytest.mark.anyio
async def test_telemetry_for_both_endpoints_contains_no_question_evidence_or_answer(observability) -> None:
    deps, _, _ = dependencies(llm=RepeatingLLM())
    use_service(deps, redis_enabled=False)

    await post(ASK)
    await stream()

    recorded = repr(
        [
            (node.name, node.metadata, node.generation_records, node.error_types)
            for root in observability.roots
            for node in (root, *root.children)
        ]
    )
    for forbidden in ("PRIVATE_QUESTION_SENTINEL", EVIDENCE_TEXT, ANSWER, "Private Paper Title", "You are", "UNTRUSTED_"):
        assert forbidden not in recorded
    assert [root.name for root in observability.roots] == ["agent.request", "agent.request"]
    assert all(root.end_count == 1 for root in observability.roots)
    assert "rag.generation" in [g.name for g in observability.roots[0].generations]


def test_agent_routes_are_registered_alongside_the_unchanged_week5_routes() -> None:
    paths = {route.path for route in app.routes}

    assert {"/api/v1/ask", "/api/v1/ask/stream", "/api/v1/agent/ask", "/api/v1/agent/ask/stream"} <= paths
    assert AgentGraphConfig().live_fallback_enabled is True


@pytest.mark.anyio
async def test_stream_sends_keepalive_comments_during_long_silent_work(observability, monkeypatch) -> None:
    class SlowLLM(FakeLLMProvider):
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(0.12)
            return await super().complete(messages, **kwargs)

    monkeypatch.setattr("src.routers.agent.KEEPALIVE_SECONDS", 0.03)
    deps, _, _ = dependencies(llm=SlowLLM())
    use_service(deps)

    response = await post(STREAM)

    frames = response.text.strip().split("\n\n")
    keepalives = [frame for frame in frames if frame == ": keepalive"]
    assert len(keepalives) >= 2
    events = parse_sse("\n\n".join(frame for frame in frames if frame != ": keepalive"))
    assert names(events)[0] == "metadata" and names(events)[-3:] == ["answer", "sources", "done"]
    assert frames.index(": keepalive") < frames.index(next(f for f in frames if f.startswith("event: answer")))


PARTIAL = "Private partial synthesis cut off [S1] and then"
RECOVERED = "Private recovered complete synthesis [S1]."
REJECTED_ANSWER_CASES = [
    (Truncated(PARTIAL), Truncated(PARTIAL + " more")),
    (Truncated(PARTIAL), Stopped("Private recovery without a citation.")),
    (Stopped("Private answer with an unknown label [S9]."),),
]


@pytest.mark.anyio
async def test_ask_and_stream_return_only_the_recovered_complete_answer_and_cache_it(observability) -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=(Truncated(PARTIAL), Stopped(RECOVERED))))
    _, cache = use_service(deps)

    events = await stream()
    repeat = await post(ASK)

    assert names(events).count("answer") == 1
    answer_at = names(events).index("answer")
    assert events[answer_at][1] == {"text": RECOVERED}
    statuses = [(data["code"], data["message"]) for name, data in events[:answer_at] if name == "status"]
    assert ("generation_length_recovery_started", "Initial answer was incomplete; regenerating with a larger output limit") in statuses
    assert statuses[-2][0] == "grounding_passed"
    assert events[-1][1]["outcome"] == "answered" and events[-1][1]["grounding_passed"] is True
    assert "rivate partial" not in json.dumps(events)
    assert len(cache.set_calls) == 1 and "rivate partial" not in cache.set_calls[0][1]
    assert repeat.status_code == 200
    assert (repeat.json()["cache_status"], repeat.json()["answer"]) == ("hit", RECOVERED)
    assert len(provider.generation_calls) == 2
    assert "rivate partial" not in repr([root.metadata for root in observability.roots])


@pytest.mark.anyio
@pytest.mark.parametrize("answers", REJECTED_ANSWER_CASES)
async def test_ask_rejected_model_answer_is_a_safe_200_outcome_without_model_text(observability, answers) -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=answers))
    _, cache = use_service(deps)

    response = await post(ASK)

    assert response.status_code == 200
    body = response.json()
    assert (body["outcome"], body["terminal_reason"]) == ("generation_failed", "generation_failed")
    assert (body["answer"], body["sources"], body["grounding_passed"]) == (ANSWER_NOT_VALIDATED_MESSAGE, [], None)
    assert "rivate" not in response.text.replace("PRIVATE_QUESTION", "")
    assert cache.set_calls == [] and provider.grounding_calls == []
    assert observability.roots[0].metadata[-1]["outcome"] == "generation_failed"


@pytest.mark.anyio
@pytest.mark.parametrize("answers", REJECTED_ANSWER_CASES)
async def test_stream_rejected_model_answer_ends_with_outcome_and_never_an_answer(observability, answers) -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(answer=answers))
    _, cache = use_service(deps)

    events = await stream()

    assert names(events)[0] == "metadata" and names(events)[-2:] == ["outcome", "done"]
    assert not {"answer", "sources", "error"} & set(names(events))
    assert events[-2][1] == {"outcome": "generation_failed", "message": ANSWER_NOT_VALIDATED_MESSAGE}
    assert (events[-1][1]["terminal_reason"], events[-1][1]["citations"]) == ("generation_failed", [])
    codes = [data["code"] for name, data in events if name == "status"]
    assert "grounding_started" not in codes and "grounding_failed" not in codes
    assert "rivate" not in json.dumps(events)
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_structural_and_semantic_failures_are_distinct_public_outcomes(observability) -> None:
    structural, _, _ = dependencies(llm=FakeLLMProvider(answer=Stopped("Private answer without a citation.")))
    use_service(structural)
    structural_body = (await post(ASK)).json()
    semantic, _, _ = dependencies(
        llm=FakeLLMProvider(answer=(Stopped(ANSWER), Stopped(REGENERATED_ANSWER)), grounding=LOW)
    )
    use_service(semantic)
    semantic_body = (await post(ASK)).json()

    assert (structural_body["outcome"], structural_body["answer"]) == ("generation_failed", ANSWER_NOT_VALIDATED_MESSAGE)
    assert (semantic_body["outcome"], semantic_body["answer"]) == ("grounding_failed", GROUNDING_FAILED_MESSAGE)
    assert (structural_body["grounding_passed"], semantic_body["grounding_passed"]) == (None, False)
    assert (structural_body["execution"]["grounding_attempts"], semantic_body["execution"]["grounding_attempts"]) == (0, 2)
