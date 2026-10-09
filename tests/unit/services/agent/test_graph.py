from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.agent import (
    AgentErrorCategory,
    AgentExecutionStatus,
    AgentGraphConfig,
    AgentGraphDependencies,
    EvidenceGrade,
    TerminalReason,
    build_agent_graph,
    create_initial_agent_state,
)
from src.services.evidence import EvidenceContextBuilder
from src.services.search.hybrid_service import HybridSearchService

S = AgentExecutionStatus
EVIDENCE_TEXT = "private evidence text about clinical NLP"
INDIA_QUESTION = "What are the current healthcare issues in India and how can AI help solve them?"

GUARDRAIL_PASSED = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GUARDRAIL_PASSED]
RETRIEVED = [*GUARDRAIL_PASSED, S.LOCAL_RETRIEVAL_STARTED, S.LOCAL_RETRIEVAL_COMPLETED]
SUFFICIENT_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_SUFFICIENT, S.GRAPH_COMPLETED]
INSUFFICIENT_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT, S.GRAPH_COMPLETED]
GRADER_FAILED_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.GRAPH_FAILED]
REJECTED_PATH = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GUARDRAIL_REJECTED, S.GRAPH_COMPLETED]
GUARDRAIL_FAILED_PATH = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GRAPH_FAILED]
RETRIEVAL_FAILED_PATH = [*GUARDRAIL_PASSED, S.LOCAL_RETRIEVAL_STARTED, S.GRAPH_FAILED]


class FakeLLMProvider:
    """Returns scripted completions in call order: guardrail first, then grader."""

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses or ('{"score":90}', '{"score":85}'))
        self.calls: list[tuple] = []

    async def complete(self, messages, **kwargs):
        response = self.responses[len(self.calls)]
        self.calls.append((messages, kwargs))
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


def make_hit(chunk_id: str = "c1", text: str = EVIDENCE_TEXT) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=chunk_id,
        arxiv_id="2401.00001",
        chunk_index=0,
        chunk_text=text,
        word_count=len(text.split()),
        paper_title="Private Paper Title",
        rrf_score=0.5,
    )


def retrieval_result(mode: str = "hybrid", hits: list[HybridSearchHit] | None = None) -> HybridSearchResult:
    results = [make_hit()] if hits is None else hits
    return HybridSearchResult(query="question", retrieval_mode=mode, count=len(results), results=results)


def dependencies(*, llm=None, result=None, retrieval_error=None, max_context_tokens=1000):
    provider = llm or FakeLLMProvider()
    retrieval = Mock(spec=HybridSearchService)
    retrieval.search.return_value = result or retrieval_result()
    retrieval.search.side_effect = retrieval_error
    deps = AgentGraphDependencies(
        llm_provider=provider,
        hybrid_search_service=retrieval,
        evidence_context_builder=EvidenceContextBuilder(max_context_tokens=max_context_tokens),
    )
    return deps, provider, retrieval


async def run(deps, question: str = "AI question", config: AgentGraphConfig | None = None):
    return await build_agent_graph(dependencies=deps, config=config).ainvoke(create_initial_agent_state(question))


def statuses(result) -> list[AgentExecutionStatus]:
    return [event.status for event in result["execution_events"]]


def assert_no_downstream_progress(result) -> None:
    assert result["rewritten_query"] is None
    assert result["generated_answer"] is None
    assert result["grounding_passed"] is None
    assert result["grounding_attempts"] == 0
    assert result["live_fallback_used"] is False
    assert result["live_evidence"] is None
    assert result["final_evidence"] is None


@pytest.mark.anyio
async def test_sufficient_path_grades_after_one_retrieval_and_completes() -> None:
    expected = retrieval_result()
    deps, provider, retrieval = dependencies(result=expected)
    question = "How can AI improve healthcare access?"

    result = await run(deps, question, AgentGraphConfig(retrieval_size=7))

    assert statuses(result) == SUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(8))
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once_with(question, size=7)
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] == expected
    assert result["guardrail_score"] == 90
    assert result["guardrail_passed"] is True
    assert result["evidence_sufficient"] is True
    assert result["evidence_grade"] == EvidenceGrade(score=85, sufficient=True, source_count=1)
    assert result["current_query"] == question
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_insufficient_path_completes_normally_without_rewrite_or_retry() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":25}'))

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == INSUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(8))
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=25, sufficient=False, source_count=1)
    assert result["current_query"] == INDIA_QUESTION
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_evidence_event_metadata_is_bounded_and_operational() -> None:
    deps, _, _ = dependencies(result=retrieval_result(hits=[make_hit("c1", "first"), make_hit("c2", "second")]))

    result = await run(deps)

    started, graded = result["execution_events"][5:7]
    assert started.metadata.model_dump(exclude_none=True) == {"retrieval_attempt": 1, "source_count": 2}
    assert graded.metadata.model_dump(exclude_none=True) == {
        "retrieval_attempt": 1,
        "source_count": 2,
        "evidence_sufficient": True,
        "evidence_score": 85,
    }
    assert result["execution_events"][4].status == S.LOCAL_RETRIEVAL_COMPLETED
    assert S.GRAPH_COMPLETED not in statuses(result)[:7]
    assert statuses(result).count(S.GRAPH_COMPLETED) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("threshold", "score", "sufficient"),
    [(60, 59, False), (60, 60, True), (80, 75, False), (70, 75, True)],
)
async def test_routing_uses_configured_evidence_threshold(threshold: int, score: int, sufficient: bool) -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider('{"score":90}', f'{{"score":{score}}}'))

    result = await run(deps, config=AgentGraphConfig(evidence_sufficiency_threshold=threshold))

    assert result["evidence_sufficient"] is sufficient
    assert result["evidence_grade"].score == score
    assert statuses(result) == (SUFFICIENT_PATH if sufficient else INSUFFICIENT_PATH)


@pytest.mark.anyio
async def test_grader_evaluates_original_question_while_retrieval_uses_current_query() -> None:
    deps, provider, retrieval = dependencies()
    state = create_initial_agent_state(INDIA_QUESTION)
    state["current_query"] = "India healthcare AI applications"

    result = await build_agent_graph(dependencies=deps).ainvoke(state)

    retrieval.search.assert_called_once_with("India healthcare AI applications", size=5)
    grader_user_message = provider.calls[1][0][1].content
    assert INDIA_QUESTION in grader_user_message
    assert "India healthcare AI applications" not in grader_user_message
    assert result["original_question"] == INDIA_QUESTION
    assert result["current_query"] == "India healthcare AI applications"


@pytest.mark.anyio
async def test_grader_receives_bounded_deterministic_evidence_context() -> None:
    hits = [make_hit("c1", "alpha " * 40), make_hit("c1", "duplicate chunk id"), make_hit("c3", "omega " * 400)]
    deps, provider, _ = dependencies(result=retrieval_result(hits=hits), max_context_tokens=120)

    result = await run(deps)

    grader_user_message = provider.calls[1][0][1].content
    assert len(grader_user_message) < 120 * 4 + 200
    assert "duplicate chunk id" not in grader_user_message
    assert grader_user_message.index("[S1]") < grader_user_message.index("[S2]")
    assert "omega " * 400 not in grader_user_message
    assert result["evidence_grade"].source_count == 2
    assert provider.calls[1][1] == {"temperature": 0.0, "max_tokens": 32}


@pytest.mark.anyio
async def test_empty_retrieval_is_insufficient_without_grader_llm_call() -> None:
    deps, provider, retrieval = dependencies(result=retrieval_result(hits=[]))

    result = await run(deps)

    assert statuses(result) == INSUFFICIENT_PATH
    assert len(provider.calls) == 1
    retrieval.search.assert_called_once()
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=0, sufficient=False, source_count=0)
    assert result["execution_events"][6].metadata.evidence_score == 0
    assert result["execution_events"][6].metadata.source_count == 0
    assert result["terminal_reason"] is None
    assert result["error_category"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "grader_response",
    [RuntimeError("private grader detail"), "not json", '{"score":73,"reason":"private grader detail"}'],
)
async def test_grader_failure_is_distinct_from_insufficient_evidence(grader_response) -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', grader_response))

    result = await run(deps)

    assert statuses(result) == GRADER_FAILED_PATH
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert "private grader detail" not in repr(result["execution_events"])
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_context_builder_failure_is_controlled_evidence_grading_failure() -> None:
    expected = retrieval_result()
    deps, provider, retrieval = dependencies(result=expected)
    builder = Mock(spec=EvidenceContextBuilder)
    builder.build.side_effect = RuntimeError("private context builder detail")
    deps = AgentGraphDependencies(
        llm_provider=provider,
        hybrid_search_service=retrieval,
        evidence_context_builder=builder,
    )

    result = await run(deps)

    assert statuses(result) == GRADER_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert [event.sequence for event in result["execution_events"]] == list(range(7))
    builder.build.assert_called_once_with(expected)
    assert len(provider.calls) == 1
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] == expected
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert result["execution_events"][5].metadata.model_dump(exclude_none=True) == {"retrieval_attempt": 1}
    serialized = repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private context builder detail" not in serialized
    assert "private context builder detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_guardrail_rejection_is_normal_completion_and_skips_retrieval_and_grading() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":10}'))

    result = await run(deps, "What is a dog?")

    assert statuses(result) == REJECTED_PATH
    assert len(provider.calls) == 1
    assert result["guardrail_score"] == 10
    assert result["guardrail_passed"] is False
    assert result["retrieval_attempts"] == 0
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.OUT_OF_SCOPE
    assert result["error_category"] is None
    retrieval.search.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("guardrail_response", [RuntimeError("private"), "not json"])
async def test_guardrail_failure_is_internal_and_safe(guardrail_response) -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider(guardrail_response))

    result = await run(deps)

    assert statuses(result) == GUARDRAIL_FAILED_PATH
    assert len(provider.calls) == 1
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.GUARDRAIL_FAILURE
    assert result["guardrail_passed"] is None
    assert result["retrieval_attempts"] == 0
    assert result["evidence_sufficient"] is None
    retrieval.search.assert_not_called()
    assert "private" not in repr(result["execution_events"])


@pytest.mark.anyio
async def test_retrieval_failure_is_not_insufficient_evidence_and_skips_grading() -> None:
    deps, provider, retrieval = dependencies(retrieval_error=RuntimeError("private retrieval detail"))

    result = await run(deps)

    assert statuses(result) == RETRIEVAL_FAILED_PATH
    assert len(provider.calls) == 1
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.RETRIEVAL_FAILED
    assert result["error_category"] == AgentErrorCategory.RETRIEVAL_FAILURE
    retrieval.search.assert_called_once()
    assert "private retrieval detail" not in repr(result["execution_events"])


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["hybrid", "bm25_fallback", "vector_fallback"])
async def test_every_valid_hybrid_mode_is_graded(mode: str) -> None:
    expected = retrieval_result(mode)
    deps, _, _ = dependencies(result=expected)

    result = await run(deps)

    assert result["local_retrieval_result"] == expected
    assert statuses(result) == SUFFICIENT_PATH


@pytest.mark.anyio
async def test_sync_retrieval_is_offloaded_without_timing_assertions() -> None:
    expected = retrieval_result()
    deps, _, retrieval = dependencies(result=expected)

    with patch("src.services.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=expected)) as offload:
        result = await run(deps, "AI question")

    offload.assert_awaited_once_with(retrieval.search, "AI question", size=5)
    retrieval.search.assert_not_called()
    assert result["local_retrieval_result"] == expected


@pytest.mark.anyio
@pytest.mark.parametrize(("threshold", "passed"), [(85, True), (86, False)])
async def test_routing_uses_configured_guardrail_threshold(threshold: int, passed: bool) -> None:
    deps, _, retrieval = dependencies(llm=FakeLLMProvider('{"score":85}', '{"score":85}'))

    result = await run(deps, config=AgentGraphConfig(guardrail_threshold=threshold))

    assert result["guardrail_passed"] is passed
    assert retrieval.search.call_count == (1 if passed else 0)
    assert result["terminal_reason"] == (None if passed else TerminalReason.OUT_OF_SCOPE)


def test_graph_compilation_has_conditional_m03_topology() -> None:
    deps, _, _ = dependencies()

    graph = build_agent_graph(dependencies=deps).get_graph()

    assert set(graph.nodes) == {
        "__start__",
        "graph_start",
        "guardrail",
        "out_of_scope",
        "local_retrieval",
        "evidence_grading",
        "graph_complete",
        "__end__",
    }
    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "graph_start", False),
        ("graph_start", "guardrail", False),
        ("guardrail", "local_retrieval", True),
        ("guardrail", "out_of_scope", True),
        ("guardrail", "__end__", True),
        ("out_of_scope", "__end__", False),
        ("local_retrieval", "evidence_grading", True),
        ("local_retrieval", "__end__", True),
        ("evidence_grading", "graph_complete", True),
        ("evidence_grading", "__end__", True),
        ("graph_complete", "__end__", False),
    }


@pytest.mark.anyio
async def test_every_path_has_exactly_one_terminal_event_and_contiguous_sequences() -> None:
    cases = [
        dependencies()[0],
        dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":5}'))[0],
        dependencies(result=retrieval_result(hits=[]))[0],
        dependencies(llm=FakeLLMProvider('{"score":1}'))[0],
        dependencies(llm=FakeLLMProvider("invalid"))[0],
        dependencies(retrieval_error=RuntimeError("down"))[0],
        dependencies(llm=FakeLLMProvider('{"score":90}', "invalid"))[0],
        dependencies(llm=FakeLLMProvider('{"score":90}', RuntimeError("down")))[0],
    ]

    for deps in cases:
        result = await run(deps)
        path = statuses(result)
        assert path.count(S.GRAPH_COMPLETED) + path.count(S.GRAPH_FAILED) == 1
        assert path[-1] in {S.GRAPH_COMPLETED, S.GRAPH_FAILED}
        assert [event.sequence for event in result["execution_events"]] == list(range(len(path)))


@pytest.mark.anyio
@pytest.mark.parametrize("grader_response", ['{"score":85}', '{"score":20}', '{"score":85,"reasoning":"secret rationale"}'])
async def test_events_are_deterministic_and_content_free(grader_response: str) -> None:
    question = "How is NLP used in clinical decision support?"

    async def events():
        deps, _, _ = dependencies(llm=FakeLLMProvider('{"score":95}', grader_response))
        result = await run(deps, question)
        return [event.model_dump(mode="json") for event in result["execution_events"]]

    first, second = await events(), await events()

    assert first == second
    assert first[2]["metadata"]["guardrail_score"] == 95
    assert first[4]["metadata"]["source_count"] == 1
    serialized = repr(first)
    for forbidden in (
        question,
        EVIDENCE_TEXT,
        "Private Paper Title",
        "UNTRUSTED_QUESTION_JSON",
        "UNTRUSTED_GRADING_INPUT_JSON",
        "evidence-sufficiency grader",
        grader_response,
        "secret rationale",
    ):
        assert forbidden not in serialized
