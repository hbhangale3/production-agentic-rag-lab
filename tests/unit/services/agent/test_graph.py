import dataclasses
from datetime import UTC, datetime
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
    LiveArxivPaper,
    LiveArxivSearchResult,
    LiveSearchError,
    TerminalReason,
    build_agent_graph,
    create_initial_agent_state,
)
from src.services.agent.query_rewriter import MAX_REWRITTEN_QUERY_CHARACTERS
from src.services.evidence import EvidenceContextBuilder
from src.services.search.hybrid_service import HybridSearchService

S = AgentExecutionStatus
EVIDENCE_TEXT = "private evidence text about clinical NLP"
INDIA_QUESTION = "What are the current healthcare issues in India and how can AI help solve them?"
REWRITTEN = "India healthcare challenges artificial intelligence health equity"
REWRITE_JSON = f'{{"query":"{REWRITTEN}"}}'
SINGLE_ATTEMPT = AgentGraphConfig(max_local_retrieval_attempts=1, live_fallback_enabled=False)
LOCAL_ONLY = AgentGraphConfig(live_fallback_enabled=False)

GUARDRAIL_PASSED = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GUARDRAIL_PASSED]
RETRIEVED = [*GUARDRAIL_PASSED, S.LOCAL_RETRIEVAL_STARTED, S.LOCAL_RETRIEVAL_COMPLETED]
SUFFICIENT_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_SUFFICIENT, S.GRAPH_COMPLETED]
INSUFFICIENT_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT, S.GRAPH_COMPLETED]
GRADER_FAILED_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.GRAPH_FAILED]
REJECTED_PATH = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GUARDRAIL_REJECTED, S.GRAPH_COMPLETED]
GUARDRAIL_FAILED_PATH = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GRAPH_FAILED]
RETRIEVAL_FAILED_PATH = [*GUARDRAIL_PASSED, S.LOCAL_RETRIEVAL_STARTED, S.GRAPH_FAILED]
FIRST_INSUFFICIENT = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT]
REWRITTEN_PATH = [*FIRST_INSUFFICIENT, S.QUERY_REWRITE_STARTED, S.QUERY_REWRITTEN]
SECOND_RETRIEVED = [*REWRITTEN_PATH, S.LOCAL_RETRIEVAL_STARTED, S.LOCAL_RETRIEVAL_COMPLETED]
RETRY_SUFFICIENT_PATH = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_SUFFICIENT, S.GRAPH_COMPLETED]
RETRY_INSUFFICIENT_PATH = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT, S.GRAPH_COMPLETED]
REWRITE_FAILED_PATH = [*FIRST_INSUFFICIENT, S.QUERY_REWRITE_STARTED, S.GRAPH_FAILED]
SECOND_RETRIEVAL_FAILED_PATH = [*REWRITTEN_PATH, S.LOCAL_RETRIEVAL_STARTED, S.GRAPH_FAILED]
SECOND_GRADER_FAILED_PATH = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.GRAPH_FAILED]
LOCAL_EXHAUSTED = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT]
LIVE_COMPLETED_PATH = [*LOCAL_EXHAUSTED, S.LIVE_FALLBACK_STARTED, S.LIVE_FALLBACK_COMPLETED, S.GRAPH_COMPLETED]
LIVE_FAILED_PATH = [*LOCAL_EXHAUSTED, S.LIVE_FALLBACK_STARTED, S.GRAPH_FAILED]


class FakeLLMProvider:
    """Returns scripted completions in call order: guardrail, grader, rewrite, grader."""

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


def make_live_paper(arxiv_id: str = "2501.00001") -> LiveArxivPaper:
    return LiveArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Private Live Title {arxiv_id}",
        abstract="private live abstract",
        authors=("Private Author",),
        categories=("cs.CY",),
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        updated_date=None,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def live_result(*arxiv_ids: str) -> LiveArxivSearchResult:
    return LiveArxivSearchResult(query="live query", candidates=tuple(make_live_paper(i) for i in arxiv_ids))


def dependencies(
    *, llm=None, result=None, results=None, retrieval_error=None, max_context_tokens=1000, live=None, live_error=None
):
    """``results`` scripts successive search calls; an exception item is raised."""
    live_service = Mock()
    live_service.search = AsyncMock(
        return_value=live_result("2501.00001") if live is None else live, side_effect=live_error
    )
    provider = llm or FakeLLMProvider()
    retrieval = Mock(spec=HybridSearchService)
    retrieval.search.return_value = result or retrieval_result()
    retrieval.search.side_effect = results if results is not None else retrieval_error
    deps = AgentGraphDependencies(
        llm_provider=provider,
        hybrid_search_service=retrieval,
        evidence_context_builder=EvidenceContextBuilder(max_context_tokens=max_context_tokens),
        live_search_service=live_service,
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
    assert result["live_search_result"] is None
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
async def test_single_attempt_config_completes_insufficient_without_rewrite_or_retry() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":25}'))

    result = await run(deps, INDIA_QUESTION, SINGLE_ATTEMPT)

    assert statuses(result) == INSUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(8))
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=25, sufficient=False, source_count=1)
    assert result["current_query"] == INDIA_QUESTION
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    deps.live_search_service.search.assert_not_awaited()
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

    result = await run(
        deps,
        config=AgentGraphConfig(
            evidence_sufficiency_threshold=threshold,
            max_local_retrieval_attempts=1,
            live_fallback_enabled=False,
        ),
    )

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

    result = await run(deps, config=SINGLE_ATTEMPT)

    assert statuses(result) == INSUFFICIENT_PATH
    assert len(provider.calls) == 1
    retrieval.search.assert_called_once()
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=0, sufficient=False, source_count=0)
    assert result["execution_events"][6].metadata.evidence_score == 0
    assert result["execution_events"][6].metadata.source_count == 0
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
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
    deps = dataclasses.replace(deps, evidence_context_builder=builder)

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


def test_graph_compilation_has_conditional_m05_topology() -> None:
    deps, _, _ = dependencies()

    graph = build_agent_graph(dependencies=deps).get_graph()

    assert set(graph.nodes) == {
        "__start__",
        "graph_start",
        "guardrail",
        "out_of_scope",
        "local_retrieval",
        "evidence_grading",
        "query_rewrite",
        "live_arxiv_search",
        "insufficient_evidence",
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
        ("evidence_grading", "query_rewrite", True),
        ("evidence_grading", "__end__", True),
        ("query_rewrite", "local_retrieval", True),
        ("query_rewrite", "__end__", True),
        ("evidence_grading", "live_arxiv_search", True),
        ("evidence_grading", "insufficient_evidence", True),
        ("live_arxiv_search", "graph_complete", True),
        ("live_arxiv_search", "insufficient_evidence", True),
        ("live_arxiv_search", "__end__", True),
        ("insufficient_evidence", "__end__", False),
        ("graph_complete", "__end__", False),
    }


def assert_no_downstream_progress_after_rewrite(result) -> None:
    assert result["generated_answer"] is None
    assert result["grounding_passed"] is None
    assert result["live_fallback_used"] is False
    assert result["live_search_result"] is None
    assert result["live_evidence"] is None
    assert result["final_evidence"] is None


def retry_dependencies(*grader_and_rewrite: str | Exception, results=None):
    """Guardrail passes and attempt 1 grades 25; the remaining responses are scripted."""
    first, second = retrieval_result(hits=[make_hit("a1", "attempt one evidence")]), retrieval_result(
        hits=[make_hit("b1", "attempt two evidence"), make_hit("b2", "more attempt two evidence")]
    )
    llm = FakeLLMProvider('{"score":90}', '{"score":25}', *grader_and_rewrite)
    deps, provider, retrieval = dependencies(llm=llm, results=[first, second] if results is None else results)
    return deps, provider, retrieval, first, second


def exhausted_dependencies(*, live=None, live_error=None):
    """Both local attempts grade insufficient, so the default config reaches live fallback."""
    first, second = retrieval_result(hits=[make_hit("a1", "attempt one evidence")]), retrieval_result(
        hits=[make_hit("b1", "attempt two evidence")]
    )
    llm = FakeLLMProvider('{"score":90}', '{"score":25}', REWRITE_JSON, '{"score":40}')
    deps, provider, retrieval = dependencies(llm=llm, results=[first, second], live=live, live_error=live_error)
    return deps, provider, retrieval, second


def user_payload(call) -> str:
    return call[0][1].content


@pytest.mark.anyio
async def test_retry_then_sufficient_uses_rewritten_query_and_grades_original_question() -> None:
    deps, provider, retrieval, _, second = retry_dependencies(REWRITE_JSON, '{"score":88}')

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == RETRY_SUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(14))
    assert [call.args[0] for call in retrieval.search.call_args_list] == [INDIA_QUESTION, REWRITTEN]
    assert len(provider.calls) == 4
    guardrail_call, first_grade, rewrite, second_grade = provider.calls
    assert user_payload(rewrite).startswith("UNTRUSTED_REWRITE_INPUT_JSON\n")
    for grade_call in (first_grade, second_grade):
        assert user_payload(grade_call).startswith("UNTRUSTED_GRADING_INPUT_JSON\n")
        assert INDIA_QUESTION in user_payload(grade_call)
        assert REWRITTEN not in user_payload(grade_call)
    assert "attempt one evidence" in user_payload(first_grade)
    assert "attempt two evidence" in user_payload(second_grade)
    assert "attempt one evidence" not in user_payload(second_grade)
    assert result["original_question"] == INDIA_QUESTION
    assert result["current_query"] == REWRITTEN
    assert result["rewritten_query"] == REWRITTEN
    assert result["retrieval_attempts"] == 2
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is True
    assert result["evidence_grade"] == EvidenceGrade(score=88, sufficient=True, source_count=2)
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert result["generated_answer"] is None
    deps.live_search_service.search.assert_not_awaited()
    assert_no_downstream_progress_after_rewrite(result)


@pytest.mark.anyio
async def test_exhausted_with_fallback_disabled_is_insufficient_evidence_without_live_search() -> None:
    deps, provider, retrieval, _, second = retry_dependencies(REWRITE_JSON, '{"score":40}')

    result = await run(deps, INDIA_QUESTION, LOCAL_ONLY)

    assert statuses(result) == RETRY_INSUFFICIENT_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 4
    assert statuses(result).count(S.QUERY_REWRITTEN) == 1
    assert result["retrieval_attempts"] == 2
    assert result["rewritten_query"] == REWRITTEN
    assert result["current_query"] == REWRITTEN
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=40, sufficient=False, source_count=2)
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    deps.live_search_service.search.assert_not_awaited()
    assert_no_downstream_progress_after_rewrite(result)


@pytest.mark.anyio
async def test_attempt_metadata_distinguishes_first_and_second_attempts() -> None:
    deps, _, _, _, _ = retry_dependencies(REWRITE_JSON, '{"score":88}')

    result = await run(deps)

    metadata = [
        (event.status, event.metadata.model_dump(exclude_none=True)) for event in result["execution_events"][3:14]
    ]
    assert metadata == [
        (S.LOCAL_RETRIEVAL_STARTED, {"retrieval_attempt": 1}),
        (S.LOCAL_RETRIEVAL_COMPLETED, {"retrieval_attempt": 1, "source_count": 1}),
        (S.EVIDENCE_GRADING_STARTED, {"retrieval_attempt": 1, "source_count": 1}),
        (
            S.EVIDENCE_INSUFFICIENT,
            {"retrieval_attempt": 1, "source_count": 1, "evidence_sufficient": False, "evidence_score": 25},
        ),
        (S.QUERY_REWRITE_STARTED, {"retrieval_attempt": 1, "next_retrieval_attempt": 2}),
        (S.QUERY_REWRITTEN, {"retrieval_attempt": 1, "next_retrieval_attempt": 2}),
        (S.LOCAL_RETRIEVAL_STARTED, {"retrieval_attempt": 2}),
        (S.LOCAL_RETRIEVAL_COMPLETED, {"retrieval_attempt": 2, "source_count": 2}),
        (S.EVIDENCE_GRADING_STARTED, {"retrieval_attempt": 2, "source_count": 2}),
        (
            S.EVIDENCE_SUFFICIENT,
            {"retrieval_attempt": 2, "source_count": 2, "evidence_sufficient": True, "evidence_score": 88},
        ),
        (S.GRAPH_COMPLETED, {}),
    ]


@pytest.mark.anyio
async def test_empty_first_retrieval_is_rewritten_and_retried_under_default_config() -> None:
    empty = retrieval_result(hits=[])
    deps, provider, retrieval = dependencies(
        llm=FakeLLMProvider('{"score":90}', REWRITE_JSON), results=[empty, empty]
    )

    result = await run(deps, config=LOCAL_ONLY)

    assert statuses(result) == RETRY_INSUFFICIENT_PATH
    assert len(provider.calls) == 2
    assert retrieval.search.call_count == 2
    assert result["retrieval_attempts"] == 2
    assert result["evidence_grade"] == EvidenceGrade(score=0, sufficient=False, source_count=0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "rewrite_response",
    [
        RuntimeError("private rewrite detail"),
        "private rewrite detail, not json",
        '{"query":"new query","reason":"private rewrite detail"}',
        '{"query":"   "}',
        '{"query":"  ai   QUESTION "}',
        f'{{"query":"{"x" * (MAX_REWRITTEN_QUERY_CHARACTERS + 1)}"}}',
    ],
)
async def test_rewrite_failure_is_controlled_and_skips_second_retrieval(rewrite_response) -> None:
    deps, provider, retrieval, first, _ = retry_dependencies(rewrite_response)

    result = await run(deps, "AI question")

    assert statuses(result) == REWRITE_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert len(provider.calls) == 3
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["rewritten_query"] is None
    assert result["current_query"] == "AI question"
    assert result["local_retrieval_result"] == first
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"].score == 25
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.QUERY_REWRITE_FAILURE
    serialized = repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private rewrite detail" not in serialized
    assert "new query" not in serialized


@pytest.mark.anyio
async def test_second_retrieval_failure_counts_as_attempt_two_and_skips_grading() -> None:
    first = retrieval_result()
    deps, provider, retrieval, _, _ = retry_dependencies(
        REWRITE_JSON, results=[first, RuntimeError("private retrieval detail")]
    )

    result = await run(deps)

    assert statuses(result) == SECOND_RETRIEVAL_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 3
    assert result["retrieval_attempts"] == 2
    assert result["current_query"] == REWRITTEN
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.RETRIEVAL_FAILED
    assert result["error_category"] == AgentErrorCategory.RETRIEVAL_FAILURE
    assert "private retrieval detail" not in repr(result["execution_events"])


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["provider", "parser", "context_builder"])
async def test_second_grading_failure_does_not_restore_first_grade_or_retry(failure: str) -> None:
    grader_response = {"provider": RuntimeError("private grader detail"), "parser": "private grader detail"}.get(
        failure, '{"score":99}'
    )
    deps, provider, retrieval, _, second = retry_dependencies(REWRITE_JSON, grader_response)
    if failure == "context_builder":
        real = deps.evidence_context_builder
        builder = Mock(spec=EvidenceContextBuilder)
        builder.build.side_effect = [real.build(retrieval_result()), RuntimeError("private grader detail")]
        deps = dataclasses.replace(deps, evidence_context_builder=builder)

    result = await run(deps)

    assert statuses(result) == SECOND_GRADER_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert retrieval.search.call_count == 2
    assert statuses(result).count(S.QUERY_REWRITE_STARTED) == 1
    assert len(provider.calls) == (3 if failure == "context_builder" else 4)
    assert result["retrieval_attempts"] == 2
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert "private grader detail" not in repr(result["execution_events"])


@pytest.mark.anyio
async def test_first_attempt_grade_is_cleared_before_second_attempt_is_graded() -> None:
    deps, _, _, first, second = retry_dependencies(REWRITE_JSON, '{"score":88}')
    graph = build_agent_graph(dependencies=deps)

    snapshots = {}
    async for state in graph.astream(create_initial_agent_state("AI question"), stream_mode="values"):
        if state["execution_events"]:
            snapshots[len(state["execution_events"])] = state

    after_first_grade = snapshots[len(FIRST_INSUFFICIENT)]
    assert after_first_grade["local_retrieval_result"] == first
    assert after_first_grade["evidence_sufficient"] is False
    assert after_first_grade["evidence_grade"].score == 25

    after_rewrite = snapshots[len(REWRITTEN_PATH)]
    assert after_rewrite["current_query"] == REWRITTEN
    assert after_rewrite["retrieval_attempts"] == 1
    assert after_rewrite["local_retrieval_result"] is None
    assert after_rewrite["evidence_sufficient"] is None
    assert after_rewrite["evidence_grade"] is None

    after_second_retrieval = snapshots[len(SECOND_RETRIEVED)]
    assert after_second_retrieval["retrieval_attempts"] == 2
    assert after_second_retrieval["local_retrieval_result"] == second
    assert after_second_retrieval["evidence_sufficient"] is None
    assert after_second_retrieval["evidence_grade"] is None

    final = snapshots[len(RETRY_SUFFICIENT_PATH)]
    assert final["evidence_grade"] == EvidenceGrade(score=88, sufficient=True, source_count=2)


@pytest.mark.anyio
async def test_total_attempts_follow_configuration_and_never_exceed_it() -> None:
    llm = FakeLLMProvider(
        '{"score":90}',
        '{"score":10}',
        '{"query":"first rewrite"}',
        '{"score":10}',
        '{"query":"second rewrite"}',
        '{"score":10}',
    )
    deps, provider, retrieval = dependencies(llm=llm)

    result = await run(deps, config=AgentGraphConfig(max_local_retrieval_attempts=3, live_fallback_enabled=False))

    assert [call.args[0] for call in retrieval.search.call_args_list] == [
        "AI question",
        "first rewrite",
        "second rewrite",
    ]
    assert len(provider.calls) == 6
    assert result["retrieval_attempts"] == 3
    assert result["rewritten_query"] == "second rewrite"
    assert result["evidence_sufficient"] is False
    assert statuses(result)[-1] == S.GRAPH_COMPLETED
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE


@pytest.mark.anyio
@pytest.mark.parametrize(("attempts", "maximum"), [(2, 2), (3, 2), (1, 1)])
async def test_retrieval_node_refuses_to_exceed_configured_attempts(attempts: int, maximum: int) -> None:
    deps, provider, retrieval = dependencies()
    state = create_initial_agent_state("AI question")
    state["retrieval_attempts"] = attempts
    config = AgentGraphConfig(max_local_retrieval_attempts=maximum)

    result = await build_agent_graph(dependencies=deps, config=config).ainvoke(state)

    retrieval.search.assert_not_called()
    assert statuses(result) == [*GUARDRAIL_PASSED, S.GRAPH_FAILED]
    assert len(provider.calls) == 1
    assert result["retrieval_attempts"] == attempts
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.INTERNAL_FAILURE


@pytest.mark.anyio
async def test_live_fallback_discovers_candidates_after_local_exhaustion() -> None:
    candidates = live_result("2501.00001", "2501.00002", "2501.00003")
    deps, provider, retrieval, second = exhausted_dependencies(live=candidates)

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == LIVE_COMPLETED_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(16))
    assert S.GRAPH_COMPLETED not in statuses(result)[:-1]
    deps.live_search_service.search.assert_awaited_once_with(
        REWRITTEN, max_results=5, exclude_arxiv_ids=("2401.00001",)
    )
    assert [call.args[0] for call in retrieval.search.call_args_list] == [INDIA_QUESTION, REWRITTEN]
    assert len(provider.calls) == 4
    assert statuses(result).count(S.QUERY_REWRITTEN) == 1
    assert result["original_question"] == INDIA_QUESTION
    assert result["current_query"] == REWRITTEN
    assert result["retrieval_attempts"] == 2
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] == candidates
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=40, sufficient=False, source_count=1)
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert result["live_evidence"] is None
    assert result["final_evidence"] is None
    assert result["generated_answer"] is None
    started, completed = result["execution_events"][13:15]
    assert started.metadata.model_dump(exclude_none=True) == {"live_fallback_used": True, "max_results": 5}
    assert completed.metadata.model_dump(exclude_none=True) == {
        "live_fallback_used": True,
        "max_results": 5,
        "candidate_count": 3,
    }


@pytest.mark.anyio
async def test_live_search_uses_configured_bound_even_if_service_returns_more() -> None:
    deps, _, _, _ = exhausted_dependencies(live=live_result("2501.00001", "2501.00002", "2501.00003"))

    result = await run(deps, config=AgentGraphConfig(live_arxiv_max_results=2))

    assert deps.live_search_service.search.await_args.kwargs["max_results"] == 2
    assert [paper.arxiv_id for paper in result["live_search_result"].candidates] == ["2501.00001", "2501.00002"]
    assert result["execution_events"][14].metadata.candidate_count == 2


@pytest.mark.anyio
async def test_single_attempt_config_falls_back_live_with_unrewritten_query() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":25}'))

    result = await run(deps, INDIA_QUESTION, AgentGraphConfig(max_local_retrieval_attempts=1))

    assert statuses(result) == [
        *FIRST_INSUFFICIENT,
        S.LIVE_FALLBACK_STARTED,
        S.LIVE_FALLBACK_COMPLETED,
        S.GRAPH_COMPLETED,
    ]
    assert deps.live_search_service.search.await_args.args == (INDIA_QUESTION,)
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert result["rewritten_query"] is None
    assert result["live_fallback_used"] is True


@pytest.mark.anyio
async def test_live_search_with_zero_candidates_is_insufficient_evidence_not_failure() -> None:
    empty = live_result()
    deps, _, retrieval, second = exhausted_dependencies(live=empty)

    result = await run(deps)

    assert statuses(result) == LIVE_COMPLETED_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    deps.live_search_service.search.assert_awaited_once()
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] == empty
    assert result["live_search_result"].count == 0
    assert result["execution_events"][14].metadata.candidate_count == 0
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    assert retrieval.search.call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("error", [LiveSearchError("private live detail"), RuntimeError("private live detail")])
async def test_live_search_failure_is_distinct_from_zero_candidates(error: Exception) -> None:
    deps, provider, retrieval, second = exhausted_dependencies(live_error=error)

    result = await run(deps)

    assert statuses(result) == LIVE_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    deps.live_search_service.search.assert_awaited_once()
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] is None
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["retrieval_attempts"] == 2
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.LIVE_SEARCH_FAILURE
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 4
    assert "private live detail" not in repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private live detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )


@pytest.mark.anyio
async def test_live_search_is_skipped_when_first_attempt_is_sufficient() -> None:
    deps, _, _ = dependencies()

    result = await run(deps)

    assert statuses(result) == SUFFICIENT_PATH
    deps.live_search_service.search.assert_not_awaited()
    assert result["live_fallback_used"] is False
    assert result["live_search_result"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "llm",
    [
        FakeLLMProvider('{"score":1}'),
        FakeLLMProvider("invalid"),
        FakeLLMProvider('{"score":90}', "invalid"),
        FakeLLMProvider('{"score":90}', '{"score":25}', "invalid"),
    ],
)
async def test_live_search_is_never_reached_from_earlier_terminal_paths(llm: FakeLLMProvider) -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(*llm.responses))

    result = await run(deps)

    deps.live_search_service.search.assert_not_awaited()
    assert result["live_fallback_used"] is False
    assert S.LIVE_FALLBACK_STARTED not in statuses(result)


def test_enabled_live_fallback_requires_a_live_search_service() -> None:
    deps, provider, retrieval = dependencies()
    without_live = AgentGraphDependencies(
        llm_provider=provider,
        hybrid_search_service=retrieval,
        evidence_context_builder=deps.evidence_context_builder,
    )

    with pytest.raises(ValueError, match="live_search_service is required"):
        build_agent_graph(dependencies=without_live)

    graph = build_agent_graph(dependencies=without_live, config=LOCAL_ONLY).get_graph()
    assert "live_arxiv_search" not in graph.nodes
    assert "insufficient_evidence" in graph.nodes


def scenario_dependencies():
    """One (dependencies, config) pair per distinct graph path."""
    return [
        (dependencies()[0], None),
        (dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":5}'))[0], SINGLE_ATTEMPT),
        (dependencies(result=retrieval_result(hits=[]))[0], SINGLE_ATTEMPT),
        (dependencies(llm=FakeLLMProvider('{"score":1}'))[0], None),
        (dependencies(llm=FakeLLMProvider("invalid"))[0], None),
        (dependencies(retrieval_error=RuntimeError("down"))[0], None),
        (dependencies(llm=FakeLLMProvider('{"score":90}', "invalid"))[0], None),
        (dependencies(llm=FakeLLMProvider('{"score":90}', RuntimeError("down")))[0], None),
        (retry_dependencies(REWRITE_JSON, '{"score":88}')[0], None),
        (retry_dependencies(REWRITE_JSON, '{"score":40}')[0], None),
        (retry_dependencies("invalid")[0], None),
        (retry_dependencies(RuntimeError("down"))[0], None),
        (retry_dependencies(REWRITE_JSON, "invalid")[0], None),
        (retry_dependencies(REWRITE_JSON, results=[retrieval_result(), RuntimeError("down")])[0], None),
        (retry_dependencies(REWRITE_JSON, '{"score":40}')[0], LOCAL_ONLY),
        (exhausted_dependencies(live=live_result())[0], None),
        (exhausted_dependencies(live_error=LiveSearchError("down"))[0], None),
    ]


@pytest.mark.anyio
async def test_every_path_has_exactly_one_terminal_event_and_contiguous_sequences() -> None:
    for deps, config in scenario_dependencies():
        result = await run(deps, config=config)
        path = statuses(result)
        assert path.count(S.GRAPH_COMPLETED) + path.count(S.GRAPH_FAILED) == 1
        assert path[-1] in {S.GRAPH_COMPLETED, S.GRAPH_FAILED}
        assert [event.sequence for event in result["execution_events"]] == list(range(len(path)))
        assert path.count(S.LOCAL_RETRIEVAL_STARTED) <= 2
        assert path.count(S.QUERY_REWRITE_STARTED) <= 1
        assert path.count(S.LIVE_FALLBACK_STARTED) <= 1
        assert result["retrieval_attempts"] <= 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "responses",
    [
        ('{"score":85}',),
        ('{"score":85,"reasoning":"secret rationale"}',),
        ('{"score":20}', REWRITE_JSON, '{"score":88}'),
        ('{"score":20}', REWRITE_JSON, '{"score":30}'),
        ('{"score":20}', f'{{"query":"{REWRITTEN}","rationale":"secret rationale"}}'),
    ],
)
async def test_events_are_deterministic_and_content_free(responses: tuple[str, ...]) -> None:
    question = "How is NLP used in clinical decision support?"

    async def events():
        deps, _, _ = dependencies(llm=FakeLLMProvider('{"score":95}', *responses))
        result = await run(deps, question)
        return [event.model_dump(mode="json") for event in result["execution_events"]]

    first, second = await events(), await events()

    assert first == second
    assert first[2]["metadata"]["guardrail_score"] == 95
    assert first[4]["metadata"]["source_count"] == 1
    serialized = repr(first)
    for forbidden in (
        question,
        REWRITTEN,
        EVIDENCE_TEXT,
        "Private Paper Title",
        "Private Live Title",
        "private live abstract",
        "Private Author",
        "cs.CY",
        "arxiv.org",
        "live query",
        "UNTRUSTED_QUESTION_JSON",
        "UNTRUSTED_GRADING_INPUT_JSON",
        "UNTRUSTED_REWRITE_INPUT_JSON",
        "evidence-sufficiency grader",
        "retrieval-query rewriter",
        *responses,
        "secret rationale",
    ):
        assert forbidden not in serialized
