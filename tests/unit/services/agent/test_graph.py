from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.agent import (
    AgentErrorCategory,
    AgentExecutionStatus,
    AgentGraphConfig,
    AgentGraphDependencies,
    TerminalReason,
    build_agent_graph,
    create_initial_agent_state,
)
from src.services.search.hybrid_service import HybridSearchService


class FakeLLMProvider:
    def __init__(self, content: str = '{"score":95}', error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def retrieval_result(mode: str = "hybrid") -> HybridSearchResult:
    return HybridSearchResult(query="question", retrieval_mode=mode, count=0, results=[])


def dependencies(*, llm=None, result=None, retrieval_error=None):
    provider = llm or FakeLLMProvider()
    retrieval = Mock(spec=HybridSearchService)
    retrieval.search.return_value = result or retrieval_result()
    retrieval.search.side_effect = retrieval_error
    return AgentGraphDependencies(llm_provider=provider, hybrid_search_service=retrieval), provider, retrieval


def statuses(result) -> list[AgentExecutionStatus]:
    return [event.status for event in result["execution_events"]]


@pytest.mark.anyio
async def test_guardrail_pass_retrieves_once_with_config_and_stores_candidates() -> None:
    expected = retrieval_result()
    deps, provider, retrieval = dependencies(result=expected)
    config = AgentGraphConfig(retrieval_size=7)
    question = "How can AI improve healthcare access?"

    result = await build_agent_graph(dependencies=deps, config=config).ainvoke(create_initial_agent_state(question))

    assert statuses(result) == [
        AgentExecutionStatus.GRAPH_STARTED,
        AgentExecutionStatus.GUARDRAIL_STARTED,
        AgentExecutionStatus.GUARDRAIL_PASSED,
        AgentExecutionStatus.LOCAL_RETRIEVAL_STARTED,
        AgentExecutionStatus.LOCAL_RETRIEVAL_COMPLETED,
        AgentExecutionStatus.GRAPH_COMPLETED,
    ]
    assert [event.sequence for event in result["execution_events"]] == list(range(6))
    assert provider.calls == 1
    retrieval.search.assert_called_once_with(question, size=7)
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] == expected
    assert result["guardrail_score"] == 95
    assert result["guardrail_passed"] is True
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["generated_answer"] is None
    assert result["terminal_reason"] is None
    assert result["error_category"] is None


@pytest.mark.anyio
async def test_guardrail_rejection_is_normal_completion_and_skips_retrieval() -> None:
    deps, _, retrieval = dependencies(llm=FakeLLMProvider('{"score":10}'))

    result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state("What is a dog?"))

    assert statuses(result) == [
        AgentExecutionStatus.GRAPH_STARTED,
        AgentExecutionStatus.GUARDRAIL_STARTED,
        AgentExecutionStatus.GUARDRAIL_REJECTED,
        AgentExecutionStatus.GRAPH_COMPLETED,
    ]
    assert result["guardrail_score"] == 10
    assert result["guardrail_passed"] is False
    assert result["retrieval_attempts"] == 0
    assert result["local_retrieval_result"] is None
    assert result["terminal_reason"] == TerminalReason.OUT_OF_SCOPE
    assert result["error_category"] is None
    retrieval.search.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider",
    [FakeLLMProvider(error=RuntimeError("private")), FakeLLMProvider("not json")],
)
async def test_guardrail_failure_is_internal_and_safe(provider: FakeLLMProvider) -> None:
    deps, _, retrieval = dependencies(llm=provider)

    result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state("Question"))

    assert statuses(result) == [
        AgentExecutionStatus.GRAPH_STARTED,
        AgentExecutionStatus.GUARDRAIL_STARTED,
        AgentExecutionStatus.GRAPH_FAILED,
    ]
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.GUARDRAIL_FAILURE
    assert result["guardrail_passed"] is None
    assert result["retrieval_attempts"] == 0
    retrieval.search.assert_not_called()
    assert "private" not in repr(result["execution_events"])


@pytest.mark.anyio
async def test_retrieval_failure_is_not_insufficient_evidence() -> None:
    deps, _, retrieval = dependencies(retrieval_error=RuntimeError("private retrieval detail"))

    result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state("AI question"))

    assert statuses(result) == [
        AgentExecutionStatus.GRAPH_STARTED,
        AgentExecutionStatus.GUARDRAIL_STARTED,
        AgentExecutionStatus.GUARDRAIL_PASSED,
        AgentExecutionStatus.LOCAL_RETRIEVAL_STARTED,
        AgentExecutionStatus.GRAPH_FAILED,
    ]
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["terminal_reason"] == TerminalReason.RETRIEVAL_FAILED
    assert result["error_category"] == AgentErrorCategory.RETRIEVAL_FAILURE
    retrieval.search.assert_called_once()
    assert "private retrieval detail" not in repr(result["execution_events"])


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["hybrid", "bm25_fallback", "vector_fallback"])
async def test_every_valid_hybrid_mode_is_successful(mode: str) -> None:
    expected = retrieval_result(mode)
    deps, _, _ = dependencies(result=expected)

    result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state("AI question"))

    assert result["local_retrieval_result"] == expected
    assert AgentExecutionStatus.GRAPH_COMPLETED in statuses(result)
    assert AgentExecutionStatus.GRAPH_FAILED not in statuses(result)


@pytest.mark.anyio
async def test_sync_retrieval_is_offloaded_without_timing_assertions() -> None:
    expected = retrieval_result()
    deps, _, retrieval = dependencies(result=expected)
    question = "AI question"

    with patch("src.services.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=expected)) as offload:
        result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state(question))

    offload.assert_awaited_once_with(retrieval.search, question, size=5)
    retrieval.search.assert_not_called()
    assert result["local_retrieval_result"] == expected


def test_graph_compilation_has_conditional_m02_topology() -> None:
    deps, _, _ = dependencies()

    graph = build_agent_graph(dependencies=deps).get_graph()

    assert set(graph.nodes) == {"__start__", "graph_start", "guardrail", "out_of_scope", "local_retrieval", "__end__"}
    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "graph_start", False),
        ("graph_start", "guardrail", False),
        ("guardrail", "local_retrieval", True),
        ("guardrail", "out_of_scope", True),
        ("guardrail", "__end__", True),
        ("out_of_scope", "__end__", False),
        ("local_retrieval", "__end__", False),
    }


@pytest.mark.anyio
async def test_retrieval_uses_current_query_rather_than_original_question() -> None:
    deps, _, retrieval = dependencies()
    state = create_initial_agent_state("Original AI question")
    state["current_query"] = "Current AI query"

    await build_agent_graph(dependencies=deps).ainvoke(state)

    retrieval.search.assert_called_once_with("Current AI query", size=5)


@pytest.mark.anyio
@pytest.mark.parametrize(("threshold", "passed"), [(85, True), (86, False)])
async def test_routing_uses_configured_guardrail_threshold(threshold: int, passed: bool) -> None:
    deps, _, retrieval = dependencies(llm=FakeLLMProvider('{"score":85}'))
    config = AgentGraphConfig(guardrail_threshold=threshold)

    result = await build_agent_graph(dependencies=deps, config=config).ainvoke(create_initial_agent_state("Question"))

    assert result["guardrail_passed"] is passed
    assert retrieval.search.call_count == (1 if passed else 0)
    assert result["terminal_reason"] == (None if passed else TerminalReason.OUT_OF_SCOPE)


@pytest.mark.anyio
async def test_events_are_deterministic_and_content_free() -> None:
    question = "How is NLP used in clinical decision support?"
    hit = HybridSearchHit(
        chunk_id="c1",
        arxiv_id="2401.00001",
        chunk_index=0,
        chunk_text="private evidence text",
        word_count=3,
        rrf_score=0.5,
    )
    expected = HybridSearchResult(query=question, retrieval_mode="hybrid", count=1, results=[hit])

    async def run():
        deps, _, _ = dependencies(result=expected)
        result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state(question))
        return [event.model_dump(mode="json") for event in result["execution_events"]]

    first, second = await run(), await run()

    assert first == second
    assert first[2]["metadata"]["guardrail_score"] == 95
    assert first[4]["metadata"]["source_count"] == 1
    serialized = repr(first)
    assert question not in serialized
    assert "private evidence text" not in serialized
    assert "UNTRUSTED_QUESTION_JSON" not in serialized


@pytest.mark.anyio
async def test_no_path_emits_both_completed_and_failed() -> None:
    cases = [
        dependencies()[0],
        dependencies(llm=FakeLLMProvider('{"score":1}'))[0],
        dependencies(llm=FakeLLMProvider("invalid"))[0],
        dependencies(retrieval_error=RuntimeError("down"))[0],
    ]

    for deps in cases:
        result = await build_agent_graph(dependencies=deps).ainvoke(create_initial_agent_state("Question"))
        path = statuses(result)
        assert not ({AgentExecutionStatus.GRAPH_COMPLETED, AgentExecutionStatus.GRAPH_FAILED} <= set(path))
