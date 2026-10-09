"""LangGraph topology for domain guardrail and one local retrieval attempt."""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.services.agent.config import AgentGraphConfig
from src.services.agent.events import AgentExecutionEvent, AgentExecutionMetadata, AgentExecutionStatus
from src.services.agent.guardrail import GuardrailEvaluationError, GuardrailEvaluator
from src.services.agent.state import AgentState
from src.services.agent.types import AgentErrorCategory, TerminalReason
from src.services.llm import LLMProvider
from src.services.search.hybrid_service import HybridSearchService

GuardrailRoute = Literal["passed", "rejected", "failed"]


@dataclass(frozen=True)
class AgentGraphDependencies:
    """Worker-scoped services injected into graph nodes."""

    llm_provider: LLMProvider
    hybrid_search_service: HybridSearchService


def _next_sequence(state: AgentState) -> int:
    return len(state["execution_events"])


def _start_graph(state: AgentState) -> dict[str, Any]:
    return {
        "execution_events": [
            AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_STARTED, sequence=_next_sequence(state))
        ]
    }


def _make_guardrail_node(evaluator: GuardrailEvaluator, config: AgentGraphConfig):
    async def guardrail(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        started = AgentExecutionEvent(status=AgentExecutionStatus.GUARDRAIL_STARTED, sequence=sequence)
        try:
            result = await evaluator.evaluate(state["original_question"], threshold=config.guardrail_threshold)
        except GuardrailEvaluationError:
            return {
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.GUARDRAIL_FAILURE,
            }
        status = AgentExecutionStatus.GUARDRAIL_PASSED if result.passed else AgentExecutionStatus.GUARDRAIL_REJECTED
        return {
            "guardrail_score": result.score,
            "guardrail_passed": result.passed,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=status,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(
                        guardrail_score=result.score,
                        guardrail_passed=result.passed,
                    ),
                ),
            ],
        }

    return guardrail


def _route_after_guardrail(state: AgentState) -> GuardrailRoute:
    if state["error_category"] is not None:
        return "failed"
    return "passed" if state["guardrail_passed"] else "rejected"


def _complete_out_of_scope(state: AgentState) -> dict[str, Any]:
    return {
        "terminal_reason": TerminalReason.OUT_OF_SCOPE,
        "execution_events": [
            AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_COMPLETED, sequence=_next_sequence(state))
        ],
    }


def _make_local_retrieval_node(service: HybridSearchService, config: AgentGraphConfig):
    async def local_retrieval(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        attempt = state["retrieval_attempts"] + 1
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.LOCAL_RETRIEVAL_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(retrieval_attempt=attempt),
        )
        try:
            result = await asyncio.to_thread(
                service.search,
                state["current_query"],
                size=config.retrieval_size,
            )
        except Exception:
            return {
                "retrieval_attempts": attempt,
                "local_retrieval_result": None,
                "terminal_reason": TerminalReason.RETRIEVAL_FAILED,
                "error_category": AgentErrorCategory.RETRIEVAL_FAILURE,
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
            }
        return {
            "retrieval_attempts": attempt,
            "local_retrieval_result": result,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.LOCAL_RETRIEVAL_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(retrieval_attempt=attempt, source_count=result.count),
                ),
                AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_COMPLETED, sequence=sequence + 2),
            ],
        }

    return local_retrieval


def build_agent_graph(
    *,
    dependencies: AgentGraphDependencies,
    config: AgentGraphConfig | None = None,
) -> CompiledStateGraph:
    """Compile W7-M02 with injected services and no infrastructure creation."""

    graph_config = config or AgentGraphConfig()
    evaluator = GuardrailEvaluator(llm_provider=dependencies.llm_provider)

    builder = StateGraph(AgentState)
    builder.add_node("graph_start", _start_graph)
    builder.add_node("guardrail", _make_guardrail_node(evaluator, graph_config))
    builder.add_node("out_of_scope", _complete_out_of_scope)
    builder.add_node("local_retrieval", _make_local_retrieval_node(dependencies.hybrid_search_service, graph_config))
    builder.add_edge(START, "graph_start")
    builder.add_edge("graph_start", "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"passed": "local_retrieval", "rejected": "out_of_scope", "failed": END},
    )
    builder.add_edge("out_of_scope", END)
    builder.add_edge("local_retrieval", END)
    return builder.compile()
