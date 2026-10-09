"""LangGraph topology for guardrail, bounded local retrieval, grading, and one-shot query rewrite."""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.services.agent.config import AgentGraphConfig
from src.services.agent.events import AgentExecutionEvent, AgentExecutionMetadata, AgentExecutionStatus
from src.services.agent.evidence_grader import EvidenceGradingError, EvidenceSufficiencyGrader
from src.services.agent.guardrail import GuardrailEvaluationError, GuardrailEvaluator
from src.services.agent.query_rewriter import QueryRewriteError, QueryRewriter
from src.services.agent.state import AgentState
from src.services.agent.types import AgentErrorCategory, TerminalReason
from src.services.evidence import EvidenceContextBuilder
from src.services.llm import LLMProvider
from src.services.search.hybrid_service import HybridSearchService

GuardrailRoute = Literal["passed", "rejected", "failed"]
RetrievalRoute = Literal["retrieved", "failed"]
EvidenceRoute = Literal["sufficient", "rewrite", "exhausted", "failed"]
RewriteRoute = Literal["rewritten", "failed"]


@dataclass(frozen=True)
class AgentGraphDependencies:
    """Worker-scoped services injected into graph nodes."""

    llm_provider: LLMProvider
    hybrid_search_service: HybridSearchService
    evidence_context_builder: EvidenceContextBuilder


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
        if state["retrieval_attempts"] >= config.max_local_retrieval_attempts:
            # Routing prevents this; the node still refuses to exceed the configured total.
            return {
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.INTERNAL_FAILURE,
                "execution_events": [
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence)
                ],
            }
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
                "evidence_sufficient": None,
                "evidence_grade": None,
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
            "evidence_sufficient": None,
            "evidence_grade": None,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.LOCAL_RETRIEVAL_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(retrieval_attempt=attempt, source_count=result.count),
                ),
            ],
        }

    return local_retrieval


def _route_after_local_retrieval(state: AgentState) -> RetrievalRoute:
    return "failed" if state["error_category"] is not None else "retrieved"


def _make_evidence_grading_node(
    grader: EvidenceSufficiencyGrader,
    context_builder: EvidenceContextBuilder,
    config: AgentGraphConfig,
):
    async def evidence_grading(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        attempt = state["retrieval_attempts"]

        def failed(started: AgentExecutionEvent) -> dict[str, Any]:
            return {
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.EVIDENCE_GRADING_FAILURE,
            }

        try:
            context = context_builder.build(state["local_retrieval_result"])
        except Exception:
            return failed(
                AgentExecutionEvent(
                    status=AgentExecutionStatus.EVIDENCE_GRADING_STARTED,
                    sequence=sequence,
                    metadata=AgentExecutionMetadata(retrieval_attempt=attempt),
                )
            )
        source_count = len(context.sources)
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.EVIDENCE_GRADING_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(retrieval_attempt=attempt, source_count=source_count),
        )
        try:
            grade = await grader.grade(
                state["original_question"],
                context,
                threshold=config.evidence_sufficiency_threshold,
            )
        except EvidenceGradingError:
            return failed(started)
        status = (
            AgentExecutionStatus.EVIDENCE_SUFFICIENT if grade.sufficient else AgentExecutionStatus.EVIDENCE_INSUFFICIENT
        )
        return {
            "evidence_sufficient": grade.sufficient,
            "evidence_grade": grade,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=status,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(
                        retrieval_attempt=attempt,
                        source_count=source_count,
                        evidence_score=grade.score,
                        evidence_sufficient=grade.sufficient,
                    ),
                ),
            ],
        }

    return evidence_grading


def _make_evidence_router(config: AgentGraphConfig):
    def route_after_evidence_grading(state: AgentState) -> EvidenceRoute:
        if state["error_category"] is not None:
            return "failed"
        if state["evidence_sufficient"]:
            return "sufficient"
        if state["retrieval_attempts"] < config.max_local_retrieval_attempts:
            return "rewrite"
        return "exhausted"

    return route_after_evidence_grading


def _make_query_rewrite_node(rewriter: QueryRewriter):
    async def query_rewrite(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        attempt = state["retrieval_attempts"]
        metadata = AgentExecutionMetadata(retrieval_attempt=attempt, next_retrieval_attempt=attempt + 1)
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.QUERY_REWRITE_STARTED,
            sequence=sequence,
            metadata=metadata,
        )
        try:
            result = await rewriter.rewrite(state["original_question"], state["current_query"])
        except QueryRewriteError:
            return {
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.QUERY_REWRITE_FAILURE,
            }
        # The previous result and grade describe the previous query, not the one about to run.
        return {
            "rewritten_query": result.query,
            "current_query": result.query,
            "local_retrieval_result": None,
            "evidence_sufficient": None,
            "evidence_grade": None,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.QUERY_REWRITTEN,
                    sequence=sequence + 1,
                    metadata=metadata,
                ),
            ],
        }

    return query_rewrite


def _route_after_query_rewrite(state: AgentState) -> RewriteRoute:
    return "failed" if state["error_category"] is not None else "rewritten"


def _complete_graph(state: AgentState) -> dict[str, Any]:
    return {
        "execution_events": [
            AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_COMPLETED, sequence=_next_sequence(state))
        ]
    }


def build_agent_graph(
    *,
    dependencies: AgentGraphDependencies,
    config: AgentGraphConfig | None = None,
) -> CompiledStateGraph:
    """Compile W7-M04 with injected services and no infrastructure creation."""

    graph_config = config or AgentGraphConfig()
    evaluator = GuardrailEvaluator(llm_provider=dependencies.llm_provider)
    grader = EvidenceSufficiencyGrader(llm_provider=dependencies.llm_provider)
    rewriter = QueryRewriter(llm_provider=dependencies.llm_provider)

    builder = StateGraph(AgentState)
    builder.add_node("graph_start", _start_graph)
    builder.add_node("guardrail", _make_guardrail_node(evaluator, graph_config))
    builder.add_node("out_of_scope", _complete_out_of_scope)
    builder.add_node("local_retrieval", _make_local_retrieval_node(dependencies.hybrid_search_service, graph_config))
    builder.add_node(
        "evidence_grading",
        _make_evidence_grading_node(grader, dependencies.evidence_context_builder, graph_config),
    )
    builder.add_node("query_rewrite", _make_query_rewrite_node(rewriter))
    builder.add_node("graph_complete", _complete_graph)
    builder.add_edge(START, "graph_start")
    builder.add_edge("graph_start", "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"passed": "local_retrieval", "rejected": "out_of_scope", "failed": END},
    )
    builder.add_edge("out_of_scope", END)
    builder.add_conditional_edges(
        "local_retrieval",
        _route_after_local_retrieval,
        {"retrieved": "evidence_grading", "failed": END},
    )
    builder.add_conditional_edges(
        "evidence_grading",
        _make_evidence_router(graph_config),
        {
            "sufficient": "graph_complete",
            "rewrite": "query_rewrite",
            "exhausted": "graph_complete",
            "failed": END,
        },
    )
    builder.add_conditional_edges(
        "query_rewrite",
        _route_after_query_rewrite,
        {"rewritten": "local_retrieval", "failed": END},
    )
    builder.add_edge("graph_complete", END)
    return builder.compile()
