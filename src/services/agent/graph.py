"""LangGraph topology: guardrail, local retrieval and rewrite, grading, live fallback, rerank, generation."""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.exceptions import InsufficientEvidenceError
from src.services.agent.config import AgentGraphConfig
from src.services.agent.events import AgentExecutionEvent, AgentExecutionMetadata, AgentExecutionStatus
from src.services.agent.evidence_grader import EvidenceGradingError, EvidenceSufficiencyGrader
from src.services.agent.final_evidence import (
    FinalEvidenceSelector,
    merge_evidence,
    normalize_live_evidence,
    normalize_local_evidence,
    to_evidence_inputs,
)
from src.services.agent.guardrail import GuardrailEvaluationError, GuardrailEvaluator
from src.services.agent.live_documents import LiveDocumentProcessor
from src.services.agent.live_search import LiveArxivSearchResult, LiveResearchSearchService
from src.services.agent.live_selection import LivePaperSelector, LiveSelectionError
from src.services.agent.query_rewriter import QueryRewriteError, QueryRewriter
from src.services.agent.state import AgentState
from src.services.agent.types import AgentErrorCategory, TerminalReason
from src.services.embeddings.base import EmbeddingProvider
from src.services.evidence import EvidenceContextBuilder
from src.services.llm import LLMProvider
from src.services.rag.service import RAGGenerationService
from src.services.search.hybrid_service import HybridSearchService

GuardrailRoute = Literal["passed", "rejected", "failed"]
RetrievalRoute = Literal["retrieved", "failed"]
EvidenceRoute = Literal["sufficient", "rewrite", "live_fallback", "exhausted", "failed"]
LiveSearchRoute = Literal["candidates", "empty", "failed"]
LiveSelectionRoute = Literal["selected", "none", "failed"]
LiveDocumentRoute = Literal["evidence", "unusable", "failed"]
RerankRoute = Literal["selected", "empty", "failed"]
GenerationRoute = Literal["generated", "insufficient", "failed"]
RewriteRoute = Literal["rewritten", "failed"]


@dataclass(frozen=True)
class AgentGraphDependencies:
    """Worker-scoped services injected into graph nodes."""

    llm_provider: LLMProvider
    hybrid_search_service: HybridSearchService
    evidence_context_builder: EvidenceContextBuilder
    embedding_provider: EmbeddingProvider
    rag_generation_service: RAGGenerationService
    live_search_service: LiveResearchSearchService | None = None
    live_document_processor: LiveDocumentProcessor | None = None


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
        return "live_fallback" if config.live_fallback_enabled else "exhausted"

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


def _make_live_search_node(service: LiveResearchSearchService, config: AgentGraphConfig):
    async def live_arxiv_search(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        max_results = config.live_arxiv_max_results
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.LIVE_FALLBACK_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(live_fallback_used=True, max_results=max_results),
        )
        local_result = state["local_retrieval_result"]
        local_arxiv_ids = tuple(hit.arxiv_id for hit in local_result.results) if local_result else ()
        try:
            result = await service.search(
                state["current_query"],
                max_results=max_results,
                exclude_arxiv_ids=local_arxiv_ids,
            )
            if result.count > max_results:
                result = LiveArxivSearchResult(query=result.query, candidates=result.candidates[:max_results])
        except Exception:
            return {
                "live_fallback_used": True,
                "live_search_result": None,
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.LIVE_SEARCH_FAILURE,
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
            }
        return {
            "live_fallback_used": True,
            "live_search_result": result,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.LIVE_FALLBACK_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(
                        live_fallback_used=True,
                        max_results=max_results,
                        candidate_count=result.count,
                    ),
                ),
            ],
        }

    return live_arxiv_search


def _route_after_live_search(state: AgentState) -> LiveSearchRoute:
    if state["error_category"] is not None:
        return "failed"
    return "candidates" if state["live_search_result"].count else "empty"


def _make_live_selection_node(selector: LivePaperSelector, config: AgentGraphConfig):
    async def live_paper_selection(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        candidates = state["live_search_result"].candidates
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.LIVE_SELECTION_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(candidate_count=len(candidates)),
        )
        try:
            selection = await selector.select(
                state["original_question"],
                candidates,
                max_papers=config.live_pdf_max_papers,
            )
        except LiveSelectionError:
            return {
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.LIVE_SELECTION_FAILURE,
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
            }
        papers = selection.papers[: config.live_pdf_max_papers]
        return {
            "live_selected_papers": papers,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.LIVE_SELECTION_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(candidate_count=len(candidates), selected_count=len(papers)),
                ),
            ],
        }

    return live_paper_selection


def _route_after_live_selection(state: AgentState) -> LiveSelectionRoute:
    if state["error_category"] is not None:
        return "failed"
    return "selected" if state["live_selected_papers"] else "none"


def _make_live_document_node(processor: LiveDocumentProcessor, config: AgentGraphConfig):
    async def live_document_processing(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        papers = state["live_selected_papers"]
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.LIVE_DOCUMENT_PROCESSING_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(selected_count=len(papers)),
        )

        def failed() -> dict[str, Any]:
            return {
                "transient_live_evidence": (),
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.LIVE_DOCUMENT_PROCESSING_FAILURE,
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
            }

        try:
            result = await processor.process(
                papers,
                question=state["original_question"],
                max_chunks_per_paper=config.live_max_chunks_per_paper,
            )
        except Exception:
            return failed()
        # Enforce the bounds here too: only selected papers, and a capped number of chunks from each.
        kept_per_paper = {paper.arxiv_id: 0 for paper in papers}
        chunks = []
        for chunk in result.chunks:
            if kept_per_paper.get(chunk.arxiv_id, config.live_max_chunks_per_paper) < config.live_max_chunks_per_paper:
                kept_per_paper[chunk.arxiv_id] += 1
                chunks.append(chunk)
        if not chunks and result.failed_count >= len(papers):
            return failed()
        return {
            "transient_live_evidence": tuple(chunks),
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.LIVE_DOCUMENT_PROCESSING_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(
                        selected_count=len(papers),
                        processed_count=sum(1 for count in kept_per_paper.values() if count),
                        failed_count=result.failed_count,
                        transient_chunk_count=len(chunks),
                    ),
                ),
            ],
        }

    return live_document_processing


def _route_after_live_documents(state: AgentState) -> LiveDocumentRoute:
    if state["error_category"] is not None:
        return "failed"
    return "evidence" if state["transient_live_evidence"] else "unusable"


def _make_evidence_rerank_node(selector: FinalEvidenceSelector, config: AgentGraphConfig):
    async def evidence_rerank(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        # Locally insufficient evidence is still partial support, so it is merged with any live chunks.
        local = normalize_local_evidence(state["local_retrieval_result"])
        live = normalize_live_evidence(state["transient_live_evidence"])
        merged = merge_evidence(local, live)
        counts = {
            "local_candidate_count": len(local),
            "live_candidate_count": len(live),
            "merged_candidate_count": len(merged),
        }
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.EVIDENCE_RERANK_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(**counts),
        )
        try:
            final = await selector.select(
                state["original_question"],
                merged,
                max_sources=config.final_evidence_max_sources,
            )
        except Exception:
            return {
                "terminal_reason": TerminalReason.INTERNAL_ERROR,
                "error_category": AgentErrorCategory.EVIDENCE_RERANK_FAILURE,
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
            }
        return {
            "final_evidence": final,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.EVIDENCE_RERANK_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(**counts, final_source_count=len(final)),
                ),
            ],
        }

    return evidence_rerank


def _route_after_evidence_rerank(state: AgentState) -> RerankRoute:
    if state["error_category"] is not None:
        return "failed"
    return "selected" if state["final_evidence"] else "empty"


def _make_generation_node(service: RAGGenerationService, context_builder: EvidenceContextBuilder):
    async def answer_generation(state: AgentState) -> dict[str, Any]:
        sequence = _next_sequence(state)
        final = state["final_evidence"]
        local_result = state["local_retrieval_result"]
        started = AgentExecutionEvent(
            status=AgentExecutionStatus.GENERATION_STARTED,
            sequence=sequence,
            metadata=AgentExecutionMetadata(final_source_count=len(final)),
        )
        try:
            evidence = context_builder.build_from_sources(
                to_evidence_inputs(final),
                query=state["current_query"],
                retrieval_mode=local_result.retrieval_mode,
                degradation_reason=local_result.degradation_reason,
            )
            result = await service.generate_from_context(question=state["original_question"], evidence=evidence)
        except InsufficientEvidenceError:
            # Nothing fit the context budget: there is no evidence to answer from, which is not a failure.
            return {"execution_events": [started]}
        except Exception:
            return {
                "terminal_reason": TerminalReason.GENERATION_FAILED,
                "error_category": AgentErrorCategory.GENERATION_FAILURE,
                "execution_events": [
                    started,
                    AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_FAILED, sequence=sequence + 1),
                ],
            }
        return {
            "generation_result": result,
            "generated_answer": result.answer,
            "execution_events": [
                started,
                AgentExecutionEvent(
                    status=AgentExecutionStatus.GENERATION_COMPLETED,
                    sequence=sequence + 1,
                    metadata=AgentExecutionMetadata(
                        final_source_count=len(result.sources),
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    ),
                ),
            ],
        }

    return answer_generation


def _route_after_generation(state: AgentState) -> GenerationRoute:
    if state["error_category"] is not None:
        return "failed"
    return "generated" if state["generation_result"] is not None else "insufficient"


def _complete_insufficient_evidence(state: AgentState) -> dict[str, Any]:
    return {
        "terminal_reason": TerminalReason.INSUFFICIENT_EVIDENCE,
        "execution_events": [
            AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_COMPLETED, sequence=_next_sequence(state))
        ],
    }


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
    """Compile W7-M07 with injected services and no infrastructure creation."""

    graph_config = config or AgentGraphConfig()
    if graph_config.live_fallback_enabled and dependencies.live_search_service is None:
        raise ValueError("live_search_service is required when live_fallback_enabled is true")
    if graph_config.live_fallback_enabled and dependencies.live_document_processor is None:
        raise ValueError("live_document_processor is required when live_fallback_enabled is true")
    evaluator = GuardrailEvaluator(llm_provider=dependencies.llm_provider)
    grader = EvidenceSufficiencyGrader(llm_provider=dependencies.llm_provider)
    rewriter = QueryRewriter(llm_provider=dependencies.llm_provider)
    final_selector = FinalEvidenceSelector(embedding_provider=dependencies.embedding_provider)

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
    if graph_config.live_fallback_enabled:
        builder.add_node("live_arxiv_search", _make_live_search_node(dependencies.live_search_service, graph_config))
        builder.add_conditional_edges(
            "live_arxiv_search",
            _route_after_live_search,
            {"candidates": "live_paper_selection", "empty": "insufficient_evidence", "failed": END},
        )
        selector = LivePaperSelector(llm_provider=dependencies.llm_provider)
        builder.add_node("live_paper_selection", _make_live_selection_node(selector, graph_config))
        builder.add_conditional_edges(
            "live_paper_selection",
            _route_after_live_selection,
            {"selected": "live_document_processing", "none": "insufficient_evidence", "failed": END},
        )
        builder.add_node(
            "live_document_processing",
            _make_live_document_node(dependencies.live_document_processor, graph_config),
        )
        builder.add_conditional_edges(
            "live_document_processing",
            _route_after_live_documents,
            {"evidence": "evidence_rerank", "unusable": "insufficient_evidence", "failed": END},
        )
    builder.add_node("evidence_rerank", _make_evidence_rerank_node(final_selector, graph_config))
    builder.add_conditional_edges(
        "evidence_rerank",
        _route_after_evidence_rerank,
        {"selected": "answer_generation", "empty": "insufficient_evidence", "failed": END},
    )
    builder.add_node(
        "answer_generation",
        _make_generation_node(dependencies.rag_generation_service, dependencies.evidence_context_builder),
    )
    builder.add_conditional_edges(
        "answer_generation",
        _route_after_generation,
        {"generated": "graph_complete", "insufficient": "insufficient_evidence", "failed": END},
    )
    builder.add_node("insufficient_evidence", _complete_insufficient_evidence)
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
            "sufficient": "evidence_rerank",
            "rewrite": "query_rewrite",
            "live_fallback": "live_arxiv_search" if graph_config.live_fallback_enabled else "insufficient_evidence",
            "exhausted": "insufficient_evidence",
            "failed": END,
        },
    )
    builder.add_conditional_edges(
        "query_rewrite",
        _route_after_query_rewrite,
        {"rewritten": "local_retrieval", "failed": END},
    )
    builder.add_edge("insufficient_evidence", END)
    builder.add_edge("graph_complete", END)
    return builder.compile()
