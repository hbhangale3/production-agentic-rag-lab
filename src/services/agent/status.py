"""Deterministic user-facing text for safe execution events and terminal outcomes.

These are operational states only. Nothing here carries a question, query,
evidence, prompt, answer, score rationale, or error detail.
"""

from src.schemas.agent import AgentStatus
from src.services.agent.events import AgentExecutionEvent, AgentExecutionStatus
from src.services.agent.types import TerminalReason

S = AgentExecutionStatus

OUT_OF_SCOPE_MESSAGE = (
    "This research assistant answers questions about AI, machine learning, NLP, and healthcare AI research. "
    "This question is outside that scope."
)
INSUFFICIENT_EVIDENCE_MESSAGE = "I could not find enough reliable evidence to answer this question."
GROUNDING_FAILED_MESSAGE = (
    "I found relevant evidence, but could not produce an answer that passed grounding validation."
)
# The model responded, but its answer was structurally invalid or incomplete. No model text is shown.
ANSWER_NOT_VALIDATED_MESSAGE = "I could not produce a validated answer from the available evidence."
SEMANTIC_OUTCOME_MESSAGES: dict[TerminalReason, str] = {
    TerminalReason.OUT_OF_SCOPE: OUT_OF_SCOPE_MESSAGE,
    TerminalReason.INSUFFICIENT_EVIDENCE: INSUFFICIENT_EVIDENCE_MESSAGE,
    TerminalReason.GROUNDING_FAILED: GROUNDING_FAILED_MESSAGE,
}

CACHE_HIT_STATUS = AgentStatus(code="cache_hit", message="Validated cached answer found")

_FIXED_MESSAGES: dict[AgentExecutionStatus, str] = {
    S.GRAPH_STARTED: "Request accepted",
    S.GUARDRAIL_STARTED: "Checking question scope",
    S.GUARDRAIL_PASSED: "Guardrail passed",
    S.GUARDRAIL_REJECTED: "Question is outside the supported research scope",
    S.EVIDENCE_GRADING_STARTED: "Grading evidence sufficiency",
    S.EVIDENCE_SUFFICIENT: "Evidence sufficient",
    S.EVIDENCE_INSUFFICIENT: "Evidence insufficient",
    S.QUERY_REWRITE_STARTED: "Rewriting the retrieval query",
    S.QUERY_REWRITTEN: "Query rewritten",
    S.LOCAL_RETRY_STARTED: "Retrying local retrieval",
    S.LIVE_FALLBACK_STARTED: "Live arXiv fallback used",
    S.LIVE_SELECTION_STARTED: "Selecting live papers",
    S.LIVE_DOCUMENT_PROCESSING_STARTED: "Processing selected live papers",
    S.EVIDENCE_RERANK_STARTED: "Reranking evidence",
    S.GENERATION_LENGTH_RECOVERY_STARTED: "Initial answer was incomplete; regenerating with a larger output limit",
    S.GROUNDING_STARTED: "Validating answer grounding",
    S.GROUNDING_PASSED: "Grounding validation passed",
    S.GROUNDING_FAILED: "Grounding validation failed",
    S.GRAPH_COMPLETED: "Completed",
    S.GRAPH_FAILED: "The request could not be completed",
}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def status_message(event: AgentExecutionEvent) -> str:
    """Return safe text for one event, using only its allowlisted counts."""

    metadata = event.metadata
    retry = (metadata.retrieval_attempt or 1) > 1
    regeneration = (metadata.generation_attempt or 1) > 1
    match event.status:
        case S.LOCAL_RETRIEVAL_STARTED:
            return "Retrying local retrieval" if retry else "Searching the local corpus"
        case S.LOCAL_RETRIEVAL_COMPLETED:
            return "Local retry completed" if retry else "Local retrieval completed"
        case S.LIVE_FALLBACK_COMPLETED:
            return f"Live arXiv search found {_plural(metadata.candidate_count or 0, 'candidate paper')}"
        case S.LIVE_SELECTION_COMPLETED:
            return f"{_plural(metadata.selected_count or 0, 'live paper')} selected"
        case S.LIVE_DOCUMENT_PROCESSING_COMPLETED:
            return f"Live document processing completed ({_plural(metadata.processed_count or 0, 'paper')} usable)"
        case S.EVIDENCE_RERANK_COMPLETED:
            return f"Evidence reranked ({_plural(metadata.final_source_count or 0, 'source')} selected)"
        case S.GENERATION_STARTED:
            return "Regenerating the answer" if regeneration else "Generating the answer"
        case S.GENERATION_COMPLETED:
            return "Answer regenerated" if regeneration else "Answer generated"
    return _FIXED_MESSAGES.get(event.status, "Working")


def to_public_status(event: AgentExecutionEvent) -> AgentStatus:
    return AgentStatus(code=event.status.value, message=status_message(event))
