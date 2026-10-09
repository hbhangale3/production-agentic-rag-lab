"""Constrained terminal and failure vocabularies for agent execution."""

from enum import StrEnum


class TerminalReason(StrEnum):
    COMPLETED = "completed"
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GROUNDING_FAILED = "grounding_failed"
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"
    INTERNAL_ERROR = "internal_error"


class AgentErrorCategory(StrEnum):
    GUARDRAIL_FAILURE = "guardrail_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    EVIDENCE_FAILURE = "evidence_failure"
    EVIDENCE_GRADING_FAILURE = "evidence_grading_failure"
    QUERY_REWRITE_FAILURE = "query_rewrite_failure"
    LIVE_SEARCH_FAILURE = "live_search_failure"
    LIVE_SELECTION_FAILURE = "live_selection_failure"
    LIVE_DOCUMENT_PROCESSING_FAILURE = "live_document_processing_failure"
    GENERATION_FAILURE = "generation_failure"
    GROUNDING_FAILURE = "grounding_failure"
    INTERNAL_FAILURE = "internal_failure"
    CANCELLED = "cancelled"
