"""Safe, provider-neutral execution events for the future agent graph."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AgentExecutionStatus(StrEnum):
    """Factual workflow transitions that are safe to expose externally."""

    GRAPH_STARTED = "graph_started"
    GUARDRAIL_STARTED = "guardrail_started"
    GUARDRAIL_PASSED = "guardrail_passed"
    GUARDRAIL_REJECTED = "guardrail_rejected"
    LOCAL_RETRIEVAL_STARTED = "local_retrieval_started"
    LOCAL_RETRIEVAL_COMPLETED = "local_retrieval_completed"
    EVIDENCE_GRADING_STARTED = "evidence_grading_started"
    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    QUERY_REWRITE_STARTED = "query_rewrite_started"
    QUERY_REWRITTEN = "query_rewritten"
    LOCAL_RETRY_STARTED = "local_retry_started"
    LIVE_FALLBACK_STARTED = "live_fallback_started"
    LIVE_FALLBACK_COMPLETED = "live_fallback_completed"
    LIVE_SELECTION_STARTED = "live_selection_started"
    LIVE_SELECTION_COMPLETED = "live_selection_completed"
    LIVE_DOCUMENT_PROCESSING_STARTED = "live_document_processing_started"
    LIVE_DOCUMENT_PROCESSING_COMPLETED = "live_document_processing_completed"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GROUNDING_STARTED = "grounding_started"
    GROUNDING_PASSED = "grounding_passed"
    GROUNDING_FAILED = "grounding_failed"
    GRAPH_COMPLETED = "graph_completed"
    GRAPH_FAILED = "graph_failed"


class AgentExecutionMetadata(BaseModel):
    """Allowlisted operational metadata; never reasoning or RAG content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval_attempt: int | None = Field(default=None, ge=1)
    next_retrieval_attempt: int | None = Field(default=None, ge=2)
    source_count: int | None = Field(default=None, ge=0)
    live_fallback_used: bool | None = None
    max_results: int | None = Field(default=None, ge=1)
    candidate_count: int | None = Field(default=None, ge=0)
    selected_count: int | None = Field(default=None, ge=0)
    processed_count: int | None = Field(default=None, ge=0)
    failed_count: int | None = Field(default=None, ge=0)
    transient_chunk_count: int | None = Field(default=None, ge=0)
    guardrail_passed: bool | None = None
    guardrail_score: int | None = Field(default=None, ge=0, le=100)
    evidence_sufficient: bool | None = None
    evidence_score: int | None = Field(default=None, ge=0, le=100)
    grounding_passed: bool | None = None


class AgentExecutionEvent(BaseModel):
    """One deterministic, content-free graph transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AgentExecutionStatus
    sequence: int = Field(ge=0)
    metadata: AgentExecutionMetadata = Field(default_factory=AgentExecutionMetadata)
