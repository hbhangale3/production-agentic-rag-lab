"""Validated behavior configuration for bounded agent graph execution."""

from pydantic import BaseModel, ConfigDict, Field


class AgentGraphConfig(BaseModel):
    """Agent-domain controls, intentionally free of infrastructure clients.

    ``max_local_retrieval_attempts`` counts total attempts. Its default of two
    means one initial local retrieval plus at most one rewritten-query retry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    guardrail_threshold: int = Field(default=60, ge=0, le=100)
    evidence_sufficiency_threshold: int = Field(default=60, ge=0, le=100)
    max_local_retrieval_attempts: int = Field(default=2, ge=1)
    retrieval_size: int = Field(default=5, ge=1)
    max_grounding_attempts: int = Field(default=2, ge=1)
    live_fallback_enabled: bool = True
    live_arxiv_max_results: int = Field(default=5, ge=1, le=10)
