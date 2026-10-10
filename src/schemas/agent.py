from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.services.evidence import EvidenceSource

AgentOutcome = Literal["answered", "out_of_scope", "insufficient_evidence", "grounding_failed", "generation_failed"]
AgentCacheStatus = Literal["hit", "miss", "bypass", "failure"]


class AgentSource(BaseModel):
    """One source the final answer was generated from and grounded against."""

    citation: str = Field(pattern=r"^\[S[1-9]\d*\]$")
    source_type: Literal["local", "live_arxiv"]
    arxiv_id: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    title: str
    section: str | None = None
    content: str
    truncated: bool
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_date: datetime | None = None
    source_url: str | None = None


class AgentStatus(BaseModel):
    """One safe operational step; never reasoning, content, or error detail."""

    code: str
    message: str


class AgentExecutionSummary(BaseModel):
    """Content-free counts describing the run that produced an answer."""

    retrieval_attempts: int = Field(default=0, ge=0)
    query_rewritten: bool = False
    live_fallback_used: bool = False
    live_papers_selected: int = Field(default=0, ge=0)
    generation_attempts: int = Field(default=0, ge=0)
    grounding_attempts: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)


class AgentAskResponse(BaseModel):
    """Public agent result. ``answer`` is model-written only when ``outcome`` is ``answered``."""

    answer: str
    outcome: AgentOutcome
    sources: list[AgentSource] = Field(default_factory=list)
    cache_status: AgentCacheStatus
    terminal_reason: str | None = None
    grounding_passed: bool | None = None
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    execution: AgentExecutionSummary = Field(default_factory=AgentExecutionSummary)
    statuses: list[AgentStatus] = Field(default_factory=list)
    pipeline_version: str
    latency_ms: float | None = Field(default=None, ge=0)


def build_agent_source(source: "EvidenceSource") -> AgentSource:
    """Map application-owned evidence to its public representation, keeping provenance."""
    return AgentSource(
        citation=source.label,
        source_type=source.source_type,
        arxiv_id=source.arxiv_id,
        chunk_id=source.chunk_id,
        chunk_index=source.chunk_index,
        title=source.paper_title,
        section=source.section_title,
        content=source.content,
        truncated=source.truncated,
        authors=list(source.authors),
        categories=list(source.categories),
        published_date=source.published_date,
        source_url=source.source_url,
    )
