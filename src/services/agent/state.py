"""Typed state exchanged by current and future LangGraph nodes."""

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from src.schemas.hybrid_search import HybridSearchResult
from src.services.agent.answer_grounding import AnswerGroundingResult
from src.services.agent.events import AgentExecutionEvent
from src.services.agent.final_evidence import AgentEvidenceCandidate
from src.services.agent.live_documents import TransientLiveEvidenceChunk
from src.services.agent.live_search import LiveArxivPaper, LiveArxivSearchResult
from src.services.agent.prompt_identity import AgentPromptIdentityBundle
from src.services.agent.types import AgentErrorCategory, TerminalReason
from src.services.evidence import EvidenceSource
from src.services.rag.service import RAGGenerationResult


@dataclass(frozen=True)
class EvidenceGrade:
    """Content-free evidence sufficiency result; never reasoning or evidence text."""

    score: int
    sufficient: bool
    source_count: int
    grader_version: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, int) or not 0 <= self.score <= 100:
            raise ValueError("evidence grade score must be between 0 and 100")
        if not isinstance(self.sufficient, bool):
            raise ValueError("evidence grade sufficient must be a boolean")
        if isinstance(self.source_count, bool) or not isinstance(self.source_count, int) or self.source_count < 0:
            raise ValueError("evidence grade source_count must be non-negative")


class AgentState(TypedDict):
    """Provider-neutral graph data; service clients belong outside state."""

    original_question: str
    current_query: str
    guardrail_score: int | None
    guardrail_passed: bool | None
    retrieval_attempts: int
    local_retrieval_result: HybridSearchResult | None
    rewritten_query: str | None
    evidence_sufficient: bool | None
    evidence_grade: EvidenceGrade | None
    live_fallback_used: bool
    live_search_result: LiveArxivSearchResult | None
    live_selected_papers: tuple[LiveArxivPaper, ...] | None
    transient_live_evidence: tuple[TransientLiveEvidenceChunk, ...] | None
    live_evidence: tuple[EvidenceSource, ...] | None
    final_evidence: tuple[AgentEvidenceCandidate, ...] | None
    generation_result: RAGGenerationResult | None
    generated_answer: str | None
    grounding_passed: bool | None
    grounding_result: AnswerGroundingResult | None
    grounding_attempts: int
    prompt_identities: AgentPromptIdentityBundle | None
    execution_events: Annotated[list[AgentExecutionEvent], operator.add]
    terminal_reason: TerminalReason | None
    error_category: AgentErrorCategory | None


def create_initial_agent_state(question: str) -> AgentState:
    """Create validated state without performing I/O or mutating caller data."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be blank")
    return AgentState(
        original_question=question,
        current_query=question,
        guardrail_score=None,
        guardrail_passed=None,
        retrieval_attempts=0,
        local_retrieval_result=None,
        rewritten_query=None,
        evidence_sufficient=None,
        evidence_grade=None,
        live_fallback_used=False,
        live_search_result=None,
        live_selected_papers=None,
        transient_live_evidence=None,
        live_evidence=None,
        final_evidence=None,
        generation_result=None,
        generated_answer=None,
        grounding_passed=None,
        grounding_result=None,
        grounding_attempts=0,
        prompt_identities=None,
        execution_events=[],
        terminal_reason=None,
        error_category=None,
    )
