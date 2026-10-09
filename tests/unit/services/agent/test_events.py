import pytest
from pydantic import ValidationError
from src.services.agent import (
    AgentErrorCategory,
    AgentExecutionEvent,
    AgentExecutionMetadata,
    AgentExecutionStatus,
    TerminalReason,
)


def test_execution_event_has_deterministic_representation() -> None:
    event = AgentExecutionEvent(
        status=AgentExecutionStatus.LOCAL_RETRIEVAL_COMPLETED,
        sequence=3,
        metadata=AgentExecutionMetadata(retrieval_attempt=1, source_count=5),
    )

    assert event.model_dump(mode="json") == {
        "status": "local_retrieval_completed",
        "sequence": 3,
        "metadata": {
            "retrieval_attempt": 1,
            "next_retrieval_attempt": None,
            "source_count": 5,
            "live_fallback_used": None,
            "max_results": None,
            "candidate_count": None,
            "selected_count": None,
            "processed_count": None,
            "failed_count": None,
            "transient_chunk_count": None,
            "local_candidate_count": None,
            "live_candidate_count": None,
            "merged_candidate_count": None,
            "final_source_count": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "guardrail_passed": None,
            "guardrail_score": None,
            "evidence_sufficient": None,
            "evidence_score": None,
            "grounding_passed": None,
        },
    }


def test_execution_event_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        AgentExecutionEvent(status="model_thought_about_evidence", sequence=0)  # type: ignore[arg-type]


def test_execution_metadata_rejects_non_allowlisted_content() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentExecutionMetadata(prompt="private prompt")  # type: ignore[call-arg]


def test_event_metadata_defaults_are_not_shared() -> None:
    first = AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_STARTED, sequence=0)
    second = AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_STARTED, sequence=0)

    assert first.metadata is not second.metadata


def test_terminal_reasons_distinguish_domain_outcomes_and_failures() -> None:
    assert TerminalReason.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
    assert TerminalReason.GENERATION_FAILED.value == "generation_failed"
    assert TerminalReason.INSUFFICIENT_EVIDENCE is not TerminalReason.GENERATION_FAILED
    assert AgentErrorCategory.GENERATION_FAILURE.value == "generation_failure"


@pytest.mark.parametrize("score", [-1, 101])
def test_evidence_score_metadata_is_bounded(score: int) -> None:
    assert AgentExecutionMetadata(evidence_score=0).evidence_score == 0
    assert AgentExecutionMetadata(evidence_score=100).evidence_score == 100
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(evidence_score=score)


def test_evidence_grading_failure_is_a_distinct_error_category() -> None:
    assert AgentErrorCategory.EVIDENCE_GRADING_FAILURE.value == "evidence_grading_failure"
    assert AgentErrorCategory.EVIDENCE_GRADING_FAILURE is not AgentErrorCategory.EVIDENCE_FAILURE


def test_next_retrieval_attempt_metadata_is_bounded() -> None:
    assert AgentExecutionMetadata(retrieval_attempt=1, next_retrieval_attempt=2).next_retrieval_attempt == 2
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(next_retrieval_attempt=1)


def test_query_rewrite_failure_is_a_distinct_error_category() -> None:
    assert AgentErrorCategory.QUERY_REWRITE_FAILURE.value == "query_rewrite_failure"
    assert AgentErrorCategory.QUERY_REWRITE_FAILURE is not AgentErrorCategory.EVIDENCE_GRADING_FAILURE


def test_live_fallback_metadata_is_bounded_and_count_only() -> None:
    metadata = AgentExecutionMetadata(live_fallback_used=True, max_results=5, candidate_count=0)

    assert metadata.candidate_count == 0
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(candidate_count=-1)
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(max_results=0)
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(title="Private Title")  # type: ignore[call-arg]


def test_live_search_failure_is_a_distinct_error_category() -> None:
    assert AgentErrorCategory.LIVE_SEARCH_FAILURE.value == "live_search_failure"
    assert AgentErrorCategory.LIVE_SEARCH_FAILURE is not AgentErrorCategory.RETRIEVAL_FAILURE


@pytest.mark.parametrize("field", ["selected_count", "processed_count", "failed_count", "transient_chunk_count"])
def test_live_document_metadata_is_count_only(field: str) -> None:
    assert getattr(AgentExecutionMetadata(**{field: 0}), field) == 0
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(**{field: -1})


def test_live_document_metadata_rejects_paper_content() -> None:
    for forbidden in ("arxiv_id", "pdf_url", "chunk_text"):
        with pytest.raises(ValidationError):
            AgentExecutionMetadata(**{forbidden: "private"})


def test_live_selection_and_processing_vocabulary_is_distinct() -> None:
    assert AgentExecutionStatus.LIVE_SELECTION_COMPLETED.value == "live_selection_completed"
    assert AgentExecutionStatus.LIVE_DOCUMENT_PROCESSING_COMPLETED.value == "live_document_processing_completed"
    assert AgentErrorCategory.LIVE_SELECTION_FAILURE.value == "live_selection_failure"
    assert AgentErrorCategory.LIVE_DOCUMENT_PROCESSING_FAILURE.value == "live_document_processing_failure"


@pytest.mark.parametrize(
    "field",
    [
        "local_candidate_count",
        "live_candidate_count",
        "merged_candidate_count",
        "final_source_count",
        "prompt_tokens",
        "completion_tokens",
    ],
)
def test_rerank_and_generation_metadata_is_count_only(field: str) -> None:
    assert getattr(AgentExecutionMetadata(**{field: 0}), field) == 0
    with pytest.raises(ValidationError):
        AgentExecutionMetadata(**{field: -1})


def test_rerank_and_generation_metadata_rejects_content_and_scores() -> None:
    for forbidden in ("answer", "model", "relevance_score", "embedding", "source_url"):
        with pytest.raises(ValidationError):
            AgentExecutionMetadata(**{forbidden: "private"})


def test_rerank_vocabulary_is_distinct_from_grading_and_generation() -> None:
    assert AgentExecutionStatus.EVIDENCE_RERANK_STARTED.value == "evidence_rerank_started"
    assert AgentExecutionStatus.EVIDENCE_RERANK_COMPLETED.value == "evidence_rerank_completed"
    assert AgentErrorCategory.EVIDENCE_RERANK_FAILURE.value == "evidence_rerank_failure"
    assert AgentErrorCategory.EVIDENCE_RERANK_FAILURE is not AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert AgentErrorCategory.EVIDENCE_RERANK_FAILURE is not AgentErrorCategory.GENERATION_FAILURE
