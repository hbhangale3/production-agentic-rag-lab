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
            "source_count": 5,
            "live_fallback_used": None,
            "guardrail_passed": None,
            "guardrail_score": None,
            "evidence_sufficient": None,
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
