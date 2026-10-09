import pytest
from src.services.agent import EvidenceGrade, create_initial_agent_state


def test_initial_state_preserves_question_and_initializes_future_fields() -> None:
    question = "  How can AI improve healthcare access?  "

    state = create_initial_agent_state(question)

    assert state["original_question"] == question
    assert state["current_query"] == question
    assert state["retrieval_attempts"] == 0
    assert state["grounding_attempts"] == 0
    assert state["live_fallback_used"] is False
    assert state["guardrail_score"] is None
    assert state["guardrail_passed"] is None
    assert state["local_retrieval_result"] is None
    assert state["rewritten_query"] is None
    assert state["evidence_sufficient"] is None
    assert state["evidence_grade"] is None
    assert state["live_evidence"] is None
    assert state["final_evidence"] is None
    assert state["generated_answer"] is None
    assert state["grounding_passed"] is None
    assert state["terminal_reason"] is None
    assert state["error_category"] is None
    assert state["execution_events"] == []


def test_initial_states_have_independent_event_collections() -> None:
    first = create_initial_agent_state("First question")
    second = create_initial_agent_state("Second question")

    first["execution_events"].append(object())  # type: ignore[arg-type]

    assert second["execution_events"] == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_initial_state_rejects_blank_question(question: str) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        create_initial_agent_state(question)


def test_evidence_grade_is_content_free_and_validated() -> None:
    grade = EvidenceGrade(score=75, sufficient=True, source_count=3, grader_version="v1")

    assert grade.score == 75
    assert grade.sufficient is True
    assert grade.source_count == 3
    with pytest.raises(ValueError, match="between 0 and 100"):
        EvidenceGrade(score=101, sufficient=True, source_count=3)
    with pytest.raises(ValueError, match="must be a boolean"):
        EvidenceGrade(score=75, sufficient=1, source_count=3)  # type: ignore[arg-type]
