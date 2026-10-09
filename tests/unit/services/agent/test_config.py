import pytest
from pydantic import ValidationError
from src.services.agent import AgentGraphConfig


def test_graph_config_defaults_are_bounded() -> None:
    config = AgentGraphConfig()

    assert config.guardrail_threshold == 60
    assert config.evidence_sufficiency_threshold == 60
    assert config.max_local_retrieval_attempts == 2
    assert config.retrieval_size == 5
    assert config.max_grounding_attempts == 2
    assert config.live_fallback_enabled is True
    assert config.live_arxiv_max_results == 5
    assert config.live_pdf_max_papers == 2
    assert config.live_max_chunks_per_paper == 8


@pytest.mark.parametrize("threshold", [0, 100])
def test_guardrail_threshold_boundaries_are_valid(threshold: int) -> None:
    assert AgentGraphConfig(guardrail_threshold=threshold).guardrail_threshold == threshold


@pytest.mark.parametrize("threshold", [-1, 101])
def test_invalid_guardrail_threshold_is_rejected(threshold: int) -> None:
    with pytest.raises(ValidationError):
        AgentGraphConfig(guardrail_threshold=threshold)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_local_retrieval_attempts": 0},
        {"retrieval_size": 0},
        {"max_grounding_attempts": 0},
    ],
)
def test_positive_execution_limits_are_required(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        AgentGraphConfig(**overrides)


def test_two_local_attempts_mean_initial_plus_one_retry() -> None:
    config = AgentGraphConfig(max_local_retrieval_attempts=2)

    initial_attempts = 1
    permitted_retries = config.max_local_retrieval_attempts - initial_attempts

    assert permitted_retries == 1


def test_live_fallback_can_be_disabled() -> None:
    assert AgentGraphConfig(live_fallback_enabled=False).live_fallback_enabled is False


@pytest.mark.parametrize("threshold", [0, 100])
def test_evidence_sufficiency_threshold_boundaries_are_valid(threshold: int) -> None:
    assert AgentGraphConfig(evidence_sufficiency_threshold=threshold).evidence_sufficiency_threshold == threshold


@pytest.mark.parametrize("threshold", [-1, 101])
def test_invalid_evidence_sufficiency_threshold_is_rejected(threshold: int) -> None:
    with pytest.raises(ValidationError):
        AgentGraphConfig(evidence_sufficiency_threshold=threshold)


@pytest.mark.parametrize("limit", [1, 10])
def test_live_arxiv_max_results_boundaries_are_valid(limit: int) -> None:
    config = AgentGraphConfig(live_arxiv_max_results=limit, live_pdf_max_papers=1)

    assert config.live_arxiv_max_results == limit


@pytest.mark.parametrize("limit", [0, -1, 11, 100])
def test_live_arxiv_max_results_must_stay_small(limit: int) -> None:
    with pytest.raises(ValidationError):
        AgentGraphConfig(live_arxiv_max_results=limit)


def test_live_pdf_max_papers_cannot_exceed_live_candidates() -> None:
    assert AgentGraphConfig(live_arxiv_max_results=3, live_pdf_max_papers=3).live_pdf_max_papers == 3
    assert AgentGraphConfig(live_arxiv_max_results=1, live_pdf_max_papers=1).live_pdf_max_papers == 1
    with pytest.raises(ValidationError, match="must not exceed live_arxiv_max_results"):
        AgentGraphConfig(live_arxiv_max_results=3, live_pdf_max_papers=4)
    with pytest.raises(ValidationError):
        AgentGraphConfig(live_arxiv_max_results=1)
    with pytest.raises(ValidationError):
        AgentGraphConfig(live_pdf_max_papers=0)


@pytest.mark.parametrize("limit", [0, 51])
def test_live_max_chunks_per_paper_is_bounded(limit: int) -> None:
    with pytest.raises(ValidationError):
        AgentGraphConfig(live_max_chunks_per_paper=limit)
