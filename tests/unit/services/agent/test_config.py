import pytest
from pydantic import ValidationError
from src.services.agent import AgentGraphConfig


def test_graph_config_defaults_are_bounded() -> None:
    config = AgentGraphConfig()

    assert config.guardrail_threshold == 60
    assert config.max_local_retrieval_attempts == 2
    assert config.retrieval_size == 5
    assert config.max_grounding_attempts == 2
    assert config.live_fallback_enabled is True


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
