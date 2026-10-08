import pytest
from pydantic import ValidationError
from src.schemas.discovery import ArxivDiscoveryProfile
from src.services.arxiv.discovery_profiles import DISCOVERY_PROFILES, get_discovery_profiles


def test_expected_profiles_exist_with_distinct_intent() -> None:
    assert set(DISCOVERY_PROFILES) == {"ai", "healthcare_ai", "health_equity_tech"}
    assert DISCOVERY_PROFILES["ai"].category == "cs.AI"

    healthcare_query = DISCOVERY_PROFILES["healthcare_ai"].search_query.lower()
    assert "clinical" in healthcare_query
    assert "machine learning" in healthcare_query

    equity_query = DISCOVERY_PROFILES["health_equity_tech"].search_query.lower()
    assert "health equity" in equity_query
    assert "machine learning" in equity_query


def test_profile_requires_exactly_one_query_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ArxivDiscoveryProfile(name="bad", description="ambiguous", category="cs.AI", search_query="all:AI")


def test_enabled_profile_configuration_preserves_order_and_deduplicates() -> None:
    profiles = get_discovery_profiles("healthcare_ai, ai,healthcare_ai")
    assert [profile.name for profile in profiles] == ["healthcare_ai", "ai"]


def test_unknown_profile_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown arXiv discovery profile 'unknown'"):
        get_discovery_profiles("ai,unknown")
