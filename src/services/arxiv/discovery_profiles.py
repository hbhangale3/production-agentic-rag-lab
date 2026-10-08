from collections.abc import Iterable

from src.schemas.discovery import ArxivDiscoveryProfile

TECHNOLOGY_TERMS = (
    'all:"artificial intelligence" OR all:"machine learning" OR '
    'all:"natural language processing" OR all:"large language model" OR all:"digital health"'
)

DISCOVERY_PROFILES: dict[str, ArxivDiscoveryProfile] = {
    "ai": ArxivDiscoveryProfile(
        name="ai",
        description="General artificial-intelligence research",
        category="cs.AI",
    ),
    "healthcare_ai": ArxivDiscoveryProfile(
        name="healthcare_ai",
        description="AI and computational technology applied to healthcare",
        search_query=(
            '(all:healthcare OR all:clinical OR all:medical OR all:patient OR all:"electronic health record") '
            f"AND ({TECHNOLOGY_TERMS})"
        ),
    ),
    "health_equity_tech": ArxivDiscoveryProfile(
        name="health_equity_tech",
        description="Health equity, disparities, and access research involving technology",
        search_query=(
            '(all:"health equity" OR all:"health disparities" OR all:underserved OR all:"healthcare access") '
            f"AND ({TECHNOLOGY_TERMS})"
        ),
    ),
}


def get_discovery_profiles(names: str | Iterable[str]) -> list[ArxivDiscoveryProfile]:
    """Resolve enabled profiles in configured order, rejecting unknown names."""
    requested = names.split(",") if isinstance(names, str) else names
    selected: list[ArxivDiscoveryProfile] = []
    seen: set[str] = set()
    for raw_name in requested:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        try:
            profile = DISCOVERY_PROFILES[name]
        except KeyError as exc:
            available = ", ".join(DISCOVERY_PROFILES)
            raise ValueError(f"Unknown arXiv discovery profile '{name}'. Available profiles: {available}") from exc
        if profile.enabled:
            selected.append(profile)
            seen.add(name)
    return selected
