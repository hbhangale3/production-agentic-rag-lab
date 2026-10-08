from datetime import date
from unittest.mock import Mock

import pytest
from src.dependencies import get_paper_search_service
from src.main import app
from src.schemas.search import PaperSearchHit, PaperSearchResult

SEARCH_PATHS = {
    "/api/v1/search/simple",
    "/api/v1/search/match",
    "/api/v1/search/multi-match",
    "/api/v1/search/boosting",
    "/api/v1/search/filter",
    "/api/v1/search/sort",
    "/api/v1/search/combined",
}


def successful_result(query: str = "clinical") -> PaperSearchResult:
    return PaperSearchResult(
        query=query,
        total=1,
        took_ms=3,
        hits=[
            PaperSearchHit(
                arxiv_id="2610.01963",
                title="Clinical Triage",
                authors=["Author"],
                abstract="Abstract",
                categories=["cs.AI"],
                score=5.4,
            )
        ],
    )


@pytest.fixture
def search_service() -> Mock:
    service = Mock()
    for method in (
        "search_simple",
        "search_match",
        "search_multi_match",
        "search_boosting",
        "search_filtered",
        "search_sorted",
        "search_combined",
    ):
        getattr(service, method).return_value = successful_result()
    return service


@pytest.fixture(autouse=True)
def override_search_dependency(search_service: Mock):
    app.dependency_overrides[get_paper_search_service] = lambda: search_service
    yield
    app.dependency_overrides.pop(get_paper_search_service, None)


@pytest.mark.anyio
async def test_simple_route_forwards_query_and_size(client, search_service: Mock) -> None:
    response = await client.get("/api/v1/search/simple", params={"q": "clinical triage", "size": 4})

    assert response.status_code == 200
    search_service.search_simple.assert_called_once_with("clinical triage", size=4)
    assert response.json()["hits"][0]["arxiv_id"] == "2610.01963"
    assert "raw_text" not in response.json()["hits"][0]


@pytest.mark.anyio
async def test_match_route_accepts_safe_field_and_rejects_unknown(client, search_service: Mock) -> None:
    accepted = await client.get("/api/v1/search/match", params={"q": "clinical", "field": "abstract"})
    rejected = await client.get("/api/v1/search/match", params={"q": "clinical", "field": "secret"})

    assert accepted.status_code == 200
    search_service.search_match.assert_called_once_with("clinical", field="abstract", size=10)
    assert rejected.status_code == 422


@pytest.mark.anyio
async def test_multi_match_parses_repeated_fields_and_defaults(client, search_service: Mock) -> None:
    repeated = await client.get(
        "/api/v1/search/multi-match",
        params=[("q", "counterfactual"), ("fields", "title"), ("fields", "abstract")],
    )
    defaulted = await client.get("/api/v1/search/multi-match", params={"q": "counterfactual"})

    assert repeated.status_code == defaulted.status_code == 200
    assert search_service.search_multi_match.call_args_list[0].kwargs == {
        "fields": ["title", "abstract"],
        "size": 10,
    }
    assert search_service.search_multi_match.call_args_list[1].kwargs == {"fields": None, "size": 10}


@pytest.mark.anyio
async def test_boosting_forwards_values_and_rejects_invalid_boost(client, search_service: Mock) -> None:
    accepted = await client.get(
        "/api/v1/search/boosting",
        params={"positive": "language models", "negative": "survey", "negative_boost": 0.3},
    )
    rejected = await client.get(
        "/api/v1/search/boosting",
        params={"positive": "language models", "negative": "survey", "negative_boost": 2},
    )

    assert accepted.status_code == 200
    search_service.search_boosting.assert_called_once_with(
        "language models", "survey", negative_boost=0.3, size=10
    )
    assert rejected.status_code == 422


@pytest.mark.anyio
async def test_filter_parses_categories_and_dates(client, search_service: Mock) -> None:
    response = await client.get(
        "/api/v1/search/filter",
        params=[
            ("q", "clinical"),
            ("category", "cs.AI"),
            ("category", "cs.CL"),
            ("published_from", "2026-01-01"),
            ("published_to", "2026-12-31"),
        ],
    )

    assert response.status_code == 200
    search_service.search_filtered.assert_called_once_with(
        "clinical",
        categories=["cs.AI", "cs.CL"],
        published_from=date(2026, 1, 1),
        published_to=date(2026, 12, 31),
        size=10,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("sort_by", ["relevance", "published_date"])
async def test_sort_accepts_safe_options(client, search_service: Mock, sort_by: str) -> None:
    response = await client.get(
        "/api/v1/search/sort",
        params={"q": "machine learning", "sort_by": sort_by, "sort_order": "asc"},
    )

    assert response.status_code == 200
    search_service.search_sorted.assert_called_once_with(
        "machine learning", sort_by=sort_by, sort_order="asc", size=10
    )


@pytest.mark.anyio
async def test_sort_rejects_unknown_option(client) -> None:
    response = await client.get("/api/v1/search/sort", params={"q": "ai", "sort_by": "title"})

    assert response.status_code == 422


@pytest.mark.anyio
async def test_combined_forwards_all_supported_parameters(client, search_service: Mock) -> None:
    response = await client.get(
        "/api/v1/search/combined",
        params={
            "q": "healthcare",
            "category": "cs.AI",
            "published_from": "2026-01-01",
            "preferred_category": "cs.CL",
            "sort_by": "published_date",
            "sort_order": "desc",
            "size": 5,
        },
    )

    assert response.status_code == 200
    search_service.search_combined.assert_called_once_with(
        "healthcare",
        categories=["cs.AI"],
        published_from=date(2026, 1, 1),
        published_to=None,
        preferred_category="cs.CL",
        sort_by="published_date",
        sort_order="desc",
        size=5,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("params", [{"q": "clinical", "size": 0}, {"q": "   ", "size": 10}])
async def test_common_input_validation(client, params: dict) -> None:
    response = await client.get("/api/v1/search/simple", params=params)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_backend_failure_maps_to_service_unavailable(client, search_service: Mock) -> None:
    search_service.search_simple.return_value = PaperSearchResult(
        query="clinical", successful=False, error="OpenSearch search failed"
    )

    response = await client.get("/api/v1/search/simple", params={"q": "clinical"})

    assert response.status_code == 503
    assert response.json()["detail"] == "OpenSearch search failed"


@pytest.mark.anyio
async def test_openapi_contains_all_search_routes(client) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert SEARCH_PATHS <= set(response.json()["paths"])
