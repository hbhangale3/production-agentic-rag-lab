from unittest.mock import Mock

import pytest
from src.dependencies import get_hybrid_search_service
from src.exceptions import HybridSearchError
from src.main import app
from src.schemas.hybrid_search import HybridSearchResult


@pytest.fixture
def hybrid_service():
    service = Mock()
    service.search.return_value = HybridSearchResult(
        query="clinical", retrieval_mode="hybrid", count=0, results=[]
    )
    app.dependency_overrides[get_hybrid_search_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_hybrid_search_service, None)


@pytest.mark.anyio
async def test_endpoint_forwards_request_and_returns_mode(client, hybrid_service) -> None:
    response = await client.post(
        "/api/v1/hybrid-search",
        json={"query": "clinical", "size": 5, "categories": ["cs.AI"]},
    )
    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "hybrid"
    hybrid_service.search.assert_called_once()


@pytest.mark.anyio
async def test_endpoint_validation_and_complete_failure(client, hybrid_service) -> None:
    assert (await client.post("/api/v1/hybrid-search", json={"query": "   "})).status_code == 422
    hybrid_service.search.side_effect = HybridSearchError("both down")
    response = await client.post("/api/v1/hybrid-search", json={"query": "valid"})
    assert response.status_code == 503
