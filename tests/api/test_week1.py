import pytest


@pytest.mark.anyio
async def test_ping(client):
    """Ping endpoint should confirm that the API is reachable."""
    response = await client.get("/api/v1/ping")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "pong",
    }


@pytest.mark.anyio
async def test_health(client):
    """Health endpoint should confirm API and database availability."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["services"]["database"]["status"] == "healthy"


@pytest.mark.anyio
async def test_ask_returns_week1_mock_response(client):
    """Week 1 ask endpoint should return the expected mock contract."""
    response = await client.post(
        "/api/v1/ask",
        json={"question": "What is retrieval augmented generation?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert "mock response for week 1" in body["answer"].lower()
    assert len(body["sources"]) == 2

    assert body["sources"][0]["arxiv_id"] == "2401.00001"
    assert body["sources"][1]["arxiv_id"] == "2401.00002"
