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
