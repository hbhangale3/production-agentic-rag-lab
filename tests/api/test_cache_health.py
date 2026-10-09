import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from src.config import Settings
from src.dependencies import get_rag_generation_service, get_rag_response_cache
from src.main import app
from src.services.cache import CacheHealth, RAGResponseCacheCoordinator

from tests.api.test_ask_cache import cache_setup, generated_result
from tests.api.test_ask_stream_cache import stream_cache_setup


class HealthCache:
    def __init__(self, status: CacheHealth) -> None:
        self.status = status
        self.health_calls = 0

    async def health(self) -> CacheHealth:
        self.health_calls += 1
        return self.status

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        return False


def coordinator(cache, *, enabled: bool = True, timeout: float = 1.0):
    return RAGResponseCacheCoordinator(
        cache=cache,
        settings=Settings(
            redis_enabled=enabled,
            redis_health_timeout_seconds=timeout,
        ),
    )


@pytest.fixture
def override_cache():
    original = app.dependency_overrides.get(get_rag_response_cache)

    def apply(value):
        app.dependency_overrides[get_rag_response_cache] = lambda: value

    yield apply
    if original is None:
        app.dependency_overrides.pop(get_rag_response_cache, None)
    else:
        app.dependency_overrides[get_rag_response_cache] = original


@pytest.mark.anyio
async def test_health_reports_disabled_without_backend_ping(client, override_cache) -> None:
    cache = HealthCache(CacheHealth.HEALTHY)
    override_cache(coordinator(cache, enabled=False))

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["cache"] == {
        "enabled": False,
        "status": "disabled",
        "backend": "disabled",
        "ttl_seconds": 86400,
        "schema_version": "v1",
        "hits": 0,
        "misses": 0,
        "bypasses": 0,
        "writes": 0,
        "read_failures": 0,
        "write_failures": 0,
        "invalid_entries": 0,
        "hit_rate": 0.0,
    }
    assert cache.health_calls == 0

    disabled = app.dependency_overrides[get_rag_response_cache]()
    await disabled.lookup(question="Question")
    assert disabled.stats_snapshot().bypasses == 1


@pytest.mark.anyio
@pytest.mark.parametrize("status", [CacheHealth.HEALTHY, CacheHealth.UNAVAILABLE])
async def test_optional_cache_health_does_not_change_core_http_status(
    client, override_cache, status
) -> None:
    cache = HealthCache(status)
    override_cache(coordinator(cache))

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["cache"]["status"] == status.value
    assert response.json()["cache"]["backend"] == "redis"
    assert "redis://" not in response.text
    assert "password" not in response.text


@pytest.mark.anyio
async def test_health_exception_and_timeout_degrade_only_cache(client, override_cache) -> None:
    failing = HealthCache(CacheHealth.HEALTHY)
    failing.health = AsyncMock(side_effect=RuntimeError("private credential"))
    override_cache(coordinator(failing))
    exception_response = await client.get("/api/v1/health")

    async def hang():
        await asyncio.Event().wait()

    timing_out = HealthCache(CacheHealth.HEALTHY)
    timing_out.health = hang
    override_cache(coordinator(timing_out, timeout=0.001))
    timeout_response = await client.get("/api/v1/health")

    assert exception_response.status_code == timeout_response.status_code == 200
    assert exception_response.json()["cache"]["status"] == "unavailable"
    assert timeout_response.json()["cache"]["status"] == "unavailable"
    assert "private credential" not in exception_response.text


@pytest.mark.anyio
async def test_miss_write_hit_counters_and_hit_rate(client, cache_setup) -> None:
    _, _, cache_coordinator, _, rag = cache_setup

    await client.post("/api/v1/ask", json={"question": "Same"})
    await client.post("/api/v1/ask", json={"question": "Same"})

    stats = cache_coordinator.stats_snapshot()
    assert (stats.hits, stats.misses, stats.writes) == (1, 1, 1)
    assert stats.hit_rate == 0.5
    assert rag.generate.await_count == 1


@pytest.mark.anyio
async def test_bypass_corrupt_and_write_failure_accounting(client, cache_setup) -> None:
    cache, fingerprint, cache_coordinator, _, _ = cache_setup
    fingerprint.value = None
    await client.post("/api/v1/ask", json={"question": "Bypass"})
    fingerprint.value = "a" * 64
    lookup = await cache_coordinator.lookup(question="Corrupt")
    cache.values[lookup.key] = "invalid"
    cache.set_result = False
    await client.post("/api/v1/ask", json={"question": "Corrupt"})

    stats = cache_coordinator.stats_snapshot()
    assert stats.bypasses == 1
    assert stats.misses == 2  # direct clean lookup plus endpoint corrupt lookup
    assert stats.invalid_entries == 1
    assert stats.write_failures == 1
    assert stats.hits == stats.writes == 0


@pytest.mark.anyio
async def test_read_failure_is_distinct_from_miss_and_rag_still_succeeds(
    client, cache_setup
) -> None:
    cache, _, cache_coordinator, _, rag = cache_setup
    cache.get = AsyncMock(side_effect=RuntimeError("unavailable"))

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    stats = cache_coordinator.stats_snapshot()
    assert response.status_code == 200
    assert stats.read_failures == 1
    assert stats.misses == 0
    assert stats.writes == 1
    assert rag.generate.await_count == 1


@pytest.mark.anyio
async def test_cross_endpoint_accounting_is_shared(client, stream_cache_setup) -> None:
    _, _, cache_coordinator, _, provider, _ = stream_cache_setup
    rag = Mock()
    rag.generate = AsyncMock(return_value=generated_result())
    app.dependency_overrides[get_rag_generation_service] = lambda: rag

    await client.post("/api/v1/ask", json={"question": "Shared"})
    await client.post("/api/v1/ask/stream", json={"question": "Shared"})

    stats = cache_coordinator.stats_snapshot()
    assert (stats.hits, stats.misses, stats.writes) == (1, 1, 1)
    assert provider.calls == []
