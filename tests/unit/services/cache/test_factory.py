from unittest.mock import Mock, patch

import pytest
from src.config import Settings
from src.services.cache import CacheHealth, NoOpCache, RedisCache, make_cache


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, debug=False, **overrides)


def test_disabled_factory_does_not_construct_redis_client() -> None:
    with patch("src.services.cache.factory.Redis.from_url") as from_url:
        result = make_cache(settings(redis_enabled=False))

    assert isinstance(result, NoOpCache)
    assert result.status is CacheHealth.DISABLED
    from_url.assert_not_called()


def test_enabled_factory_builds_owned_async_redis_client_without_connecting() -> None:
    client = Mock()
    config = settings(redis_enabled=True, redis_url="redis://cache.internal:6379/0")
    with patch("src.services.cache.factory.Redis.from_url", return_value=client) as from_url:
        result = make_cache(config)

    assert isinstance(result, RedisCache)
    assert result._client is client
    assert result._owns_client is True
    from_url.assert_called_once_with("redis://cache.internal:6379/0", decode_responses=True)


def test_injected_client_is_caller_owned() -> None:
    client = Mock()

    result = make_cache(settings(redis_enabled=True), client=client)

    assert isinstance(result, RedisCache)
    assert result._client is client
    assert result._owns_client is False


def test_disabled_factory_ignores_injected_client() -> None:
    result = make_cache(settings(redis_enabled=False), client=Mock())

    assert isinstance(result, NoOpCache)
    assert result.status is CacheHealth.DISABLED


def test_construction_failure_degrades_to_unavailable_noop_without_secret(caplog) -> None:
    config = settings(redis_enabled=True, redis_url="redis://cache.internal/0")
    with patch(
        "src.services.cache.factory.Redis.from_url",
        side_effect=ValueError("private connection detail"),
    ):
        result = make_cache(config)

    assert isinstance(result, NoOpCache)
    assert result.status is CacheHealth.UNAVAILABLE
    assert "private connection detail" not in caplog.text
    assert "redis://cache.internal/0" not in caplog.text


@pytest.mark.anyio
async def test_noop_cache_contract_is_deterministic() -> None:
    instance = NoOpCache()

    assert await instance.get("key") is None
    assert await instance.set("key", "value", ttl_seconds=60) is False
    assert await instance.delete("key") is False
    assert await instance.ping() is False
    assert await instance.health() is CacheHealth.DISABLED
    await instance.close()
    await instance.close()
