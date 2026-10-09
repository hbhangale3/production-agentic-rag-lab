from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from src.services.cache import CacheHealth, RedisCache


def cache(*, owns_client: bool = False):
    client = Mock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.ping = AsyncMock()
    client.aclose = AsyncMock()
    return RedisCache(client=client, owns_client=owns_client), client


@pytest.mark.anyio
async def test_get_hit_miss_and_utf8_bytes() -> None:
    instance, client = cache()
    client.get.side_effect = ["héllo 世界", None, "bytes".encode()]

    assert await instance.get("hit") == "héllo 世界"
    assert await instance.get("miss") is None
    assert await instance.get("bytes") == "bytes"


@pytest.mark.anyio
async def test_set_forwards_ttl_and_delete_normalizes_result() -> None:
    instance, client = cache()
    client.set.return_value = True
    client.delete.side_effect = [1, 0]

    assert await instance.set("key", "value", ttl_seconds=123) is True
    assert await instance.delete("key") is True
    assert await instance.delete("missing") is False
    client.set.assert_awaited_once_with("key", "value", ex=123)


@pytest.mark.anyio
async def test_ping_and_health() -> None:
    instance, client = cache()
    client.ping.side_effect = [True, True, False]

    assert await instance.ping() is True
    assert await instance.health() is CacheHealth.HEALTHY
    assert await instance.health() is CacheHealth.UNAVAILABLE


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["get", "set", "delete", "ping"])
async def test_redis_failures_are_fail_open(operation: str) -> None:
    instance, client = cache()
    getattr(client, operation).side_effect = RedisConnectionError("password-bearing private URL")

    if operation == "get":
        result = await instance.get("key")
        assert result is None
    elif operation == "set":
        assert await instance.set("key", "value", ttl_seconds=60) is False
    elif operation == "delete":
        assert await instance.delete("key") is False
    else:
        assert await instance.ping() is False


@pytest.mark.anyio
async def test_malformed_stored_value_is_a_safe_miss() -> None:
    instance, client = cache()
    client.get.side_effect = [b"\xff", object()]

    assert await instance.get("invalid-utf8") is None
    assert await instance.get("invalid-type") is None


@pytest.mark.anyio
async def test_owned_client_is_closed_once_and_operations_fail_open_after_close() -> None:
    instance, client = cache(owns_client=True)

    await instance.close()
    await instance.close()

    client.aclose.assert_awaited_once_with()
    assert await instance.get("key") is None
    assert await instance.set("key", "value", ttl_seconds=1) is False
    assert await instance.delete("key") is False
    assert await instance.ping() is False


@pytest.mark.anyio
async def test_caller_owned_client_is_not_closed() -> None:
    instance, client = cache(owns_client=False)

    await instance.close()

    client.aclose.assert_not_awaited()


@pytest.mark.anyio
async def test_close_failure_is_nonfatal_and_idempotent() -> None:
    instance, client = cache(owns_client=True)
    client.aclose.side_effect = RedisConnectionError("secret URL")

    await instance.close()
    await instance.close()

    client.aclose.assert_awaited_once_with()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "args", "message"),
    [
        ("get", ("",), "cache key"),
        ("set", ("key", 123), "cache value"),
        ("set", ("key", "value"), "ttl_seconds"),
    ],
)
async def test_programming_errors_are_rejected(method, args, message) -> None:
    instance, _ = cache()
    kwargs = {}
    if method == "set":
        kwargs = {"ttl_seconds": 0 if args[-1] == "value" else 60}

    with pytest.raises((TypeError, ValueError), match=message):
        await getattr(instance, method)(*args, **kwargs)
