import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError
from src.services.cache.base import CacheHealth, CacheReadResult, CacheReadStatus

logger = logging.getLogger(__name__)


class RedisCache:
    """Fail-open UTF-8 string cache backed by an asynchronous Redis client."""

    def __init__(self, *, client: Redis, owns_client: bool = False) -> None:
        self._client = client
        self._owns_client = owns_client
        self._closed = False

    async def get(self, key: str) -> str | None:
        return (await self.get_result(key)).value

    async def get_result(self, key: str) -> CacheReadResult:
        self._validate_key(key)
        if self._closed:
            return CacheReadResult(CacheReadStatus.FAILURE)
        try:
            value = await self._client.get(key)
        except RedisError:
            logger.warning("Redis cache GET failed; treating operation as a cache miss")
            return CacheReadResult(CacheReadStatus.FAILURE)
        if value is None:
            return CacheReadResult(CacheReadStatus.MISS)
        if isinstance(value, str):
            return CacheReadResult(CacheReadStatus.HIT, value)
        if isinstance(value, bytes):
            try:
                return CacheReadResult(CacheReadStatus.HIT, value.decode("utf-8"))
            except UnicodeDecodeError:
                logger.warning("Redis cache GET returned a non-UTF-8 value; treating operation as a cache miss")
                return CacheReadResult(CacheReadStatus.MISS)
        logger.warning("Redis cache GET returned an unsupported value; treating operation as a cache miss")
        return CacheReadResult(CacheReadStatus.MISS)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        self._validate_key(key)
        if not isinstance(value, str):
            raise TypeError("cache value must be a string")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer")
        if self._closed:
            return False
        try:
            result = await self._client.set(key, value, ex=ttl_seconds)
        except RedisError:
            logger.warning("Redis cache SET failed; continuing without caching")
            return False
        return bool(result)

    async def delete(self, key: str) -> bool:
        self._validate_key(key)
        if self._closed:
            return False
        try:
            deleted = await self._client.delete(key)
        except RedisError:
            logger.warning("Redis cache DELETE failed; continuing without cache invalidation")
            return False
        return bool(deleted)

    async def ping(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(await self._client.ping())
        except RedisError:
            logger.warning("Redis cache is unavailable")
            return False

    async def health(self) -> CacheHealth:
        return CacheHealth.HEALTHY if await self.ping() else CacheHealth.UNAVAILABLE

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            try:
                await self._client.aclose()
            except RedisError:
                logger.warning("Redis cache close failed during shutdown")

    @staticmethod
    def _validate_key(key: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("cache key must be a non-empty string")
