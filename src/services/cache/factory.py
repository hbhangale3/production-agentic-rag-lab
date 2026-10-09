import logging
from typing import Any

from redis.asyncio import Redis
from src.config import Settings, get_settings
from src.services.cache.base import AsyncCache, CacheHealth
from src.services.cache.noop import NoOpCache
from src.services.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


def make_cache(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
) -> AsyncCache:
    """Build one worker-scoped cache without requiring Redis connectivity."""
    config = settings or get_settings()
    if not config.redis_enabled:
        logger.info("Redis cache disabled")
        return NoOpCache(status=CacheHealth.DISABLED)
    if client is not None:
        logger.info("Redis cache enabled with caller-owned client")
        return RedisCache(client=client, owns_client=False)
    try:
        owned_client = Redis.from_url(
            config.redis_url.get_secret_value(),
            decode_responses=True,
        )
    except Exception:
        logger.warning("Redis cache client construction failed; continuing without caching")
        return NoOpCache(status=CacheHealth.UNAVAILABLE)
    logger.info("Redis cache enabled")
    return RedisCache(client=owned_client, owns_client=True)
