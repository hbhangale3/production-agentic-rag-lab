from src.services.cache.base import CacheHealth, CacheReadResult, CacheReadStatus


class NoOpCache:
    """Deterministic fail-open cache for disabled or unavailable configuration."""

    def __init__(self, *, status: CacheHealth = CacheHealth.DISABLED) -> None:
        self.status = status

    async def get(self, key: str) -> None:
        return None

    async def get_result(self, key: str) -> CacheReadResult:
        status = CacheReadStatus.FAILURE if self.status is CacheHealth.UNAVAILABLE else CacheReadStatus.MISS
        return CacheReadResult(status)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        return False

    async def delete(self, key: str) -> bool:
        return False

    async def ping(self) -> bool:
        return False

    async def health(self) -> CacheHealth:
        return self.status

    async def close(self) -> None:
        return None
