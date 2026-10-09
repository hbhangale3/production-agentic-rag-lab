from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class CacheHealth(StrEnum):
    DISABLED = "disabled"
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"


class CacheReadStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    FAILURE = "failure"


@dataclass(frozen=True)
class CacheReadResult:
    status: CacheReadStatus
    value: str | None = None


class AsyncCache(Protocol):
    """Provider-neutral asynchronous string-cache boundary."""

    async def get(self, key: str) -> str | None: ...

    async def get_result(self, key: str) -> CacheReadResult: ...

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> bool: ...

    async def delete(self, key: str) -> bool: ...

    async def ping(self) -> bool: ...

    async def health(self) -> CacheHealth: ...

    async def close(self) -> None: ...
