import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from src.config import Settings
from src.schemas.ask import AskResponse
from src.services.cache.base import AsyncCache, CacheHealth, CacheReadStatus
from src.services.cache.rag_contract import (
    ConfiguredCorpusFingerprintProvider,
    RAGCacheIdentityFactory,
    RAGCacheKeyBuilder,
    deserialize_cached_response,
    serialize_cached_response,
)
from src.services.cache.stats import CacheStats, CacheStatsSnapshot

logger = logging.getLogger(__name__)


class CorpusFingerprintProvider(Protocol):
    def get_fingerprint(self) -> str | None: ...


@dataclass(frozen=True)
class RAGCacheLookup:
    """One request's private cache state and optional validated hit."""

    key: str | None
    response: AskResponse | None = None


class RAGResponseCacheCoordinator:
    """Apply the exact RAG cache contract around successful generation."""

    def __init__(
        self,
        *,
        cache: AsyncCache,
        settings: Settings,
        fingerprint_provider: CorpusFingerprintProvider | None = None,
        stats: CacheStats | None = None,
    ) -> None:
        self.cache = cache
        self.enabled = settings.redis_enabled
        self.ttl_seconds = settings.rag_cache_ttl_seconds
        self.health_timeout_seconds = settings.redis_health_timeout_seconds
        self.identity_factory = RAGCacheIdentityFactory(settings)
        self.stats = stats or CacheStats()
        self.fingerprint_provider = fingerprint_provider or (
            ConfiguredCorpusFingerprintProvider.from_settings(settings)
        )

    async def lookup(self, *, question: str) -> RAGCacheLookup:
        """Return a validated exact hit, or private state for a later write."""
        if not self.enabled:
            self.stats.increment("bypasses")
            logger.debug("RAG response cache bypassed because caching is disabled")
            return RAGCacheLookup(key=None)

        try:
            fingerprint = self.fingerprint_provider.get_fingerprint()
            if fingerprint is None:
                self.stats.increment("bypasses")
                logger.debug("RAG response cache bypassed because corpus state is unavailable")
                return RAGCacheLookup(key=None)
            identity = self.identity_factory.create(
                question=question,
                corpus_fingerprint=fingerprint,
            )
            key = RAGCacheKeyBuilder.build(identity)
        except Exception:
            self.stats.increment("bypasses")
            logger.warning("RAG response cache identity unavailable; bypassing cache")
            return RAGCacheLookup(key=None)

        try:
            get_result = getattr(self.cache, "get_result", None)
            if get_result is None:
                payload = await self.cache.get(key)
                read_status = CacheReadStatus.MISS if payload is None else CacheReadStatus.HIT
            else:
                result = await get_result(key)
                payload = result.value
                read_status = result.status
        except Exception:
            self.stats.increment("read_failures")
            logger.warning("RAG response cache read failed; continuing without cached response")
            return RAGCacheLookup(key=key)
        if read_status is CacheReadStatus.FAILURE:
            self.stats.increment("read_failures")
            logger.debug("RAG response cache read was unavailable")
            return RAGCacheLookup(key=key)
        if read_status is CacheReadStatus.MISS or payload is None:
            self.stats.increment("misses")
            logger.debug("RAG response cache miss")
            return RAGCacheLookup(key=key)

        response = deserialize_cached_response(payload)
        if response is None:
            self.stats.increment("misses")
            self.stats.increment("invalid_entries")
            logger.warning("RAG response cache entry was invalid; treating it as a miss")
            return RAGCacheLookup(key=key)
        self.stats.increment("hits")
        logger.debug("RAG response cache hit")
        return RAGCacheLookup(key=key, response=response)

    async def store(self, lookup: RAGCacheLookup, response: AskResponse) -> None:
        """Best-effort store of an already successful public response."""
        if lookup.key is None:
            return
        try:
            payload = serialize_cached_response(response)
        except Exception:
            self.stats.increment("write_failures")
            logger.warning("RAG response serialization failed; skipping cache write")
            return
        try:
            stored = await self.cache.set(
                lookup.key,
                payload,
                ttl_seconds=self.ttl_seconds,
            )
        except Exception:
            self.stats.increment("write_failures")
            logger.warning("RAG response cache write failed; returning generated response")
            return
        if not stored:
            self.stats.increment("write_failures")
            logger.debug("RAG response cache write was unavailable")
            return
        self.stats.increment("writes")

    def stats_snapshot(self) -> CacheStatsSnapshot:
        return self.stats.snapshot()

    async def health(self) -> CacheHealth:
        """Return bounded optional-cache health without affecting core readiness."""
        if not self.enabled:
            return CacheHealth.DISABLED
        try:
            return await asyncio.wait_for(
                self.cache.health(),
                timeout=self.health_timeout_seconds,
            )
        except Exception:
            logger.warning("Cache health check failed; reporting cache unavailable")
            return CacheHealth.UNAVAILABLE
