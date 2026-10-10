"""Fail-open cache of successful, grounded agent answers in the agent's own namespace."""

import json
import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from src.config import Settings
from src.schemas.agent import AgentExecutionSummary, AgentSource
from src.services.agent.cache_identity import (
    AGENT_CACHE_SCHEMA_VERSION,
    AGENT_PIPELINE_VERSION,
    AgentCacheIdentityFactory,
    AgentCacheKeyBuilder,
)
from src.services.agent.config import AgentGraphConfig
from src.services.agent.prompt_identity import AgentPromptIdentityBundle
from src.services.cache.base import AsyncCache, CacheReadStatus
from src.services.cache.rag_contract import ConfiguredCorpusFingerprintProvider
from src.services.cache.rag_response_cache import CorpusFingerprintProvider, RAGCacheOutcome, RAGCacheWriteOutcome
from src.services.cache.stats import CacheStats, CacheStatsSnapshot

logger = logging.getLogger(__name__)


class AgentCachedAnswer(BaseModel):
    """Everything needed to reproduce one successful public answer without running the graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str
    sources: list[AgentSource] = Field(min_length=1)
    grounding_passed: Literal[True] = True
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    execution: AgentExecutionSummary = Field(default_factory=AgentExecutionSummary)

    @field_validator("answer")
    @classmethod
    def require_nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cached agent answer must not be blank")
        return value


class AgentCachedResponseEnvelope(BaseModel):
    """Versioned cache value; treated as untrusted when read back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["v1"] = AGENT_CACHE_SCHEMA_VERSION
    pipeline_version: Literal["v1"] = AGENT_PIPELINE_VERSION
    response: AgentCachedAnswer


def serialize_agent_answer(answer: AgentCachedAnswer) -> str:
    envelope = AgentCachedResponseEnvelope(response=answer)
    return json.dumps(envelope.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deserialize_agent_answer(value: object) -> AgentCachedAnswer | None:
    """Return only a fully validated grounded answer; anything else is not a hit."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            return None
        return AgentCachedResponseEnvelope.model_validate(payload).response
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return None


@dataclass(frozen=True)
class AgentCacheLookup:
    """One request's private cache state and optional validated hit."""

    key: str | None
    answer: AgentCachedAnswer | None = None
    outcome: RAGCacheOutcome = RAGCacheOutcome.BYPASS


class AgentResponseCache:
    """Apply the agent cache contract around successful grounded answers."""

    def __init__(
        self,
        *,
        cache: AsyncCache,
        settings: Settings,
        graph_config: AgentGraphConfig | None = None,
        fingerprint_provider: CorpusFingerprintProvider | None = None,
        stats: CacheStats | None = None,
    ) -> None:
        self.cache = cache
        self.enabled = settings.redis_enabled
        self.ttl_seconds = settings.rag_cache_ttl_seconds
        self.identity_factory = AgentCacheIdentityFactory(settings=settings, graph_config=graph_config)
        self.stats = stats or CacheStats()
        self.fingerprint_provider = fingerprint_provider or ConfiguredCorpusFingerprintProvider.from_settings(settings)

    async def lookup(self, *, question: str, prompts: AgentPromptIdentityBundle) -> AgentCacheLookup:
        """Return a validated exact hit, or private state for a later write. Never raises."""
        if not self.enabled:
            self.stats.increment("bypasses")
            return AgentCacheLookup(key=None, outcome=RAGCacheOutcome.BYPASS)
        try:
            fingerprint = self.fingerprint_provider.get_fingerprint()
            if fingerprint is None:
                self.stats.increment("bypasses")
                return AgentCacheLookup(key=None, outcome=RAGCacheOutcome.BYPASS)
            identity = self.identity_factory.create(
                question=question,
                corpus_fingerprint=fingerprint,
                prompts=prompts,
            )
            key = AgentCacheKeyBuilder.build(identity)
        except Exception:
            self.stats.increment("bypasses")
            logger.warning("Agent response cache identity unavailable; bypassing cache")
            return AgentCacheLookup(key=None, outcome=RAGCacheOutcome.BYPASS)

        try:
            result = await self.cache.get_result(key)
        except Exception:
            self.stats.increment("read_failures")
            logger.warning("Agent response cache read failed; continuing without cached response")
            return AgentCacheLookup(key=key, outcome=RAGCacheOutcome.FAILURE)
        if result.status is CacheReadStatus.FAILURE:
            self.stats.increment("read_failures")
            return AgentCacheLookup(key=key, outcome=RAGCacheOutcome.FAILURE)
        if result.status is CacheReadStatus.MISS or result.value is None:
            self.stats.increment("misses")
            return AgentCacheLookup(key=key, outcome=RAGCacheOutcome.MISS)

        answer = deserialize_agent_answer(result.value)
        if answer is None:
            self.stats.increment("misses")
            self.stats.increment("invalid_entries")
            logger.warning("Agent response cache entry was invalid; treating it as a miss")
            return AgentCacheLookup(key=key, outcome=RAGCacheOutcome.MISS)
        self.stats.increment("hits")
        return AgentCacheLookup(key=key, answer=answer, outcome=RAGCacheOutcome.HIT)

    async def store(self, lookup: AgentCacheLookup, answer: AgentCachedAnswer) -> RAGCacheWriteOutcome:
        """Best-effort store of an already successful grounded answer. Never raises."""
        if lookup.key is None:
            return RAGCacheWriteOutcome.SKIPPED
        try:
            stored = await self.cache.set(lookup.key, serialize_agent_answer(answer), ttl_seconds=self.ttl_seconds)
        except Exception:
            self.stats.increment("write_failures")
            logger.warning("Agent response cache write failed; returning generated response")
            return RAGCacheWriteOutcome.FAILED
        if not stored:
            self.stats.increment("write_failures")
            return RAGCacheWriteOutcome.FAILED
        self.stats.increment("writes")
        return RAGCacheWriteOutcome.STORED

    def stats_snapshot(self) -> CacheStatsSnapshot:
        return self.stats.snapshot()
