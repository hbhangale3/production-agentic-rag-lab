"""Provider-neutral asynchronous cache infrastructure."""

from src.services.cache.base import AsyncCache, CacheHealth
from src.services.cache.factory import make_cache
from src.services.cache.noop import NoOpCache
from src.services.cache.rag_contract import (
    RAG_CACHE_SCHEMA_VERSION,
    RAG_EVIDENCE_VERSION,
    RAG_GROUNDING_VERSION,
    RAG_PROMPT_VERSION,
    RAG_RETRIEVAL_VERSION,
    ConfiguredCorpusFingerprintProvider,
    RAGCachedResponseEnvelope,
    RAGCacheIdentity,
    RAGCacheIdentityFactory,
    RAGCacheKeyBuilder,
    deserialize_cached_response,
    serialize_cached_response,
)
from src.services.cache.redis_cache import RedisCache

__all__ = [
    "AsyncCache",
    "CacheHealth",
    "ConfiguredCorpusFingerprintProvider",
    "NoOpCache",
    "RAG_CACHE_SCHEMA_VERSION",
    "RAG_EVIDENCE_VERSION",
    "RAG_GROUNDING_VERSION",
    "RAG_PROMPT_VERSION",
    "RAG_RETRIEVAL_VERSION",
    "RAGCachedResponseEnvelope",
    "RAGCacheIdentity",
    "RAGCacheIdentityFactory",
    "RAGCacheKeyBuilder",
    "RedisCache",
    "deserialize_cached_response",
    "make_cache",
    "serialize_cached_response",
]
