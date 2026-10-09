import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from src.config import Settings
from src.schemas.ask import AskResponse

RAG_CACHE_SCHEMA_VERSION = "v1"
RAG_CACHE_NAMESPACE = "rag:response"
RAG_PROMPT_VERSION = "grounded-rag-v1"
RAG_RETRIEVAL_VERSION = "hybrid-bm25-bge-rrf-v1"
RAG_EVIDENCE_VERSION = "evidence-context-v1"
RAG_GROUNDING_VERSION = "structural-grounding-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class RAGCacheIdentity(BaseModel):
    """Immutable exact identity for one answer-producing RAG configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = RAG_CACHE_SCHEMA_VERSION
    question: str
    llm_provider: str
    model: str
    prompt_version: str = RAG_PROMPT_VERSION
    retrieval_version: str = RAG_RETRIEVAL_VERSION
    retrieval_size: int = Field(gt=0)
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    candidate_multiplier: int = Field(gt=0)
    retrieval_max_results: int = Field(gt=0)
    chunk_index_name: str
    evidence_version: str = RAG_EVIDENCE_VERSION
    evidence_max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    max_completion_tokens: int = Field(gt=0)
    grounding_version: str = RAG_GROUNDING_VERSION
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "question",
        "llm_provider",
        "model",
        "schema_version",
        "prompt_version",
        "retrieval_version",
        "embedding_model",
        "chunk_index_name",
        "evidence_version",
        "grounding_version",
    )
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cache identity text fields must not be blank")
        return value


class RAGCacheIdentityFactory:
    """Bind answer-affecting settings while preserving the exact validated question."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(self, *, question: str, corpus_fingerprint: str) -> RAGCacheIdentity:
        return RAGCacheIdentity(
            question=question,
            llm_provider=self.settings.llm_provider,
            model=self.settings.groq_model,
            retrieval_size=self.settings.rag_retrieval_size,
            embedding_model=self.settings.embedding_model,
            embedding_dimension=self.settings.embedding_dimension,
            rrf_k=self.settings.hybrid_rrf_k,
            candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            retrieval_max_results=self.settings.vector_search_max_results,
            chunk_index_name=self.settings.opensearch_chunk_index_name,
            evidence_max_tokens=self.settings.evidence_context_max_tokens,
            temperature=self.settings.llm_temperature,
            max_completion_tokens=self.settings.llm_max_completion_tokens,
            corpus_fingerprint=corpus_fingerprint,
        )


class RAGCacheKeyBuilder:
    """Pure deterministic SHA-256 key derivation with versioned namespace."""

    @staticmethod
    def build(identity: RAGCacheIdentity) -> str:
        if not isinstance(identity, RAGCacheIdentity):
            raise TypeError("identity must be a RAGCacheIdentity")
        canonical = _canonical_json(identity.model_dump(mode="json"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{RAG_CACHE_NAMESPACE}:{identity.schema_version}:{digest}"


class ConfiguredCorpusFingerprintProvider:
    """Cheap manual generation boundary for current paper/chunk index state."""

    def __init__(self, *, generation: str | None, chunk_index_name: str | None) -> None:
        self.generation = generation
        self.chunk_index_name = chunk_index_name

    @classmethod
    def from_settings(cls, settings: Settings) -> "ConfiguredCorpusFingerprintProvider":
        return cls(
            generation=settings.rag_corpus_generation,
            chunk_index_name=settings.opensearch_chunk_index_name,
        )

    def get_fingerprint(self) -> str | None:
        if not isinstance(self.generation, str) or not self.generation.strip():
            return None
        if not isinstance(self.chunk_index_name, str) or not self.chunk_index_name.strip():
            return None
        state = {
            "chunk_index_name": self.chunk_index_name,
            "generation": self.generation,
        }
        return hashlib.sha256(_canonical_json(state).encode("utf-8")).hexdigest()


class RAGCachedResponseEnvelope(BaseModel):
    """Versioned, validated cache value containing the authoritative public response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["v1"] = RAG_CACHE_SCHEMA_VERSION
    response: AskResponse

    @field_validator("response")
    @classmethod
    def require_nonempty_answer(cls, response: AskResponse) -> AskResponse:
        if not response.answer.strip():
            raise ValueError("cached response answer must not be blank")
        return response


def serialize_cached_response(response: AskResponse) -> str:
    """Serialize one fully constructed public response as deterministic JSON."""
    envelope = RAGCachedResponseEnvelope(response=response)
    return _canonical_json(envelope.model_dump(mode="json"))


def deserialize_cached_response(value: str) -> AskResponse | None:
    """Treat a cache value as untrusted and return only a fully validated response."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            return None
        envelope = RAGCachedResponseEnvelope.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return None
    return envelope.response
