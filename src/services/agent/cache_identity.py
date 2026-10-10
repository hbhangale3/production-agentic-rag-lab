"""Cache identity for agent responses: a contract only, nothing here reads or writes a cache.

The key covers everything that can change an agent answer: the question,
corpus, retrieval, models, generation settings, agent thresholds and limits,
the pipeline version, and every prompt's identity. Operational settings such
as hosts, timeouts, and telemetry options are deliberately absent.
"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.config import Settings
from src.services.agent.config import AgentGraphConfig
from src.services.agent.prompt_identity import AgentPromptIdentityBundle
from src.services.cache.rag_contract import RAG_EVIDENCE_VERSION, RAG_GROUNDING_VERSION, RAG_RETRIEVAL_VERSION
from src.services.rag.service import LENGTH_RECOVERY_BUDGET_MULTIPLIER

AGENT_PIPELINE_VERSION = "v1"
AGENT_CACHE_SCHEMA_VERSION = "v1"
AGENT_CACHE_NAMESPACE = "rag:agent-response"


def normalize_cache_question(question: str) -> str:
    """Trim and collapse whitespace; wording and case are preserved."""

    return " ".join(question.split())


class AgentPromptCacheIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    label: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class AgentCacheIdentity(BaseModel):
    """Immutable exact identity for one answer-producing agent configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = AGENT_CACHE_SCHEMA_VERSION
    pipeline_version: str = AGENT_PIPELINE_VERSION
    question: str
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    retrieval_version: str = RAG_RETRIEVAL_VERSION
    retrieval_size: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    candidate_multiplier: int = Field(gt=0)
    retrieval_max_results: int = Field(gt=0)
    chunk_index_name: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)

    evidence_version: str = RAG_EVIDENCE_VERSION
    evidence_max_tokens: int = Field(gt=0)
    chunk_target_words: int = Field(gt=0)
    chunk_overlap_words: int = Field(ge=0)
    chunk_min_words: int = Field(gt=0)

    llm_provider: str
    model: str
    temperature: float = Field(ge=0, le=2)
    max_completion_tokens: int = Field(gt=0)
    # Length recovery decides whether a cut-off answer can be redone, so it affects the outcome.
    length_recovery_max_tokens: int = Field(gt=0)
    length_recovery_multiplier: int = Field(default=LENGTH_RECOVERY_BUDGET_MULTIPLIER, gt=0)
    structural_grounding_version: str = RAG_GROUNDING_VERSION

    guardrail_threshold: int = Field(ge=0, le=100)
    evidence_sufficiency_threshold: int = Field(ge=0, le=100)
    max_local_retrieval_attempts: int = Field(ge=1)
    live_fallback_enabled: bool
    live_arxiv_max_results: int = Field(ge=1)
    live_pdf_max_papers: int = Field(ge=1)
    live_max_chunks_per_paper: int = Field(ge=1)
    final_evidence_max_sources: int = Field(ge=1)
    answer_grounding_threshold: int = Field(ge=0, le=100)
    max_grounding_attempts: int = Field(ge=1)

    prompts: dict[str, AgentPromptCacheIdentity]

    @field_validator(
        "schema_version",
        "pipeline_version",
        "question",
        "retrieval_version",
        "chunk_index_name",
        "embedding_model",
        "evidence_version",
        "llm_provider",
        "model",
        "structural_grounding_version",
    )
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cache identity text fields must not be blank")
        return value

    @field_validator("prompts")
    @classmethod
    def require_every_agent_prompt(cls, value: dict[str, AgentPromptCacheIdentity]) -> dict:
        expected = set(AgentPromptIdentityBundle.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("cache identity must include exactly the agent's prompt identities")
        return value


class AgentCacheIdentityFactory:
    """Bind answer-affecting settings and agent configuration to one resolved prompt bundle."""

    def __init__(self, *, settings: Settings, graph_config: AgentGraphConfig | None = None) -> None:
        self.settings = settings
        self.graph_config = graph_config or AgentGraphConfig()

    def create(
        self,
        *,
        question: str,
        corpus_fingerprint: str,
        prompts: AgentPromptIdentityBundle,
    ) -> AgentCacheIdentity:
        settings, config = self.settings, self.graph_config
        return AgentCacheIdentity(
            question=normalize_cache_question(question),
            corpus_fingerprint=corpus_fingerprint,
            retrieval_size=config.retrieval_size,
            rrf_k=settings.hybrid_rrf_k,
            candidate_multiplier=settings.hybrid_candidate_multiplier,
            retrieval_max_results=settings.vector_search_max_results,
            chunk_index_name=settings.opensearch_chunk_index_name,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            evidence_max_tokens=settings.evidence_context_max_tokens,
            chunk_target_words=settings.chunk_target_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            chunk_min_words=settings.chunk_min_words,
            llm_provider=settings.llm_provider,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_completion_tokens,
            # The same setting RAGGenerationService.from_settings reads for the recovery policy.
            length_recovery_max_tokens=settings.llm_length_recovery_max_tokens,
            guardrail_threshold=config.guardrail_threshold,
            evidence_sufficiency_threshold=config.evidence_sufficiency_threshold,
            max_local_retrieval_attempts=config.max_local_retrieval_attempts,
            live_fallback_enabled=config.live_fallback_enabled,
            live_arxiv_max_results=config.live_arxiv_max_results,
            live_pdf_max_papers=config.live_pdf_max_papers,
            live_max_chunks_per_paper=config.live_max_chunks_per_paper,
            final_evidence_max_sources=config.final_evidence_max_sources,
            answer_grounding_threshold=config.answer_grounding_threshold,
            max_grounding_attempts=config.max_grounding_attempts,
            prompts=prompts.as_cache_payload(),
        )


class AgentCacheKeyBuilder:
    """Pure deterministic SHA-256 key derivation in the agent's own namespace."""

    @staticmethod
    def canonical_payload(identity: AgentCacheIdentity) -> str:
        if not isinstance(identity, AgentCacheIdentity):
            raise TypeError("identity must be an AgentCacheIdentity")
        return json.dumps(identity.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def build(cls, identity: AgentCacheIdentity) -> str:
        digest = hashlib.sha256(cls.canonical_payload(identity).encode("utf-8")).hexdigest()
        return f"{AGENT_CACHE_NAMESPACE}:{identity.schema_version}:{digest}"
