import dataclasses
import hashlib
import json

import pytest
from pydantic import ValidationError
from src.config import Settings
from src.services.agent import (
    AGENT_CACHE_NAMESPACE,
    AGENT_PIPELINE_VERSION,
    AgentCacheIdentity,
    AgentCacheIdentityFactory,
    AgentCacheKeyBuilder,
    AgentGraphConfig,
    local_agent_prompt_bundle,
)
from src.services.agent.cache_identity import normalize_cache_question
from src.services.cache.rag_contract import RAG_CACHE_NAMESPACE, RAGCacheIdentityFactory, RAGCacheKeyBuilder
from src.services.prompts import PromptIdentity

QUESTION = "PRIVATE_QUESTION_SENTINEL What are the current healthcare issues in India?"
CORPUS = "a" * 64
PROMPTS = local_agent_prompt_bundle().identities


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, debug=False, **overrides)


def build_key(
    *,
    question: str = QUESTION,
    corpus: str = CORPUS,
    prompts=PROMPTS,
    config: AgentGraphConfig | None = None,
    **setting_overrides,
) -> str:
    factory = AgentCacheIdentityFactory(settings=settings(**setting_overrides), graph_config=config)
    return AgentCacheKeyBuilder.build(factory.create(question=question, corpus_fingerprint=corpus, prompts=prompts))


def with_prompt(key: str, **changes):
    return dataclasses.replace(PROMPTS, **{key: dataclasses.replace(getattr(PROMPTS, key), **changes)})


def test_key_is_namespaced_sha256_of_the_canonical_identity() -> None:
    identity = AgentCacheIdentityFactory(settings=settings()).create(
        question=QUESTION, corpus_fingerprint=CORPUS, prompts=PROMPTS
    )

    key = AgentCacheKeyBuilder.build(identity)

    canonical = json.dumps(identity.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert AgentCacheKeyBuilder.canonical_payload(identity) == canonical
    assert key == f"rag:agent-response:v1:{hashlib.sha256(canonical.encode()).hexdigest()}"
    assert (AGENT_CACHE_NAMESPACE, AGENT_PIPELINE_VERSION) == ("rag:agent-response", "v1")
    assert len(key.rsplit(":", 1)[1]) == 64
    assert identity.pipeline_version == "v1"


def test_known_vector_for_a_fixed_identity() -> None:
    fixed = PromptIdentity(name="p", version="local-v1", label="production", fingerprint="0" * 64)
    prompts = dataclasses.replace(PROMPTS, **{field.name: fixed for field in dataclasses.fields(PROMPTS)})
    identity = AgentCacheIdentity(
        question="What is RAG?",
        corpus_fingerprint="a" * 64,
        retrieval_size=5,
        rrf_k=60,
        candidate_multiplier=4,
        retrieval_max_results=100,
        chunk_index_name="arxiv-papers-chunks",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        evidence_max_tokens=6000,
        chunk_target_words=600,
        chunk_overlap_words=100,
        chunk_min_words=100,
        llm_provider="groq",
        model="test-model",
        temperature=0.1,
        max_completion_tokens=1024,
        guardrail_threshold=60,
        evidence_sufficiency_threshold=60,
        max_local_retrieval_attempts=2,
        live_fallback_enabled=True,
        live_arxiv_max_results=5,
        live_pdf_max_papers=2,
        live_max_chunks_per_paper=8,
        final_evidence_max_sources=5,
        answer_grounding_threshold=60,
        max_grounding_attempts=2,
        prompts=prompts.as_cache_payload(),
    )

    assert AgentCacheKeyBuilder.build(identity) == KNOWN_VECTOR_KEY


KNOWN_VECTOR_KEY = "rag:agent-response:v1:e2b84608f817ba4a787d56799ab7b6e6a6153fb7fe88c37f54d082d48b3b7173"


def test_same_identity_gives_the_same_key_and_whitespace_does_not_matter() -> None:
    assert build_key() == build_key()
    assert build_key(question=f"  {QUESTION}\n") == build_key()
    assert build_key(question=QUESTION.replace(" ", "   ")) == build_key()
    assert normalize_cache_question("  a \n b\t") == "a b"
    assert build_key(config=AgentGraphConfig()) == build_key(config=None)


def test_raw_question_is_absent_from_the_key() -> None:
    key = build_key()

    assert "PRIVATE_QUESTION_SENTINEL" not in key
    assert "India" not in key
    assert key.startswith("rag:agent-response:v1:")


def test_agent_namespace_is_distinct_from_the_week6_rag_cache() -> None:
    week6 = RAGCacheKeyBuilder.build(
        RAGCacheIdentityFactory(settings()).create(question=QUESTION, corpus_fingerprint=CORPUS)
    )

    assert RAG_CACHE_NAMESPACE == "rag:response"
    assert week6.startswith("rag:response:v1:")
    assert not build_key().startswith("rag:response:")
    assert build_key().rsplit(":", 1)[1] != week6.rsplit(":", 1)[1]


@pytest.mark.parametrize(
    "change",
    [
        {"question": QUESTION + " And costs?"},
        {"question": QUESTION.lower()},
        {"corpus": "b" * 64},
        {"embedding_model": "other/embedding-model"},
        {"embedding_dimension": 768},
        {"hybrid_rrf_k": 61},
        {"hybrid_candidate_multiplier": 5},
        {"opensearch_chunk_index_name": "other-chunks"},
        {"evidence_context_max_tokens": 5000},
        {"chunk_target_words": 500},
        {"groq_model": "other-model"},
        {"llm_temperature": 0.2},
        {"llm_max_completion_tokens": 512},
    ],
)
def test_answer_affecting_settings_change_the_key(change: dict) -> None:
    assert build_key(**change) != build_key()


@pytest.mark.parametrize(
    "overrides",
    [
        {"retrieval_size": 6},
        {"guardrail_threshold": 61},
        {"evidence_sufficiency_threshold": 61},
        {"max_local_retrieval_attempts": 1},
        {"live_fallback_enabled": False},
        {"live_arxiv_max_results": 4},
        {"live_pdf_max_papers": 1},
        {"live_max_chunks_per_paper": 9},
        {"final_evidence_max_sources": 4},
        {"answer_grounding_threshold": 61},
        {"max_grounding_attempts": 1},
    ],
)
def test_agent_thresholds_and_limits_change_the_key(overrides: dict) -> None:
    assert build_key(config=AgentGraphConfig(**overrides)) != build_key()


def test_pipeline_version_changes_the_key() -> None:
    identity = AgentCacheIdentityFactory(settings=settings()).create(
        question=QUESTION, corpus_fingerprint=CORPUS, prompts=PROMPTS
    )
    bumped = identity.model_copy(update={"pipeline_version": "v2"})

    assert AgentCacheKeyBuilder.build(bumped) != AgentCacheKeyBuilder.build(identity)


@pytest.mark.parametrize(
    "prompt",
    ["guardrail", "evidence_grader", "query_rewrite", "live_selector", "generation", "regeneration", "answer_grounding"],
)
@pytest.mark.parametrize(
    "changes",
    [{"version": "local-v2"}, {"fingerprint": "f" * 64}, {"label": "staging"}, {"version": "langfuse-v3", "fingerprint": "e" * 64}],
)
def test_any_prompt_version_fingerprint_or_label_change_changes_the_key(prompt: str, changes: dict) -> None:
    assert build_key(prompts=with_prompt(prompt, **changes)) != build_key()


def test_version_change_alone_invalidates_even_with_identical_content() -> None:
    same_content_new_version = with_prompt("generation", version="local-v2")

    assert same_content_new_version.generation.fingerprint == PROMPTS.generation.fingerprint
    assert build_key(prompts=same_content_new_version) != build_key()


@pytest.mark.parametrize(
    "operational",
    [
        {"langfuse_capture_content": True},
        {"langfuse_host": "https://other.langfuse.example"},
        {"langfuse_timeout_seconds": 9},
        {"langfuse_prompt_management_enabled": True},
        {"langfuse_prompt_cache_ttl_seconds": 5},
        {"llm_input_cost_per_million_tokens": 1.0, "llm_output_cost_per_million_tokens": 2.0},
        {"llm_timeout_seconds": 9.0},
        {"llm_max_retries": 5},
        {"redis_enabled": True},
        {"rag_cache_ttl_seconds": 60},
        {"arxiv_timeout": 5.0},
        {"embedding_device": "cuda"},
        {"embedding_batch_size": 8},
        {"environment": "production"},
    ],
)
def test_operational_only_settings_do_not_change_the_key(operational: dict) -> None:
    assert build_key(**operational) == build_key()


def test_identity_contains_no_operational_or_runtime_fields() -> None:
    fields = set(AgentCacheIdentity.model_fields)

    assert not {name for name in fields if any(term in name for term in ("langfuse", "redis", "host", "timeout", "cost", "trace", "request", "time"))}
    assert {"question", "corpus_fingerprint", "pipeline_version", "prompts", "model", "embedding_model"} <= fields
    assert set(AgentGraphConfig.model_fields) <= fields


def test_identity_is_immutable_and_strictly_validated() -> None:
    identity = AgentCacheIdentityFactory(settings=settings()).create(
        question=QUESTION, corpus_fingerprint=CORPUS, prompts=PROMPTS
    )
    payload = identity.model_dump()

    with pytest.raises(ValidationError):
        identity.question = "other"  # type: ignore[misc]
    for invalid in (
        {"question": "  "},
        {"corpus_fingerprint": "not-a-digest"},
        {"prompts": {}},
        {"prompts": {**payload["prompts"], "extra": payload["prompts"]["guardrail"]}},
        {"prompts": {**payload["prompts"], "guardrail": {**payload["prompts"]["guardrail"], "fingerprint": "short"}}},
        {"unknown_field": 1},
    ):
        with pytest.raises(ValidationError):
            AgentCacheIdentity(**{**payload, **invalid})
    with pytest.raises(TypeError):
        AgentCacheKeyBuilder.build("not an identity")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AgentCacheIdentityFactory(settings=settings()).create(question="   ", corpus_fingerprint=CORPUS, prompts=PROMPTS)
