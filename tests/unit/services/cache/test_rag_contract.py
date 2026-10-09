import json
import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from src.config import Settings
from src.schemas.ask import AskResponse, AskSource
from src.services.cache import (
    RAG_CACHE_SCHEMA_VERSION,
    RAG_EVIDENCE_VERSION,
    RAG_GROUNDING_VERSION,
    RAG_PROMPT_VERSION,
    RAG_RETRIEVAL_VERSION,
    ConfiguredCorpusFingerprintProvider,
    RAGCacheIdentity,
    RAGCacheIdentityFactory,
    RAGCacheKeyBuilder,
    deserialize_cached_response,
    serialize_cached_response,
)


def fingerprint() -> str:
    result = ConfiguredCorpusFingerprintProvider(generation="corpus-v1", chunk_index_name="arxiv-papers-chunks").get_fingerprint()
    assert result is not None
    return result


def identity(**overrides) -> RAGCacheIdentity:
    values = {
        "question": "How does AI improve healthcare access?",
        "llm_provider": "groq",
        "model": "openai/gpt-oss-120b",
        "retrieval_size": 5,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "rrf_k": 60,
        "candidate_multiplier": 4,
        "retrieval_max_results": 100,
        "chunk_index_name": "arxiv-papers-chunks",
        "evidence_max_tokens": 6000,
        "temperature": 0.1,
        "max_completion_tokens": 1024,
        "corpus_fingerprint": fingerprint(),
        **overrides,
    }
    return RAGCacheIdentity(**values)


def response() -> AskResponse:
    return AskResponse(
        answer="Unicode finding café 世界 [S2], then another [S1].",
        sources=[
            AskSource(
                citation="[S2]",
                retrieval_rank=2,
                arxiv_id="2401.00002",
                chunk_id="paper-2::chunk::001",
                chunk_index=1,
                title="Second Paper",
                section=None,
                content="Selected evidence two.",
                truncated=True,
                authors=["Author Two"],
                categories=["cs.AI"],
                published_date=None,
            ),
            AskSource(
                citation="[S1]",
                retrieval_rank=1,
                arxiv_id="2401.00001",
                chunk_id="paper-1::chunk::000",
                chunk_index=0,
                title="First Paper",
                section="Results",
                content="Selected evidence one.",
                truncated=False,
                authors=["Author One", "Author Other"],
                categories=["cs.IR", "cs.AI"],
                published_date=datetime(2024, 1, 2, tzinfo=UTC),
            ),
        ],
        retrieval_mode="bm25_fallback",
        model="original-model",
        prompt_tokens=123,
        completion_tokens=45,
    )


def test_identity_defaults_are_explicit_contract_versions() -> None:
    value = identity()

    assert value.schema_version == RAG_CACHE_SCHEMA_VERSION == "v1"
    assert value.prompt_version == RAG_PROMPT_VERSION
    assert value.retrieval_version == RAG_RETRIEVAL_VERSION
    assert value.evidence_version == RAG_EVIDENCE_VERSION
    assert value.grounding_version == RAG_GROUNDING_VERSION


def test_identity_factory_uses_answer_affecting_settings_and_exact_question() -> None:
    settings = Settings(
        _env_file=None,
        debug=False,
        groq_model="model-b",
        rag_retrieval_size=7,
        embedding_model="embedding-b",
        embedding_dimension=768,
        hybrid_rrf_k=80,
        hybrid_candidate_multiplier=3,
        vector_search_max_results=50,
        opensearch_chunk_index_name="chunks-b",
        evidence_context_max_tokens=5000,
        llm_temperature=0.25,
        llm_max_completion_tokens=900,
    )
    exact_question = "  Private Question?  "

    value = RAGCacheIdentityFactory(settings).create(question=exact_question, corpus_fingerprint="a" * 64)

    assert value.question == exact_question
    assert value.model == "model-b"
    assert value.retrieval_size == 7
    assert value.embedding_model == "embedding-b"
    assert value.embedding_dimension == 768
    assert value.rrf_k == 80
    assert value.candidate_multiplier == 3
    assert value.retrieval_max_results == 50
    assert value.chunk_index_name == "chunks-b"
    assert value.evidence_max_tokens == 5000
    assert value.temperature == 0.25
    assert value.max_completion_tokens == 900


def test_storage_and_deployment_settings_do_not_change_answer_key() -> None:
    common = {
        "_env_file": None,
        "debug": False,
        "groq_api_key": None,
    }
    first_settings = Settings(
        **common,
        redis_enabled=False,
        redis_url="redis://first:6379/0",
        rag_cache_ttl_seconds=60,
        llm_context_window_tokens=8192,
        llm_token_safety_margin=256,
    )
    second_settings = Settings(
        **common,
        redis_enabled=True,
        redis_url="redis://second:6379/9",
        rag_cache_ttl_seconds=3600,
        llm_context_window_tokens=16384,
        llm_token_safety_margin=512,
    )
    first = RAGCacheIdentityFactory(first_settings).create(question="Exact question", corpus_fingerprint="a" * 64)
    second = RAGCacheIdentityFactory(second_settings).create(question="Exact question", corpus_fingerprint="a" * 64)

    assert RAGCacheKeyBuilder.build(first) == RAGCacheKeyBuilder.build(second)


@pytest.mark.parametrize("question", ["", "   "])
def test_identity_rejects_blank_question(question: str) -> None:
    with pytest.raises(ValidationError):
        identity(question=question)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("question", "Different question"),
        ("llm_provider", "future-provider"),
        ("model", "different-model"),
        ("prompt_version", "grounded-rag-v2"),
        ("retrieval_version", "hybrid-bm25-bge-rrf-v2"),
        ("retrieval_size", 6),
        ("embedding_model", "different-embedding"),
        ("embedding_dimension", 768),
        ("rrf_k", 61),
        ("candidate_multiplier", 5),
        ("retrieval_max_results", 99),
        ("chunk_index_name", "different-chunks"),
        ("evidence_version", "evidence-context-v2"),
        ("evidence_max_tokens", 5999),
        ("temperature", 0.2),
        ("max_completion_tokens", 1000),
        ("grounding_version", "structural-grounding-v2"),
        ("corpus_fingerprint", "b" * 64),
        ("schema_version", "v2"),
    ],
)
def test_each_semantic_identity_change_produces_different_key(field, changed) -> None:
    original = identity()
    modified = original.model_copy(update={field: changed})

    assert RAGCacheKeyBuilder.build(original) != RAGCacheKeyBuilder.build(modified)


def test_key_is_deterministic_bounded_private_sha256() -> None:
    first = identity(question="private research question")
    equivalent = RAGCacheIdentity.model_validate(first.model_dump())

    first_key = RAGCacheKeyBuilder.build(first)
    second_key = RAGCacheKeyBuilder.build(equivalent)

    assert first_key == second_key
    assert first_key.startswith("rag:response:v1:")
    digest = first_key.rsplit(":", 1)[1]
    assert len(digest) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert len(first_key) == 80
    assert all(part not in first_key for part in ("private", "research", "question"))


def test_unicode_question_key_is_deterministic() -> None:
    value = identity(question="医療アクセス café")

    assert RAGCacheKeyBuilder.build(value) == RAGCacheKeyBuilder.build(value)


def test_known_key_vector_guards_canonicalization() -> None:
    assert RAGCacheKeyBuilder.build(identity()) == (
        "rag:response:v1:6db2a1a7711427ba1496b599b21bacb2dc623e62c275e13264ea478c70390f07"
    )


def test_corpus_fingerprint_is_deterministic_and_changes_with_generation_or_index() -> None:
    first = ConfiguredCorpusFingerprintProvider(generation="generation-1", chunk_index_name="chunks").get_fingerprint()
    equivalent = ConfiguredCorpusFingerprintProvider(generation="generation-1", chunk_index_name="chunks").get_fingerprint()
    changed_generation = ConfiguredCorpusFingerprintProvider(
        generation="generation-2", chunk_index_name="chunks"
    ).get_fingerprint()
    changed_index = ConfiguredCorpusFingerprintProvider(generation="generation-1", chunk_index_name="chunks-v2").get_fingerprint()

    assert first == equivalent
    assert first != changed_generation
    assert first != changed_index
    assert first is not None and re.fullmatch(r"[0-9a-f]{64}", first)


def test_corpus_fingerprint_provider_builds_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        debug=False,
        rag_corpus_generation="generation-7",
        opensearch_chunk_index_name="chunks-7",
    )

    from_settings = ConfiguredCorpusFingerprintProvider.from_settings(settings).get_fingerprint()
    explicit = ConfiguredCorpusFingerprintProvider(generation="generation-7", chunk_index_name="chunks-7").get_fingerprint()

    assert from_settings == explicit


@pytest.mark.parametrize(
    ("generation", "index"),
    [(None, "chunks"), ("", "chunks"), ("generation", None), ("generation", " ")],
)
def test_unavailable_corpus_state_returns_explicit_bypass_signal(generation, index) -> None:
    provider = ConfiguredCorpusFingerprintProvider(generation=generation, chunk_index_name=index)

    assert provider.get_fingerprint() is None


def test_cached_response_json_is_deterministic_unicode_and_round_trips() -> None:
    original = response()

    first = serialize_cached_response(original)
    second = serialize_cached_response(original)
    restored = deserialize_cached_response(first)

    assert first == second
    assert "café 世界" in first
    assert restored == original
    assert restored is not None
    assert [source.citation for source in restored.sources] == ["[S2]", "[S1]"]
    assert restored.sources[0].section is None
    assert restored.sources[0].truncated is True
    assert restored.sources[1].published_date == datetime(2024, 1, 2, tzinfo=UTC)
    assert restored.prompt_tokens == 123
    assert restored.completion_tokens == 45
    assert restored.model == "original-model"
    assert restored.retrieval_mode == "bm25_fallback"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "not-json",
        "[]",
        "{}",
        '{"schema_version":"v2","response":{}}',
        '{"schema_version":"v1"}',
        '{"schema_version":"v1","response":{"answer":123,"sources":[]}}',
        '{"schema_version":"v1","response":{"answer":" ","sources":[]}}',
        '{"schema_version":"v1","response":{"answer":"x","sources":"bad"}}',
        '{"schema_version":"v1","response":{"answer":"x","sources":[{"citation":"[S0]"}]}}',
        '{"schema_version":"v1","response":{"answer":"x","sources":[],"prompt_tokens":-1}}',
        '{"schema_version":"v1","response":{},"unexpected":"value"}',
    ],
)
def test_corrupt_cached_response_is_a_safe_miss(value: str) -> None:
    assert deserialize_cached_response(value) is None


def test_harmless_extra_response_field_follows_public_schema_ignore_policy() -> None:
    payload = json.loads(serialize_cached_response(response()))
    payload["response"]["future_harmless_field"] = "ignored"

    restored = deserialize_cached_response(json.dumps(payload))

    assert restored == response()


def test_complete_source_with_invalid_citation_is_rejected() -> None:
    payload = json.loads(serialize_cached_response(response()))
    payload["response"]["sources"][0]["citation"] = "[S0]"

    assert deserialize_cached_response(json.dumps(payload)) is None


def test_serialize_rejects_blank_answer_even_if_public_model_was_constructed() -> None:
    with pytest.raises(ValidationError):
        serialize_cached_response(AskResponse(answer=" ", sources=[]))
