from typing import Any

import pytest
from pydantic import ValidationError
from src.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Build settings independently of the developer shell and local .env."""
    return Settings(_env_file=None, debug=False, **overrides)


def test_week4_retrieval_defaults() -> None:
    settings = make_settings()

    assert settings.embedding_provider == "sentence_transformer"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dimension == 384
    assert settings.embedding_device == "cpu"
    assert settings.embedding_batch_size == 32
    assert settings.opensearch_chunk_index_name == "arxiv-papers-chunks"
    assert settings.chunk_target_words == 600
    assert settings.chunk_overlap_words == 100
    assert settings.chunk_min_words == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_dimension", 0),
        ("embedding_batch_size", 0),
        ("chunk_target_words", 0),
        ("chunk_overlap_words", -1),
        ("chunk_min_words", 0),
    ],
)
def test_positive_week4_sizes_are_required(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


@pytest.mark.parametrize("overlap", [600, 601])
def test_chunk_overlap_must_be_smaller_than_target(overlap: int) -> None:
    with pytest.raises(ValidationError, match="chunk overlap must be smaller"):
        make_settings(chunk_target_words=600, chunk_overlap_words=overlap)


def test_minimum_chunk_size_must_not_exceed_target() -> None:
    with pytest.raises(ValidationError, match="minimum chunk size must not exceed"):
        make_settings(chunk_target_words=600, chunk_min_words=601)


def test_minimum_chunk_size_may_equal_target() -> None:
    settings = make_settings(chunk_target_words=600, chunk_min_words=600)

    assert settings.chunk_min_words == settings.chunk_target_words


def test_week5_llm_defaults_do_not_require_credentials() -> None:
    settings = make_settings()

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key is None
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.llm_timeout_seconds == 30.0
    assert settings.llm_max_retries == 2
    assert settings.evidence_context_max_tokens == 6000
    assert settings.llm_temperature == 0.1
    assert settings.llm_max_completion_tokens == 1024
    assert settings.llm_context_window_tokens == 8192
    assert settings.llm_token_safety_margin == 256
    assert settings.rag_retrieval_size == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("groq_model", "  "),
        ("llm_timeout_seconds", 0),
        ("llm_timeout_seconds", -1),
        ("llm_max_retries", -1),
        ("evidence_context_max_tokens", 0),
        ("llm_temperature", -0.1),
        ("llm_temperature", 2.1),
        ("llm_max_completion_tokens", 0),
        ("llm_context_window_tokens", 0),
        ("llm_token_safety_margin", -1),
        ("rag_retrieval_size", 0),
        ("rag_retrieval_size", 101),
    ],
)
def test_invalid_week5_llm_settings_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_groq_api_key_is_redacted_from_settings_representation() -> None:
    settings = make_settings(groq_api_key="super-secret-test-key")

    assert "super-secret-test-key" not in repr(settings)
    assert "**********" in repr(settings)


def test_generation_reserves_prompt_capacity() -> None:
    with pytest.raises(ValidationError, match="leave prompt capacity"):
        make_settings(
            llm_context_window_tokens=768,
            llm_max_completion_tokens=512,
            llm_token_safety_margin=256,
        )


def test_week6_redis_defaults_are_optional_and_secret_safe() -> None:
    settings = make_settings()

    assert settings.redis_enabled is False
    assert settings.redis_url.get_secret_value() == "redis://127.0.0.1:6379/0"
    assert settings.redis_health_timeout_seconds == 1.0
    assert settings.rag_cache_ttl_seconds == 86_400
    assert settings.rag_corpus_generation == "corpus-v1"
    assert "redis://127.0.0.1:6379/0" not in repr(settings)


def test_enabled_redis_configuration_is_valid_without_connecting() -> None:
    settings = make_settings(
        redis_enabled=True,
        redis_url="redis://cache.test:6379/2",
        rag_cache_ttl_seconds=60,
    )

    assert settings.redis_enabled is True
    assert settings.redis_url.get_secret_value() == "redis://cache.test:6379/2"
    assert settings.rag_cache_ttl_seconds == 60


@pytest.mark.parametrize("ttl", [0, -1])
def test_cache_ttl_must_be_positive(ttl: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(rag_cache_ttl_seconds=ttl)


@pytest.mark.parametrize("timeout", [0, -1, 10.1])
def test_redis_health_timeout_is_small_and_positive(timeout: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(redis_health_timeout_seconds=timeout)


def test_enabled_redis_requires_nonblank_url() -> None:
    with pytest.raises(ValidationError, match="Redis URL must not be blank"):
        make_settings(redis_enabled=True, redis_url="   ")


def test_disabled_redis_allows_blank_url_and_requires_no_server() -> None:
    settings = make_settings(redis_enabled=False, redis_url="")

    assert settings.redis_enabled is False


@pytest.mark.parametrize("generation", ["", "   "])
def test_corpus_generation_must_not_be_blank(generation: str) -> None:
    with pytest.raises(ValidationError, match="corpus generation"):
        make_settings(rag_corpus_generation=generation)


def test_langfuse_defaults_are_disabled_and_require_no_credentials() -> None:
    settings = make_settings()

    assert settings.langfuse_enabled is False
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_host == "https://cloud.langfuse.com"
    assert settings.langfuse_capture_content is False
    assert settings.langfuse_timeout_seconds == 5


@pytest.mark.parametrize(
    ("public_key", "secret_key", "message"),
    [
        (None, "test-secret", "public key"),
        ("   ", "test-secret", "public key"),
        ("test-public", None, "secret key"),
        ("test-public", "   ", "secret key"),
    ],
)
def test_enabled_langfuse_requires_both_keys(public_key, secret_key, message) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(
            langfuse_enabled=True,
            langfuse_public_key=public_key,
            langfuse_secret_key=secret_key,
        )


def test_enabled_langfuse_supports_self_hosting_and_redacts_secret() -> None:
    settings = make_settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-test-only",
        langfuse_secret_key="obvious-fake-secret",
        langfuse_host="https://langfuse.internal.example",
    )

    assert settings.langfuse_host == "https://langfuse.internal.example"
    assert settings.langfuse_secret_key.get_secret_value() == "obvious-fake-secret"
    assert "obvious-fake-secret" not in repr(settings)


@pytest.mark.parametrize("timeout", [0, -1, 31])
def test_langfuse_timeout_is_bounded(timeout: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(langfuse_timeout_seconds=timeout)
