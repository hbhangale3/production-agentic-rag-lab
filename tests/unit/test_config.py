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
    assert settings.llm_max_completion_tokens == 512
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
