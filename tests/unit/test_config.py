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
