from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from src.exceptions import ConfigurationError, EmbeddingDimensionError, EmbeddingEncodingError, EmbeddingModelLoadError
from src.services.embeddings.factory import make_embedding_provider
from src.services.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider


def vector(value: float = 1.0, dimension: int = 384) -> list[float]:
    return [value] * dimension


def provider(model: Mock, *, dimension: int = 384) -> tuple[SentenceTransformerEmbeddingProvider, Mock]:
    factory = Mock(return_value=model)
    instance = SentenceTransformerEmbeddingProvider(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=dimension,
        device="cpu",
        batch_size=16,
        model_factory=factory,
    )
    return instance, factory


def test_model_is_loaded_lazily_and_reused() -> None:
    model = Mock()
    model.encode.side_effect = [np.array([vector()]), np.array([vector(2.0)])]
    instance, factory = provider(model)

    assert not instance.is_loaded
    instance.embed_query("first query")
    instance.embed_query("second query")

    factory.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cpu")
    assert instance.is_loaded
    assert instance.model_load_count == 1


def test_explicit_initialization_loads_model_once() -> None:
    instance, factory = provider(Mock())

    instance.initialize()
    instance.initialize()

    factory.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cpu")
    assert instance.model_load_count == 1


def test_passage_embeddings_preserve_input_order_and_batch_options() -> None:
    model = Mock()
    model.encode.return_value = np.array([vector(1.0), vector(2.0)])
    instance, _ = provider(model)

    result = instance.embed_passages(["passage one", "passage two"])

    assert result == [vector(1.0), vector(2.0)]
    model.encode.assert_called_once_with(
        ["passage one", "passage two"],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def test_query_embedding_returns_one_vector() -> None:
    model = Mock()
    model.encode.return_value = np.array([vector(0.25)])
    instance, _ = provider(model)

    assert instance.embed_query("clinical AI") == vector(0.25)


def test_empty_passage_list_does_not_load_model() -> None:
    instance, factory = provider(Mock())

    assert instance.embed_passages([]) == []
    factory.assert_not_called()


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected_without_dummy_fallback(text: str) -> None:
    instance, factory = provider(Mock())

    with pytest.raises(EmbeddingEncodingError, match="non-empty"):
        instance.embed_query(text)
    factory.assert_not_called()


def test_wrong_dimension_is_rejected() -> None:
    model = Mock()
    model.encode.return_value = np.array([vector(dimension=383)])
    instance, _ = provider(model)

    with pytest.raises(EmbeddingDimensionError, match="383; expected 384"):
        instance.embed_query("query")


def test_model_loading_failure_is_clear_and_has_no_fallback() -> None:
    factory = Mock(side_effect=OSError("model unavailable"))
    instance = SentenceTransformerEmbeddingProvider(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=384,
        device="cpu",
        model_factory=factory,
    )

    with pytest.raises(EmbeddingModelLoadError, match="model unavailable"):
        instance.embed_query("query")
    assert instance.model_load_count == 0


def test_encoding_failure_is_clear_and_has_no_fallback() -> None:
    model = Mock()
    model.encode.side_effect = RuntimeError("inference failed")
    instance, _ = provider(model)

    with pytest.raises(EmbeddingEncodingError, match="inference failed"):
        instance.embed_passages(["passage"])


def test_factory_returns_lazy_sentence_transformer_provider() -> None:
    settings = SimpleNamespace(
        embedding_provider="sentence_transformer",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        embedding_device="cpu",
        embedding_batch_size=32,
    )

    result = make_embedding_provider(settings)

    assert isinstance(result, SentenceTransformerEmbeddingProvider)
    assert not result.is_loaded
    assert result.batch_size == 32


def test_factory_rejects_unsupported_provider() -> None:
    settings = SimpleNamespace(embedding_provider="jina")

    with pytest.raises(ConfigurationError, match="Unsupported embedding provider: jina"):
        make_embedding_provider(settings)
