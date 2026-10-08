from collections.abc import Callable
from threading import Lock
from typing import Any

from src.exceptions import EmbeddingDimensionError, EmbeddingEncodingError, EmbeddingModelLoadError

ModelFactory = Callable[..., Any]


class SentenceTransformerEmbeddingProvider:
    """Lazy, reusable local SentenceTransformers embedding provider."""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        device: str,
        batch_size: int = 32,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be greater than zero")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self.batch_size = batch_size
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any | None = None
        self._load_lock = Lock()
        self._model_load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_load_count(self) -> int:
        return self._model_load_count

    def initialize(self) -> None:
        """Explicitly load the model; normal embedding calls also load it lazily."""
        self._get_model()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingEncodingError("passage text must be a non-empty string")
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingEncodingError("query text must be a non-empty string")
        return self._encode([text])[0]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                self._model = self._model_factory(self.model_name, device=self.device)
            except Exception as exc:
                raise EmbeddingModelLoadError(
                    f"Could not load embedding model {self.model_name!r} on {self.device!r}: {exc}"
                ) from exc
            self._model_load_count += 1
            return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        try:
            encoded = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
            vectors = [[float(value) for value in vector] for vector in raw_vectors]
        except Exception as exc:
            raise EmbeddingEncodingError(f"Embedding model failed to encode text: {exc}") from exc

        if len(vectors) != len(texts):
            raise EmbeddingEncodingError(
                f"Embedding model returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for position, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    f"Embedding {position} has dimension {len(vector)}; expected {self.dimension}"
                )
        return vectors

    @staticmethod
    def _default_model_factory(model_name: str, *, device: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)
