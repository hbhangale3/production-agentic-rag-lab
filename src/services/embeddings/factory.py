from src.config import Settings, get_settings
from src.exceptions import ConfigurationError
from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider


def make_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the configured application embedding provider without loading its model."""
    config = settings or get_settings()
    if config.embedding_provider == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(
            model_name=config.embedding_model,
            dimension=config.embedding_dimension,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
        )
    raise ConfigurationError(f"Unsupported embedding provider: {config.embedding_provider}")
