"""Application embedding-provider interfaces and implementations."""

from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.factory import make_embedding_provider
from src.services.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

__all__ = ["EmbeddingProvider", "SentenceTransformerEmbeddingProvider", "make_embedding_provider"]
