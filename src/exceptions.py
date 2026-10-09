"""Custom exceptions for the Mother-of-AI application."""


class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


class PDFParserError(ParsingException):
    """Exception raised when a PDF cannot be validated or parsed."""


# Week 3+: OpenSearch exceptions (placeholders for Week 1)
class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


class ChunkIndexError(OpenSearchException):
    """Exception raised when the chunk index cannot be managed safely."""


class VectorSearchError(Exception):
    """Exception raised when semantic vector retrieval cannot be completed safely."""


class ChunkBM25SearchError(Exception):
    """Exception raised when chunk-level lexical retrieval fails."""


class HybridSearchError(Exception):
    """Exception raised when neither hybrid retrieval path can produce results."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""


class LLMConfigurationError(LLMException):
    """Exception raised when an LLM provider cannot be configured safely."""


class LLMRequestError(LLMException):
    """Exception raised when an LLM provider request fails."""


class LLMResponseError(LLMException):
    """Exception raised when an LLM provider returns an unusable response."""


class RAGException(Exception):
    """Base exception for grounded generation failures outside the provider."""


class InsufficientEvidenceError(RAGException):
    """Exception raised when retrieval yields no evidence suitable for generation."""


class RAGPromptBudgetError(RAGException):
    """Exception raised when reserved generation capacity cannot be respected."""


# General application exceptions
class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""


class EmbeddingError(Exception):
    """Base exception for embedding-provider failures."""


class EmbeddingModelLoadError(EmbeddingError):
    """Exception raised when an embedding model cannot be loaded."""


class EmbeddingEncodingError(EmbeddingError):
    """Exception raised when text cannot be encoded."""


class EmbeddingDimensionError(EmbeddingError):
    """Exception raised when an embedding has an incompatible dimension."""


class ArxivClientError(Exception):
    """Exception raised when an arXiv API request or response cannot be handled."""


class ArxivPDFDownloadError(ArxivClientError):
    """Exception raised when an arXiv PDF cannot be downloaded or validated."""
