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


# Week 6+: LLM exceptions (placeholders for Week 1)
class LLMException(Exception):
    """Base exception for LLM-related errors."""


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
