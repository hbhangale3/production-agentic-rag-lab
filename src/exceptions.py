"""Custom exceptions for the Mother-of-AI application."""

from enum import StrEnum


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


class PDFNoTextError(PDFParserError):
    """Exception raised when a PDF converts successfully but yields no extractable text."""


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


class GroundingValidationReason(StrEnum):
    """Stable, content-free reasons a generated answer can fail structural validation."""

    NON_TEXT_ANSWER = "non_text_answer"
    EMPTY_ANSWER = "empty_answer"
    MALFORMED_CITATION = "malformed_citation"
    UNSUPPORTED_CITATION_LABEL = "unsupported_citation_label"
    MISSING_REQUIRED_CITATION = "missing_required_citation"


_SAFE_GROUNDING_VALIDATION_MESSAGES = {
    GroundingValidationReason.NON_TEXT_ANSWER: "Generated answer was not text",
    GroundingValidationReason.EMPTY_ANSWER: "Generated answer was empty",
    GroundingValidationReason.MALFORMED_CITATION: "Generated answer contained malformed citation syntax",
    GroundingValidationReason.UNSUPPORTED_CITATION_LABEL: "Generated answer contained an unsupported citation label",
    GroundingValidationReason.MISSING_REQUIRED_CITATION: "Substantive generated answer did not cite supplied evidence",
}


class GroundingValidationError(LLMResponseError):
    """A generated answer violated the structural grounding contract."""

    def __init__(self, message: str, *, reason: GroundingValidationReason | None = None) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def safe_message(self) -> str | None:
        """Return an allowlisted diagnostic, never the caller-provided exception message."""

        return _SAFE_GROUNDING_VALIDATION_MESSAGES.get(self.reason)


class IncompleteGenerationError(LLMResponseError):
    """The provider stopped at the completion-token limit; the text is partial and must not be used."""


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
