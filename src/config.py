from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )


class Settings(DefaultSettings):
    """Application settings."""

    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "rag-api"

    # PostgreSQL configuration
    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    # OpenSearch configuration
    opensearch_host: str = "http://opensearch:9200"
    opensearch_index_name: str = "arxiv-papers"

    # Week 4 retrieval configuration
    embedding_provider: Literal["sentence_transformer"] = "sentence_transformer"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)
    vector_search_max_results: int = Field(default=100, gt=0, le=10_000)
    opensearch_chunk_index_name: str = "arxiv-papers-chunks"
    chunk_target_words: int = Field(default=600, gt=0)
    chunk_overlap_words: int = Field(default=100, ge=0)
    chunk_min_words: int = Field(default=100, gt=0)

    # arXiv API configuration
    arxiv_api_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_search_category: str = "cs.AI"
    arxiv_max_results: int = 10
    arxiv_rate_limit_delay: float = 3.0
    arxiv_timeout: float = 30.0
    arxiv_max_retries: int = 3
    arxiv_pdf_cache_dir: str = "data/arxiv_pdfs"

    # PDF parser configuration
    pdf_parser_max_file_size_mb: int = 50
    pdf_parser_max_pages: int = 100

    # Scheduled ingestion configuration
    arxiv_ingestion_profiles: str = "ai,healthcare_ai,health_equity_tech"
    arxiv_ingestion_batch_size: int = 1
    arxiv_ingestion_schedule: str = "0 3 * * *"
    arxiv_ingestion_process_pdfs: bool = True

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "Settings":
        """Validate relationships between the future chunking settings."""
        if self.chunk_overlap_words >= self.chunk_target_words:
            raise ValueError("chunk overlap must be smaller than the target chunk size")
        if self.chunk_min_words > self.chunk_target_words:
            raise ValueError("minimum chunk size must not exceed the target chunk size")
        return self


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
