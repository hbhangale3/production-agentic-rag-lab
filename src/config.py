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
    opensearch_host: str = "http://localhost:9200"

    # arXiv API configuration
    arxiv_api_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_search_category: str = "cs.AI"
    arxiv_max_results: int = 10
    arxiv_rate_limit_delay: float = 3.0
    arxiv_timeout: float = 30.0
    arxiv_max_retries: int = 3


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
