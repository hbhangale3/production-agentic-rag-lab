from src.config import Settings, get_settings
from src.services.arxiv.client import ArxivClient


def make_arxiv_client(settings: Settings | None = None) -> ArxivClient:
    """Create an arXiv client from application settings."""
    config = settings or get_settings()
    return ArxivClient(
        base_url=config.arxiv_api_base_url,
        search_category=config.arxiv_search_category,
        max_results=config.arxiv_max_results,
        rate_limit_delay=config.arxiv_rate_limit_delay,
        timeout=config.arxiv_timeout,
        max_retries=config.arxiv_max_retries,
        pdf_cache_dir=config.arxiv_pdf_cache_dir,
    )
