from src.services.arxiv.client import ArxivClient
from src.services.metadata_fetcher.service import MetadataFetcher, RepositoryFactory
from src.services.pdf_parser.docling_parser import DoclingPDFParser


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: DoclingPDFParser,
    *,
    repository_factory: RepositoryFactory | None = None,
) -> MetadataFetcher:
    """Create an ingestion orchestrator from explicit service dependencies."""
    kwargs = {"repository_factory": repository_factory} if repository_factory is not None else {}
    return MetadataFetcher(arxiv_client=arxiv_client, pdf_parser=pdf_parser, **kwargs)
