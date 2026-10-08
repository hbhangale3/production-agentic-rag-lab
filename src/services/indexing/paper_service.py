import logging
from collections.abc import Iterable

from pydantic import ValidationError
from src.models.paper import Paper
from src.schemas.indexing import PaperIndexingError, PaperIndexingResult
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.document_mapper import paper_to_document

logger = logging.getLogger(__name__)


class PaperIndexingService:
    """Transfer authoritative PostgreSQL paper records into OpenSearch."""

    def __init__(self, opensearch_client: OpenSearchClient) -> None:
        self.opensearch_client = opensearch_client

    def index_papers(self, papers: Iterable[Paper]) -> PaperIndexingResult:
        """Index papers independently and report per-document failures."""
        paper_batch = list(papers)
        result = PaperIndexingResult(attempted=len(paper_batch))
        if not self.opensearch_client.create_index():
            result.failed = result.attempted
            result.errors.append(PaperIndexingError(message="OpenSearch index is unavailable"))
            return result

        for paper in paper_batch:
            try:
                document = paper_to_document(paper)
            except (ValidationError, TypeError, ValueError) as exc:
                logger.warning("Could not map paper %s for indexing: %s", paper.arxiv_id, exc)
                result.failed += 1
                result.errors.append(PaperIndexingError(arxiv_id=paper.arxiv_id, message=str(exc)))
                continue

            if self.opensearch_client.index_document(paper.arxiv_id, document):
                result.indexed += 1
            else:
                result.failed += 1
                result.errors.append(
                    PaperIndexingError(arxiv_id=paper.arxiv_id, message="OpenSearch indexing failed")
                )

        return result
