import logging
import time
from collections.abc import Callable
from datetime import date, datetime

from sqlalchemy.orm import Session
from src.repositories.paper import PaperRepository
from src.schemas.arxiv import ArxivPaper
from src.schemas.ingestion import IngestionError, IngestionResult
from src.schemas.paper import PaperUpsert
from src.schemas.parsed_pdf import ParsedPDF
from src.services.arxiv.client import ArxivClient, SortBy, SortOrder
from src.services.pdf_parser.docling_parser import DoclingPDFParser

logger = logging.getLogger(__name__)

RepositoryFactory = Callable[[Session], PaperRepository]


class MetadataFetcher:
    """Coordinate arXiv metadata, cached PDFs, parsing, and persistence."""

    def __init__(
        self,
        *,
        arxiv_client: ArxivClient,
        pdf_parser: DoclingPDFParser,
        repository_factory: RepositoryFactory = PaperRepository,
    ) -> None:
        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.repository_factory = repository_factory

    async def fetch_and_process_papers(
        self,
        *,
        max_results: int | None = None,
        category: str | None = None,
        search_query: str | None = None,
        sort_by: SortBy = "submittedDate",
        sort_order: SortOrder = "descending",
        from_date: date | datetime | None = None,
        to_date: date | datetime | None = None,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Session | None = None,
    ) -> IngestionResult:
        """Run one ingestion batch while isolating failures between papers."""
        if store_to_db and db_session is None:
            raise ValueError("db_session is required when store_to_db=True")

        started_at = time.perf_counter()
        result = IngestionResult()
        repository = self.repository_factory(db_session) if store_to_db and db_session is not None else None

        try:
            papers = await self.arxiv_client.fetch_papers(
                category=category,
                search_query=search_query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as exc:
            logger.exception("Failed to fetch arXiv metadata")
            result.errors.append(IngestionError(stage="fetch", message=self._error_message(exc)))
            result.processing_time = time.perf_counter() - started_at
            return result

        result.papers_fetched = len(papers)
        for paper in papers:
            parsed_pdf: ParsedPDF | None = None
            local_pdf_path: str | None = None

            if process_pdfs:
                try:
                    pdf_path = await self.arxiv_client.download_pdf(paper)
                    local_pdf_path = str(pdf_path)
                    result.pdfs_downloaded += 1
                except Exception as exc:
                    self._record_paper_error(result, paper, "download", exc)
                    continue

                try:
                    parsed_pdf = await self.pdf_parser.parse_pdf(pdf_path)
                    result.pdfs_parsed += 1
                except Exception as exc:
                    self._record_paper_error(result, paper, "parse", exc)
                    continue

            if repository is not None:
                try:
                    repository.upsert(self._make_paper_upsert(paper, local_pdf_path, parsed_pdf))
                    result.papers_stored += 1
                except Exception as exc:
                    db_session.rollback()
                    self._record_paper_error(result, paper, "store", exc)

        result.processing_time = time.perf_counter() - started_at
        return result

    @staticmethod
    def _make_paper_upsert(
        paper: ArxivPaper,
        local_pdf_path: str | None,
        parsed_pdf: ParsedPDF | None,
    ) -> PaperUpsert:
        values: dict[str, object] = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "published_date": paper.published_date,
            "pdf_url": str(paper.pdf_url),
        }
        if paper.updated_date is not None:
            values["updated_date"] = paper.updated_date
        if local_pdf_path is not None:
            values["local_pdf_path"] = local_pdf_path
        if parsed_pdf is not None:
            values.update(
                parser_used=parsed_pdf.parser_used,
                page_count=parsed_pdf.page_count,
                raw_text=parsed_pdf.raw_text,
            )
        return PaperUpsert.model_validate(values)

    @staticmethod
    def _record_paper_error(
        result: IngestionResult,
        paper: ArxivPaper,
        stage: str,
        exc: Exception,
    ) -> None:
        logger.exception("Failed to %s arXiv paper %s", stage, paper.arxiv_id)
        result.errors.append(
            IngestionError(
                arxiv_id=paper.arxiv_id,
                stage=stage,
                message=MetadataFetcher._error_message(exc),
            )
        )

    @staticmethod
    def _error_message(exc: Exception) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__
