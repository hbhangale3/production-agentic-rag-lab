from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest
from src.schemas.arxiv import ArxivPaper
from src.schemas.parsed_pdf import ParsedPDF, ParsedPDFSection
from src.services.metadata_fetcher.factory import make_metadata_fetcher


def make_paper(arxiv_id: str) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=["Author One", "Author Two"],
        abstract="An abstract.",
        categories=["cs.AI", "cs.LG"],
        published_date=datetime(2026, 10, 7, tzinfo=timezone.utc),
        updated_date=datetime(2026, 10, 8, tzinfo=timezone.utc),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v1",
    )


def parsed_pdf() -> ParsedPDF:
    return ParsedPDF(
        raw_text="Parsed document text.",
        sections=[ParsedPDFSection(title="Introduction", text="Section text.")],
        parser_used="docling",
        page_count=12,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.upserts = []

    def upsert(self, paper_data):
        self.upserts.append(paper_data)
        return paper_data


@pytest.mark.anyio
async def test_metadata_only_ingestion_forwards_query_and_skips_pdf_services() -> None:
    papers = [make_paper("2601.00001"), make_paper("2601.00002")]
    arxiv_client = Mock()
    arxiv_client.fetch_papers = AsyncMock(return_value=papers)
    arxiv_client.download_pdf = AsyncMock()
    pdf_parser = Mock()
    pdf_parser.parse_pdf = AsyncMock()
    repository = FakeRepository()
    session = Mock()
    fetcher = make_metadata_fetcher(
        arxiv_client,
        pdf_parser,
        repository_factory=lambda _: repository,
    )

    result = await fetcher.fetch_and_process_papers(
        max_results=2,
        category="q-bio.QM",
        sort_order="ascending",
        from_date=date(2026, 1, 1),
        to_date=date(2026, 1, 31),
        process_pdfs=False,
        db_session=session,
    )

    arxiv_client.fetch_papers.assert_awaited_once_with(
        category="q-bio.QM",
        search_query=None,
        max_results=2,
        sort_by="submittedDate",
        sort_order="ascending",
        from_date=date(2026, 1, 1),
        to_date=date(2026, 1, 31),
    )
    arxiv_client.download_pdf.assert_not_awaited()
    pdf_parser.parse_pdf.assert_not_awaited()
    assert (result.papers_fetched, result.pdfs_downloaded, result.pdfs_parsed, result.papers_stored) == (2, 0, 0, 2)
    assert len(repository.upserts) == 2
    assert repository.upserts[0].local_pdf_path is None
    assert repository.upserts[0].raw_text is None


@pytest.mark.anyio
async def test_advanced_search_query_is_forwarded_without_domain_policy() -> None:
    arxiv_client = Mock()
    arxiv_client.fetch_papers = AsyncMock(return_value=[])
    fetcher = make_metadata_fetcher(arxiv_client, Mock(), repository_factory=lambda _: FakeRepository())

    await fetcher.fetch_and_process_papers(
        search_query='all:"health equity" AND all:"machine learning"',
        sort_by="relevance",
        process_pdfs=False,
        store_to_db=False,
    )

    arxiv_client.fetch_papers.assert_awaited_once_with(
        category=None,
        search_query='all:"health equity" AND all:"machine learning"',
        max_results=None,
        sort_by="relevance",
        sort_order="descending",
        from_date=None,
        to_date=None,
    )


@pytest.mark.anyio
async def test_full_ingestion_maps_parsed_fields() -> None:
    paper = make_paper("2601.00001")
    arxiv_client = Mock()
    arxiv_client.fetch_papers = AsyncMock(return_value=[paper])
    arxiv_client.download_pdf = AsyncMock(return_value=Path("data/arxiv_pdfs/2601.00001.pdf"))
    pdf_parser = Mock()
    pdf_parser.parse_pdf = AsyncMock(return_value=parsed_pdf())
    repository = FakeRepository()
    fetcher = make_metadata_fetcher(arxiv_client, pdf_parser, repository_factory=lambda _: repository)

    result = await fetcher.fetch_and_process_papers(db_session=Mock())

    stored = repository.upserts[0]
    assert (result.papers_fetched, result.pdfs_downloaded, result.pdfs_parsed, result.papers_stored) == (1, 1, 1, 1)
    assert stored.local_pdf_path == "data/arxiv_pdfs/2601.00001.pdf"
    assert stored.parser_used == "docling"
    assert stored.page_count == 12
    assert stored.raw_text == "Parsed document text."


@pytest.mark.anyio
async def test_store_to_db_false_processes_without_repository() -> None:
    paper = make_paper("2601.00001")
    arxiv_client = Mock()
    arxiv_client.fetch_papers = AsyncMock(return_value=[paper])
    arxiv_client.download_pdf = AsyncMock(return_value=Path("cached.pdf"))
    pdf_parser = Mock()
    pdf_parser.parse_pdf = AsyncMock(return_value=parsed_pdf())
    repository_factory = Mock()
    fetcher = make_metadata_fetcher(arxiv_client, pdf_parser, repository_factory=repository_factory)

    result = await fetcher.fetch_and_process_papers(store_to_db=False)

    repository_factory.assert_not_called()
    assert (result.papers_fetched, result.pdfs_downloaded, result.pdfs_parsed, result.papers_stored) == (1, 1, 1, 0)
    assert result.errors == []


@pytest.mark.anyio
async def test_one_paper_failure_does_not_abort_batch() -> None:
    papers = [make_paper("2601.00001"), make_paper("2601.00002"), make_paper("2601.00003")]
    arxiv_client = Mock()
    arxiv_client.fetch_papers = AsyncMock(return_value=papers)
    arxiv_client.download_pdf = AsyncMock(side_effect=[Path("one.pdf"), RuntimeError("download unavailable"), Path("three.pdf")])
    pdf_parser = Mock()
    pdf_parser.parse_pdf = AsyncMock(return_value=parsed_pdf())
    repository = FakeRepository()
    fetcher = make_metadata_fetcher(arxiv_client, pdf_parser, repository_factory=lambda _: repository)

    result = await fetcher.fetch_and_process_papers(db_session=Mock())

    assert (result.papers_fetched, result.pdfs_downloaded, result.pdfs_parsed, result.papers_stored) == (3, 2, 2, 2)
    assert pdf_parser.parse_pdf.await_args_list == [call(Path("one.pdf")), call(Path("three.pdf"))]
    assert len(result.errors) == 1
    assert result.errors[0].arxiv_id == "2601.00002"
    assert result.errors[0].stage == "download"
    assert "download unavailable" in result.errors[0].message
    assert result.processing_time >= 0


@pytest.mark.anyio
async def test_store_requires_explicit_database_session() -> None:
    fetcher = make_metadata_fetcher(Mock(), Mock())

    with pytest.raises(ValueError, match="db_session is required"):
        await fetcher.fetch_and_process_papers()
