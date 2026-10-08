from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema
from src.config import Settings
from src.db.interfaces.postgresql import Base
from src.models.paper import Paper
from src.schemas.arxiv import ArxivPaper
from src.schemas.parsed_pdf import ParsedPDF
from src.services.metadata_fetcher.factory import make_metadata_fetcher


class FakeArxivClient:
    def __init__(self, paper: ArxivPaper) -> None:
        self.paper = paper

    async def fetch_papers(self, **_kwargs) -> list[ArxivPaper]:
        return [self.paper]

    async def download_pdf(self, _paper: ArxivPaper) -> Path:
        return Path("data/arxiv_pdfs/2610.10538.pdf")


class FakePDFParser:
    async def parse_pdf(self, _pdf_path: Path) -> ParsedPDF:
        return ParsedPDF(raw_text="Integration parsed text.", parser_used="docling", page_count=48)


@pytest.fixture
def session() -> Session:
    engine = create_engine(Settings().postgres_database_url)
    schema_name = f"test_fetcher_{uuid4().hex}"
    with engine.begin() as connection:
        connection.execute(CreateSchema(schema_name))
    test_engine = engine.execution_options(schema_translate_map={None: schema_name})
    Base.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    Base.metadata.drop_all(test_engine)
    with engine.begin() as connection:
        connection.execute(DropSchema(schema_name, cascade=True))
    test_engine.dispose()
    engine.dispose()


@pytest.mark.anyio
async def test_repeated_orchestration_is_idempotent_in_postgresql(session: Session) -> None:
    paper = ArxivPaper(
        arxiv_id="2610.10538",
        title="Never Look Back",
        authors=["Author One", "Author Two"],
        abstract="Abstract",
        categories=["cs.AI", "cs.RO"],
        published_date=datetime(2026, 10, 7, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/2610.10538v1",
    )
    fetcher = make_metadata_fetcher(FakeArxivClient(paper), FakePDFParser())

    first = await fetcher.fetch_and_process_papers(max_results=1, db_session=session)
    second = await fetcher.fetch_and_process_papers(max_results=1, db_session=session)
    stored = session.scalar(select(Paper).where(Paper.arxiv_id == paper.arxiv_id))
    count = session.scalar(select(func.count()).select_from(Paper).where(Paper.arxiv_id == paper.arxiv_id))

    assert first.papers_stored == second.papers_stored == 1
    assert count == 1
    assert stored is not None
    assert stored.local_pdf_path == "data/arxiv_pdfs/2610.10538.pdf"
    assert stored.parser_used == "docling"
    assert stored.page_count == 48
    assert stored.raw_text == "Integration parsed text."
