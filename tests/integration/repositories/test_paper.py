from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema
from src.config import Settings
from src.db.interfaces.postgresql import Base
from src.models.paper import Paper
from src.repositories.paper import PaperRepository
from src.schemas.paper import PaperUpsert

TEST_ARXIV_ID = "test.2610.10538"


@pytest.fixture
def session() -> Session:
    engine = create_engine(Settings().postgres_database_url)
    schema_name = f"test_paper_{uuid4().hex}"
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


@pytest.fixture
def paper_data() -> PaperUpsert:
    return PaperUpsert(
        arxiv_id=TEST_ARXIV_ID,
        title="Initial title",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="Initial abstract",
        categories=["cs.AI", "cs.LG"],
        published_date=datetime(2026, 10, 7, 17, 59, tzinfo=timezone.utc),
        updated_date=datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/2610.10538v1",
        local_pdf_path="data/arxiv_pdfs/2610.10538.pdf",
        parser_used="docling",
        page_count=48,
        raw_text="Parsed paper text.",
    )


def test_upsert_inserts_and_retrieves_complete_paper(session: Session, paper_data: PaperUpsert) -> None:
    repository = PaperRepository(session)

    inserted = repository.upsert(paper_data)
    retrieved = repository.get_by_arxiv_id(TEST_ARXIV_ID)

    assert retrieved is not None
    assert retrieved.id == inserted.id
    assert retrieved.authors == ["Ada Lovelace", "Alan Turing"]
    assert retrieved.categories == ["cs.AI", "cs.LG"]
    assert retrieved.updated_date == datetime(2026, 10, 8, 12, 0)
    assert retrieved.local_pdf_path == "data/arxiv_pdfs/2610.10538.pdf"
    assert retrieved.parser_used == "docling"
    assert retrieved.page_count == 48
    assert retrieved.raw_text == "Parsed paper text."


def test_upsert_updates_without_creating_duplicate(session: Session, paper_data: PaperUpsert) -> None:
    repository = PaperRepository(session)
    original = repository.upsert(paper_data)
    updated_data = paper_data.model_copy(
        update={
            "title": "Updated title",
            "authors": ["Updated Author"],
            "categories": ["cs.AI"],
            "raw_text": "Updated parsed text.",
        }
    )

    updated = repository.upsert(updated_data)
    row_count = session.scalar(select(func.count()).select_from(Paper).where(Paper.arxiv_id == TEST_ARXIV_ID))

    assert updated.id == original.id
    assert updated.title == "Updated title"
    assert updated.authors == ["Updated Author"]
    assert updated.categories == ["cs.AI"]
    assert updated.raw_text == "Updated parsed text."
    assert row_count == 1


def test_get_by_arxiv_id_returns_none_for_missing_paper(session: Session) -> None:
    repository = PaperRepository(session)

    assert repository.get_by_arxiv_id("test.missing") is None
