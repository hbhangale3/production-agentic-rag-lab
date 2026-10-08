from datetime import datetime, timezone

from src.models.paper import Paper
from src.services.opensearch.document_mapper import paper_to_document


def make_paper() -> Paper:
    return Paper(
        arxiv_id="2610.01963",
        title="Healthcare AI Paper",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="Clinical artificial intelligence research.",
        categories=["cs.AI", "cs.LG"],
        published_date=datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc),
        updated_date=datetime(2026, 10, 3, 13, 45, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/2610.01963v1",
        local_pdf_path="data/arxiv_pdfs/2610.01963.pdf",
        parser_used="docling",
        page_count=9,
        raw_text="Complete parsed paper text.",
    )


def test_paper_to_document_preserves_searchable_values_and_json_types() -> None:
    paper = make_paper()

    document = paper_to_document(paper)

    assert document["arxiv_id"] == paper.arxiv_id
    assert document["title"] == paper.title
    assert document["abstract"] == paper.abstract
    assert document["raw_text"] == paper.raw_text
    assert document["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert document["categories"] == ["cs.AI", "cs.LG"]
    assert document["published_date"] == "2026-10-02T12:30:00Z"
    assert document["updated_date"] == "2026-10-03T13:45:00Z"
    assert "_sa_instance_state" not in document
    assert "id" not in document


def test_paper_to_document_preserves_nullable_parsed_fields() -> None:
    paper = make_paper()
    paper.updated_date = None
    paper.local_pdf_path = None
    paper.parser_used = None
    paper.page_count = None
    paper.raw_text = None

    document = paper_to_document(paper)

    assert document["updated_date"] is None
    assert document["local_pdf_path"] is None
    assert document["parser_used"] is None
    assert document["page_count"] is None
    assert document["raw_text"] is None
