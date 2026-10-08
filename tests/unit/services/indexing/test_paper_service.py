from datetime import datetime, timezone
from unittest.mock import Mock

from src.models.paper import Paper
from src.services.indexing.paper_service import PaperIndexingService


def make_paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=["Author"],
        abstract="Abstract",
        categories=["cs.AI"],
        published_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def test_multiple_papers_are_indexed_with_success_counts() -> None:
    client = Mock()
    client.create_index.return_value = True
    client.index_document.return_value = True
    papers = [make_paper("2610.01963"), make_paper("2610.10393")]

    result = PaperIndexingService(client).index_papers(papers)

    assert (result.attempted, result.indexed, result.failed) == (2, 2, 0)
    assert [index_call.args[0] for index_call in client.index_document.call_args_list] == [
        "2610.01963",
        "2610.10393",
    ]
    assert result.errors == []


def test_failed_paper_is_isolated_and_identified() -> None:
    client = Mock()
    client.create_index.return_value = True
    client.index_document.side_effect = [False, True]
    papers = [make_paper("2610.01963"), make_paper("2610.10393")]

    result = PaperIndexingService(client).index_papers(papers)

    assert (result.attempted, result.indexed, result.failed) == (2, 1, 1)
    assert result.errors[0].arxiv_id == "2610.01963"
    assert "failed" in result.errors[0].message.lower()


def test_unavailable_index_fails_batch_without_document_writes() -> None:
    client = Mock()
    client.create_index.return_value = False

    result = PaperIndexingService(client).index_papers([make_paper("2610.01963")])

    assert (result.attempted, result.indexed, result.failed) == (1, 0, 1)
    assert result.errors[0].arxiv_id is None
    client.index_document.assert_not_called()
