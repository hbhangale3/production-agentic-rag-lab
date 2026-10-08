from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from opensearchpy.exceptions import OpenSearchException
from src.exceptions import ChunkIndexError, EmbeddingDimensionError, EmbeddingEncodingError
from src.services.chunking.models import ChunkingResult, PaperChunk
from src.services.indexing.chunk_service import ChunkIndexingService
from src.services.opensearch.chunk_document_mapper import chunk_to_document
from src.services.opensearch.chunk_index import BulkWriteResult, ChunkIndexManager

DIMENSION = 384


def paper(arxiv_id: str = "2610.01963", raw_text: str = "canonical text") -> SimpleNamespace:
    return SimpleNamespace(arxiv_id=arxiv_id, raw_text=raw_text)


def chunks(arxiv_id: str = "2610.01963", count: int = 2) -> ChunkingResult:
    values = tuple(
        PaperChunk(
            chunk_id=f"{arxiv_id}::chunk::{index:03d}",
            arxiv_id=arxiv_id,
            chunk_index=index,
            section_title="Methods",
            text=f"chunk text {index}",
            word_count=3,
            has_overlap=index > 0,
            paper_title="Clinical AI",
            authors=("Ada Lovelace",),
            categories=("cs.AI", "cs.CL"),
            published_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
        for index in range(count)
    )
    return ChunkingResult(arxiv_id=arxiv_id, mode="sections" if values else "empty", chunks=values)


def vector(value: float = 0.1, dimension: int = DIMENSION) -> list[float]:
    return [value] * dimension


def service(
    *,
    chunk_result: ChunkingResult | None = None,
    embedding_provider: Mock | None = None,
    chunk_index: Mock | None = None,
) -> tuple[ChunkIndexingService, Mock, Mock, Mock]:
    chunker = Mock()
    chunker.chunk_paper.return_value = chunk_result or chunks()
    embedder = embedding_provider or Mock()
    if embedding_provider is None:
        embedder.embed_passages.return_value = [vector(), vector(0.2)]
    index = chunk_index or Mock()
    index.delete_paper_chunks.return_value = 2
    index.bulk_index_documents.return_value = BulkWriteResult(attempted=2, indexed=2, failed=0)
    return (
        ChunkIndexingService(
            chunker=chunker,
            embedding_provider=embedder,
            chunk_index=index,
            embedding_dimension=DIMENSION,
        ),
        chunker,
        embedder,
        index,
    )


def test_paper_chunks_are_embedded_serialized_and_bulk_indexed() -> None:
    instance, chunker, embedder, index = service()

    result = instance.replace_paper_chunks(paper())

    assert result.successful
    assert (result.chunks_generated, result.embeddings_generated, result.chunks_indexed) == (2, 2, 2)
    chunker.chunk_paper.assert_called_once()
    embedder.embed_passages.assert_called_once_with(["chunk text 0", "chunk text 1"])
    documents = index.bulk_index_documents.call_args.args[0]
    assert [document_id for document_id, _ in documents] == [
        "2610.01963::chunk::000",
        "2610.01963::chunk::001",
    ]
    assert documents[0][1]["chunk_text"] == "chunk text 0"
    assert documents[0][1]["paper_title"] == "Clinical AI"
    assert documents[0][1]["authors"] == ["Ada Lovelace"]
    assert documents[0][1]["categories"] == ["cs.AI", "cs.CL"]
    assert documents[0][1]["published_date"] == "2026-10-01T00:00:00+00:00"
    assert len(documents[0][1]["embedding"]) == DIMENSION


def test_mapper_rejects_wrong_vector_dimension() -> None:
    with pytest.raises(EmbeddingDimensionError, match="383.*expected 384"):
        chunk_to_document(chunks().chunks[0], vector(dimension=383), expected_dimension=DIMENSION)


def test_delete_occurs_only_after_successful_embedding() -> None:
    events = []
    embedder = Mock()
    embedder.embed_passages.side_effect = lambda _: events.append("embed") or [vector(), vector()]
    index = Mock()
    index.delete_paper_chunks.side_effect = lambda _: events.append("delete") or 1
    index.bulk_index_documents.side_effect = lambda _: events.append("bulk") or BulkWriteResult(2, 2, 0)
    instance, _, _, _ = service(embedding_provider=embedder, chunk_index=index)

    assert instance.replace_paper_chunks(paper()).successful
    assert events == ["embed", "delete", "bulk"]


def test_embedding_failure_preserves_existing_chunks() -> None:
    embedder = Mock()
    embedder.embed_passages.side_effect = EmbeddingEncodingError("inference failed")
    instance, _, _, index = service(embedding_provider=embedder)

    result = instance.replace_paper_chunks(paper())

    assert not result.successful
    assert "inference failed" in result.errors[0].message
    index.delete_paper_chunks.assert_not_called()
    index.bulk_index_documents.assert_not_called()


def test_wrong_vector_dimension_preserves_existing_chunks() -> None:
    embedder = Mock()
    embedder.embed_passages.return_value = [vector(), vector(dimension=12)]
    instance, _, _, index = service(embedding_provider=embedder)

    result = instance.replace_paper_chunks(paper())

    assert not result.successful
    index.delete_paper_chunks.assert_not_called()


def test_empty_paper_deletes_old_chunks_without_embedding_or_fake_documents() -> None:
    instance, _, embedder, index = service(chunk_result=chunks(count=0))

    result = instance.replace_paper_chunks(paper())

    assert result.successful
    assert result.chunks_generated == 0
    assert result.chunks_deleted == 2
    embedder.embed_passages.assert_not_called()
    index.delete_paper_chunks.assert_called_once_with("2610.01963")
    index.bulk_index_documents.assert_not_called()
    index.refresh_chunk_index.assert_called_once_with()


def test_missing_or_incompatible_index_fails_before_processing() -> None:
    instance, chunker, _, index = service()
    index.ensure_compatible_index.side_effect = ChunkIndexError("dimension mismatch")

    result = instance.replace_paper_chunks(paper())

    assert not result.successful
    chunker.chunk_paper.assert_not_called()
    index.delete_paper_chunks.assert_not_called()


def test_bulk_partial_failure_marks_paper_failed() -> None:
    instance, _, _, index = service()
    index.bulk_index_documents.return_value = BulkWriteResult(
        attempted=2,
        indexed=1,
        failed=1,
        errors=({"index": {"_id": "bad", "status": 400}},),
    )

    result = instance.replace_paper_chunks(paper())

    assert not result.successful
    assert (result.chunks_indexed, result.chunks_failed) == (1, 1)
    assert "Bulk indexing failed" in result.errors[0].message


@pytest.mark.parametrize("operation", ["delete", "bulk"])
def test_opensearch_write_failure_is_reported(operation: str) -> None:
    instance, _, _, index = service()
    if operation == "delete":
        index.delete_paper_chunks.side_effect = ChunkIndexError("delete failed")
    else:
        index.bulk_index_documents.side_effect = ChunkIndexError("bulk failed")

    result = instance.replace_paper_chunks(paper())

    assert not result.successful
    assert operation in result.errors[0].message


def test_backfill_continues_after_one_paper_failure_and_aggregates_counters() -> None:
    instance, _, embedder, _ = service()
    embedder.embed_passages.side_effect = [EmbeddingEncodingError("first failed"), [vector(), vector()]]

    result = instance.backfill_papers([paper("one"), paper("two")])

    assert result.papers_attempted == 2
    assert result.papers_indexed == 1
    assert result.papers_failed == 1
    assert result.chunks_generated == 4
    assert result.embeddings_generated == 2
    assert result.chunks_indexed == 2
    assert len(result.errors) == 1


class InMemoryChunkIndex:
    def __init__(self) -> None:
        self.documents = {
            "other::chunk::000": {"arxiv_id": "other"},
        }

    def ensure_compatible_index(self) -> None: ...

    def delete_paper_chunks(self, arxiv_id: str) -> int:
        ids = [key for key, value in self.documents.items() if value["arxiv_id"] == arxiv_id]
        for document_id in ids:
            del self.documents[document_id]
        return len(ids)

    def bulk_index_documents(self, documents):
        for document_id, document in documents:
            self.documents[document_id] = document
        return BulkWriteResult(len(documents), len(documents), 0)

    def refresh_chunk_index(self) -> None: ...


def test_replacement_removes_stale_tail_is_idempotent_and_preserves_other_papers() -> None:
    index = InMemoryChunkIndex()
    chunker = Mock()
    chunker.chunk_paper.side_effect = [chunks(count=3), chunks(count=2), chunks(count=2)]
    embedder = Mock()
    embedder.embed_passages.side_effect = [
        [vector()] * 3,
        [vector()] * 2,
        [vector()] * 2,
    ]
    instance = ChunkIndexingService(
        chunker=chunker,
        embedding_provider=embedder,
        chunk_index=index,
        embedding_dimension=DIMENSION,
    )

    assert instance.replace_paper_chunks(paper()).successful
    assert instance.replace_paper_chunks(paper()).successful
    count_after_second = len(index.documents)
    assert "2610.01963::chunk::002" not in index.documents
    assert "other::chunk::000" in index.documents

    assert instance.replace_paper_chunks(paper()).successful
    assert len(index.documents) == count_after_second


def test_manager_deletes_by_exact_arxiv_id_and_refreshes() -> None:
    raw_client = Mock()
    raw_client.delete_by_query.return_value = {"deleted": 4}
    instance = ChunkIndexManager(client=raw_client, index_name="chunks", embedding_dimension=DIMENSION)

    assert instance.delete_paper_chunks("2610.01963") == 4
    raw_client.delete_by_query.assert_called_once_with(
        index="chunks",
        body={"query": {"term": {"arxiv_id": "2610.01963"}}},
        conflicts="proceed",
        refresh=True,
    )


def test_manager_builds_bulk_actions_with_chunk_id_as_document_id() -> None:
    raw_client = Mock()
    instance = ChunkIndexManager(client=raw_client, index_name="chunks", embedding_dimension=DIMENSION)
    documents = [("paper::chunk::000", {"arxiv_id": "paper"})]

    with patch("src.services.opensearch.chunk_index.bulk", return_value=(1, [])) as bulk_helper:
        result = instance.bulk_index_documents(documents)

    assert result == BulkWriteResult(attempted=1, indexed=1, failed=0)
    assert bulk_helper.call_args.args == (
        raw_client,
        [
            {
                "_op_type": "index",
                "_index": "chunks",
                "_id": "paper::chunk::000",
                "_source": {"arxiv_id": "paper"},
            }
        ],
    )
    assert bulk_helper.call_args.kwargs == {"raise_on_error": False, "raise_on_exception": False}


def test_manager_wraps_delete_and_bulk_failures() -> None:
    raw_client = Mock()
    raw_client.delete_by_query.side_effect = OpenSearchException("delete unavailable")
    instance = ChunkIndexManager(client=raw_client, index_name="chunks", embedding_dimension=DIMENSION)

    with pytest.raises(ChunkIndexError, match="delete unavailable"):
        instance.delete_paper_chunks("paper")
    with patch("src.services.opensearch.chunk_index.bulk", side_effect=OpenSearchException("bulk unavailable")):
        with pytest.raises(ChunkIndexError, match="bulk unavailable"):
            instance.bulk_index_documents([("id", {})])
