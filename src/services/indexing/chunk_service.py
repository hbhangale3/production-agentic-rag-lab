import logging
from collections.abc import Iterable
from time import perf_counter

from src.exceptions import ChunkIndexError, EmbeddingError
from src.models.paper import Paper
from src.schemas.indexing import (
    ChunkBackfillResult,
    ChunkIndexingError,
    PaperChunkIndexingResult,
)
from src.services.chunking.service import PaperChunkingService
from src.services.embeddings.base import EmbeddingProvider
from src.services.opensearch.chunk_document_mapper import chunk_to_document
from src.services.opensearch.chunk_index import ChunkIndexManager

logger = logging.getLogger(__name__)


class ChunkIndexingService:
    """Coordinate canonical paper chunking, embedding, and replacement indexing."""

    def __init__(
        self,
        *,
        chunker: PaperChunkingService,
        embedding_provider: EmbeddingProvider,
        chunk_index: ChunkIndexManager,
        embedding_dimension: int,
    ) -> None:
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.chunk_index = chunk_index
        self.embedding_dimension = embedding_dimension

    def replace_paper_chunks(self, paper: Paper, *, validate_index: bool = True) -> PaperChunkIndexingResult:
        arxiv_id = str(paper.arxiv_id)
        result = PaperChunkIndexingResult(arxiv_id=arxiv_id)
        try:
            if validate_index:
                self.chunk_index.ensure_compatible_index()

            started = perf_counter()
            chunks = list(self.chunker.chunk_paper(paper).chunks)
            result.chunking_seconds = perf_counter() - started
            result.chunks_generated = len(chunks)

            documents: list[tuple[str, dict]] = []
            if chunks:
                started = perf_counter()
                embeddings = self.embedding_provider.embed_passages([chunk.text for chunk in chunks])
                result.embedding_seconds = perf_counter() - started
                result.embeddings_generated = len(embeddings)
                if len(embeddings) != len(chunks):
                    raise EmbeddingError(
                        f"Embedding provider returned {len(embeddings)} vectors for {len(chunks)} chunks"
                    )
                documents = [
                    (
                        chunk.chunk_id,
                        chunk_to_document(chunk, embedding, expected_dimension=self.embedding_dimension),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ]

            started = perf_counter()
            result.chunks_deleted = self.chunk_index.delete_paper_chunks(arxiv_id)
            if documents:
                bulk_result = self.chunk_index.bulk_index_documents(documents)
                result.chunks_indexed = bulk_result.indexed
                result.chunks_failed = bulk_result.failed
                if bulk_result.failed:
                    result.errors.append(
                        ChunkIndexingError(
                            arxiv_id=arxiv_id,
                            message=f"Bulk indexing failed for {bulk_result.failed} chunk(s): {bulk_result.errors}",
                        )
                    )
            self.chunk_index.refresh_chunk_index()
            result.indexing_seconds = perf_counter() - started
            result.successful = result.chunks_failed == 0
        except (ChunkIndexError, EmbeddingError, TypeError, ValueError) as exc:
            logger.warning("Could not replace chunks for paper %s: %s", arxiv_id, exc)
            result.errors.append(ChunkIndexingError(arxiv_id=arxiv_id, message=str(exc)))
        return result

    def backfill_papers(self, papers: Iterable[Paper]) -> ChunkBackfillResult:
        paper_batch = list(papers)
        aggregate = ChunkBackfillResult(papers_attempted=len(paper_batch))
        try:
            self.chunk_index.ensure_compatible_index()
        except ChunkIndexError as exc:
            aggregate.papers_failed = aggregate.papers_attempted
            aggregate.errors.append(ChunkIndexingError(message=str(exc)))
            return aggregate

        for paper in paper_batch:
            result = self.replace_paper_chunks(paper, validate_index=False)
            aggregate.papers_indexed += int(result.successful)
            aggregate.papers_failed += int(not result.successful)
            aggregate.chunks_generated += result.chunks_generated
            aggregate.embeddings_generated += result.embeddings_generated
            aggregate.chunks_indexed += result.chunks_indexed
            aggregate.chunks_failed += result.chunks_failed
            aggregate.chunking_seconds += result.chunking_seconds
            aggregate.embedding_seconds += result.embedding_seconds
            aggregate.indexing_seconds += result.indexing_seconds
            aggregate.errors.extend(result.errors)
        return aggregate
