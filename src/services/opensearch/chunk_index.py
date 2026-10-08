import logging
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException as OpenSearchLibraryError
from opensearchpy.helpers import bulk
from src.exceptions import ChunkIndexError
from src.services.opensearch.index_config import CHUNK_FIELD_TYPES, build_arxiv_chunks_mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BulkWriteResult:
    attempted: int
    indexed: int
    failed: int
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class ChunkIndexManager:
    """Safe lifecycle operations for the separate chunk/vector index."""

    def __init__(self, *, client: OpenSearch, index_name: str, embedding_dimension: int) -> None:
        if not index_name.strip():
            raise ValueError("index_name cannot be empty")
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be greater than zero")
        self._client = client
        self.index_name = index_name.strip()
        self.embedding_dimension = embedding_dimension

    def chunk_index_exists(self) -> bool:
        try:
            return bool(self._client.indices.exists(index=self.index_name))
        except OpenSearchLibraryError as exc:
            raise ChunkIndexError(f"Could not check chunk index {self.index_name!r}: {exc}") from exc

    def create_chunk_index(self) -> bool:
        """Ensure a compatible chunk index exists without destructive changes."""
        if self.chunk_index_exists():
            self.validate_chunk_index_mapping(self.get_chunk_index_mapping())
            logger.info("Chunk index %s already exists with a compatible mapping", self.index_name)
            return True
        try:
            self._client.indices.create(
                index=self.index_name,
                body=build_arxiv_chunks_mapping(self.embedding_dimension),
            )
        except OpenSearchLibraryError as exc:
            raise ChunkIndexError(f"Could not create chunk index {self.index_name!r}: {exc}") from exc
        logger.info("Created chunk index %s", self.index_name)
        return True

    def get_chunk_index_mapping(self) -> dict[str, Any]:
        try:
            return self._client.indices.get_mapping(index=self.index_name)
        except OpenSearchLibraryError as exc:
            raise ChunkIndexError(f"Could not read chunk index mapping for {self.index_name!r}: {exc}") from exc

    def get_chunk_index_stats(self) -> dict[str, int]:
        try:
            stats = self._client.indices.stats(index=self.index_name)
            totals = stats["_all"]["total"]
            return {
                "document_count": int(totals["docs"]["count"]),
                "size_in_bytes": int(totals["store"]["size_in_bytes"]),
            }
        except (OpenSearchLibraryError, KeyError, TypeError, ValueError) as exc:
            raise ChunkIndexError(f"Could not read chunk index stats for {self.index_name!r}: {exc}") from exc

    def ensure_compatible_index(self) -> None:
        if not self.chunk_index_exists():
            raise ChunkIndexError(f"Chunk index {self.index_name!r} does not exist")
        self.validate_chunk_index_mapping(self.get_chunk_index_mapping())

    def search_chunks(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a read-only search against the configured chunk index."""
        try:
            response = self._client.search(index=self.index_name, body=body)
        except OpenSearchLibraryError as exc:
            raise ChunkIndexError(f"Could not search chunk index {self.index_name!r}: {exc}") from exc
        if not isinstance(response, dict):
            raise ChunkIndexError(
                f"Chunk index {self.index_name!r} returned a malformed search response"
            )
        return response

    def delete_paper_chunks(self, arxiv_id: str) -> int:
        """Delete only chunks belonging to one exact arXiv ID."""
        try:
            response = self._client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"arxiv_id": arxiv_id}}},
                conflicts="proceed",
                refresh=True,
            )
            return int(response.get("deleted", 0))
        except (OpenSearchLibraryError, TypeError, ValueError) as exc:
            raise ChunkIndexError(f"Could not delete chunks for paper {arxiv_id!r}: {exc}") from exc

    def bulk_index_documents(self, documents: list[tuple[str, dict[str, Any]]]) -> BulkWriteResult:
        """Bulk-index chunk documents using their deterministic IDs."""
        if not documents:
            return BulkWriteResult(attempted=0, indexed=0, failed=0)
        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": document_id,
                "_source": document,
            }
            for document_id, document in documents
        ]
        try:
            indexed, errors = bulk(
                self._client,
                actions,
                raise_on_error=False,
                raise_on_exception=False,
            )
        except OpenSearchLibraryError as exc:
            raise ChunkIndexError(f"Could not bulk-index chunks into {self.index_name!r}: {exc}") from exc
        error_items = tuple(errors)
        return BulkWriteResult(
            attempted=len(actions),
            indexed=int(indexed),
            failed=len(actions) - int(indexed),
            errors=error_items,
        )

    def refresh_chunk_index(self) -> None:
        try:
            self._client.indices.refresh(index=self.index_name)
        except OpenSearchLibraryError as exc:
            raise ChunkIndexError(f"Could not refresh chunk index {self.index_name!r}: {exc}") from exc

    def validate_chunk_index_mapping(self, mapping: dict[str, Any]) -> None:
        """Reject an existing index whose core fields or vector dimension are incompatible."""
        try:
            properties = mapping[self.index_name]["mappings"]["properties"]
        except (KeyError, TypeError) as exc:
            raise ChunkIndexError(f"Chunk index {self.index_name!r} has an unreadable mapping") from exc

        for field, expected_type in CHUNK_FIELD_TYPES.items():
            actual_type = properties.get(field, {}).get("type")
            if actual_type != expected_type:
                raise ChunkIndexError(
                    f"Chunk index field {field!r} has type {actual_type!r}; expected {expected_type!r}"
                )
        actual_dimension = properties["embedding"].get("dimension")
        if actual_dimension != self.embedding_dimension:
            raise ChunkIndexError(
                f"Chunk index embedding dimension is {actual_dimension}; expected {self.embedding_dimension}"
            )

    def close(self) -> None:
        self._client.close()
