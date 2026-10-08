from typing import Any

from src.exceptions import EmbeddingDimensionError
from src.services.chunking.models import PaperChunk


def chunk_to_document(chunk: PaperChunk, embedding: list[float], *, expected_dimension: int) -> dict[str, Any]:
    """Serialize one chunk and its validated vector for OpenSearch."""
    if len(embedding) != expected_dimension:
        raise EmbeddingDimensionError(
            f"Chunk {chunk.chunk_id!r} embedding has dimension {len(embedding)}; expected {expected_dimension}"
        )
    return {
        "chunk_id": chunk.chunk_id,
        "arxiv_id": chunk.arxiv_id,
        "chunk_index": chunk.chunk_index,
        "section_title": chunk.section_title,
        "chunk_text": chunk.text,
        "word_count": chunk.word_count,
        "has_overlap": chunk.has_overlap,
        "paper_title": chunk.paper_title,
        "authors": list(chunk.authors),
        "categories": list(chunk.categories),
        "published_date": chunk.published_date.isoformat() if chunk.published_date else None,
        "embedding": embedding,
    }
