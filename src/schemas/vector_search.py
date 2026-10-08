from datetime import datetime

from pydantic import BaseModel, Field


class ChunkSearchHit(BaseModel):
    """Stable application representation of one semantic chunk hit."""

    chunk_id: str
    arxiv_id: str
    chunk_index: int = Field(ge=0)
    section_title: str | None = None
    chunk_text: str
    word_count: int = Field(ge=0)
    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_date: datetime | None = None
    score: float | None = None


class VectorSearchResult(BaseModel):
    """Normalized semantic retrieval result without stored embeddings."""

    query: str
    total: int = Field(default=0, ge=0)
    hits: list[ChunkSearchHit] = Field(default_factory=list)
    took_ms: int | None = Field(default=None, ge=0)
