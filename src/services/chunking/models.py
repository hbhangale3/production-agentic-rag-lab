from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperChunk(BaseModel):
    """One stable retrieval unit derived from a canonical parsed paper."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    arxiv_id: str
    chunk_index: int = Field(ge=0)
    section_title: str | None = None
    text: str = Field(min_length=1)
    word_count: int = Field(gt=0)
    has_overlap: bool
    paper_title: str | None = None
    authors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    published_date: datetime | None = None


class ChunkingResult(BaseModel):
    """Chunks and provenance for one chunking operation."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    mode: Literal["sections", "raw_text", "empty"]
    chunks: tuple[PaperChunk, ...] = ()
