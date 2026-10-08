from datetime import datetime

from pydantic import BaseModel, Field


class PaperSearchHit(BaseModel):
    """Compact application representation of one OpenSearch hit."""

    arxiv_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    categories: list[str] = Field(default_factory=list)
    published_date: datetime | None = None
    updated_date: datetime | None = None
    page_count: int | None = None
    parser_used: str | None = None
    score: float | None = None


class PaperSearchResult(BaseModel):
    """Normalized response shared by every paper-search strategy."""

    query: str
    total: int = Field(default=0, ge=0)
    hits: list[PaperSearchHit] = Field(default_factory=list)
    took_ms: int | None = Field(default=None, ge=0)
    successful: bool = True
    error: str | None = None
