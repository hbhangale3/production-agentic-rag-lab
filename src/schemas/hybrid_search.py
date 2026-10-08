from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, pattern=r".*\S.*")
    size: int = Field(default=5, ge=1, le=100)
    categories: list[str] | None = None
    published_from: date | None = None
    published_to: date | None = None

    @model_validator(mode="after")
    def validate_filters(self) -> "HybridSearchRequest":
        if self.categories is not None and (
            not self.categories or any(not value.strip() for value in self.categories)
        ):
            raise ValueError("categories must contain non-blank values")
        if self.published_from and self.published_to and self.published_from > self.published_to:
            raise ValueError("published_from must not be after published_to")
        return self


class HybridSearchHit(BaseModel):
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
    rrf_score: float = Field(gt=0)
    bm25_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    vector_score: float | None = None


class HybridSearchResult(BaseModel):
    query: str
    retrieval_mode: Literal["hybrid", "bm25_fallback", "vector_fallback"]
    count: int = Field(ge=0)
    results: list[HybridSearchHit] = Field(default_factory=list)
    degradation_reason: str | None = None
