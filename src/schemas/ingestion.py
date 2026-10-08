from typing import Literal

from pydantic import BaseModel, Field


class IngestionError(BaseModel):
    """Public error information for one ingestion stage."""

    arxiv_id: str | None = None
    stage: Literal["fetch", "download", "parse", "store"]
    message: str


class IngestionResult(BaseModel):
    """Summary of one metadata ingestion batch."""

    papers_fetched: int = Field(default=0, ge=0)
    pdfs_downloaded: int = Field(default=0, ge=0)
    pdfs_parsed: int = Field(default=0, ge=0)
    papers_stored: int = Field(default=0, ge=0)
    stored_arxiv_ids: list[str] = Field(default_factory=list)
    processing_time: float = Field(default=0.0, ge=0)
    errors: list[IngestionError] = Field(default_factory=list)
