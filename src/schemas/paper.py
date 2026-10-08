from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PaperBase(BaseModel):
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: list[str] = Field(..., description="Paper categories")
    published_date: datetime = Field(..., description="Date published on arXiv")
    updated_date: datetime | None = Field(default=None, description="Most recent update date on arXiv")
    pdf_url: str = Field(..., description="URL to PDF")
    local_pdf_path: str | None = Field(default=None, description="Path to the cached PDF")
    parser_used: str | None = Field(default=None, description="Parser used for extracted content")
    page_count: int | None = Field(default=None, ge=0, description="Number of parsed PDF pages")
    raw_text: str | None = Field(default=None, description="Text extracted from the PDF")


class PaperCreate(PaperBase):
    pass


class PaperUpsert(PaperBase):
    """Paper ingestion data used for insert-or-update operations."""


class PaperResponse(PaperBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime | None


class PaperSearchResponse(BaseModel):
    papers: list[PaperResponse]
    total: int
