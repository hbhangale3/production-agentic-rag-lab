from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class ArxivPaper(BaseModel):
    """Metadata returned for one paper by the arXiv API."""

    arxiv_id: str = Field(..., description="Version-independent arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="Author names in arXiv order")
    abstract: str = Field(..., description="Paper abstract")
    categories: list[str] = Field(..., description="arXiv category terms")
    published_date: datetime = Field(..., description="Initial publication date")
    updated_date: datetime | None = Field(default=None, description="Most recent update date")
    pdf_url: HttpUrl = Field(..., description="URL to the paper PDF")
