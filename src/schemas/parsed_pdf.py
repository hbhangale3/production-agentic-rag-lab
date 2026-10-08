from pydantic import BaseModel, Field


class ParsedPDFSection(BaseModel):
    """One section extracted from a parsed PDF document."""

    title: str = Field(..., description="Section heading")
    text: str = Field(..., description="Text belonging to the section")


class ParsedPDF(BaseModel):
    """Application-owned representation of a parsed PDF."""

    raw_text: str = Field(..., description="Structured text exported from the document")
    sections: list[ParsedPDFSection] = Field(default_factory=list)
    parser_used: str = Field(default="docling")
    page_count: int | None = Field(default=None, ge=0, description="Number of pages processed")
