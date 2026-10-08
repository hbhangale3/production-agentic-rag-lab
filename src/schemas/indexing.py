from pydantic import BaseModel, Field


class PaperIndexingError(BaseModel):
    """Failure information for one paper or a batch-level index operation."""

    arxiv_id: str | None = None
    message: str


class PaperIndexingResult(BaseModel):
    """Summary of transferring paper records into OpenSearch."""

    attempted: int = Field(default=0, ge=0)
    indexed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: list[PaperIndexingError] = Field(default_factory=list)
