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


class ChunkIndexingError(BaseModel):
    arxiv_id: str | None = None
    message: str


class PaperChunkIndexingResult(BaseModel):
    arxiv_id: str
    successful: bool = False
    chunks_generated: int = Field(default=0, ge=0)
    embeddings_generated: int = Field(default=0, ge=0)
    chunks_deleted: int = Field(default=0, ge=0)
    chunks_indexed: int = Field(default=0, ge=0)
    chunks_failed: int = Field(default=0, ge=0)
    chunking_seconds: float = Field(default=0.0, ge=0)
    embedding_seconds: float = Field(default=0.0, ge=0)
    indexing_seconds: float = Field(default=0.0, ge=0)
    errors: list[ChunkIndexingError] = Field(default_factory=list)


class ChunkBackfillResult(BaseModel):
    papers_attempted: int = Field(default=0, ge=0)
    papers_indexed: int = Field(default=0, ge=0)
    papers_failed: int = Field(default=0, ge=0)
    chunks_generated: int = Field(default=0, ge=0)
    embeddings_generated: int = Field(default=0, ge=0)
    chunks_indexed: int = Field(default=0, ge=0)
    chunks_failed: int = Field(default=0, ge=0)
    chunking_seconds: float = Field(default=0.0, ge=0)
    embedding_seconds: float = Field(default=0.0, ge=0)
    indexing_seconds: float = Field(default=0.0, ge=0)
    errors: list[ChunkIndexingError] = Field(default_factory=list)
