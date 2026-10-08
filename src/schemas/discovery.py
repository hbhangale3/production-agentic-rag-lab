from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.schemas.indexing import PaperIndexingResult
from src.schemas.ingestion import IngestionResult


class ArxivDiscoveryProfile(BaseModel):
    """Application-owned policy for one arXiv discovery domain."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: str | None = None
    search_query: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def validate_query_source(self) -> "ArxivDiscoveryProfile":
        if bool(self.category) == bool(self.search_query):
            raise ValueError("exactly one of category or search_query is required")
        return self


class DiscoveryProfileResult(BaseModel):
    """Ingestion outcome attributed to a discovery profile."""

    profile_name: str
    result: IngestionResult
    indexing: PaperIndexingResult = Field(default_factory=PaperIndexingResult)


class DiscoveryRunResult(BaseModel):
    """Per-profile and aggregate outcomes for one scheduled run."""

    profiles: list[DiscoveryProfileResult] = Field(default_factory=list)
    aggregate: IngestionResult = Field(default_factory=IngestionResult)
    indexing: PaperIndexingResult = Field(default_factory=PaperIndexingResult)
    failed: int = Field(default=0, ge=0)
