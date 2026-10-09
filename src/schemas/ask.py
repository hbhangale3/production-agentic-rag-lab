from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.services.rag.service import RAGGenerationResult


class AskRequest(BaseModel):
    """Intentionally small public request for research questions."""

    question: str = Field(description="Question to ask about indexed research papers")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskSource(BaseModel):
    """Authoritative evidence excerpt exposed with a response-local citation."""

    citation: str = Field(pattern=r"^\[S[1-9]\d*\]$")
    retrieval_rank: int = Field(ge=1)
    arxiv_id: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    title: str
    section: str | None = None
    content: str
    truncated: bool
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_date: datetime | None = None


class AskResponse(BaseModel):
    """Public grounded answer contract."""

    answer: str
    sources: list[AskSource]
    retrieval_mode: Literal["hybrid", "bm25_fallback", "vector_fallback"] | None = None
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


def build_ask_response(result: "RAGGenerationResult") -> AskResponse:
    """Map internal grounded output to the stable public representation."""
    return AskResponse(
        answer=result.answer,
        sources=[
            AskSource(
                citation=source.label,
                retrieval_rank=source.retrieval_rank,
                arxiv_id=source.arxiv_id,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                title=source.paper_title,
                section=source.section_title,
                content=source.content,
                truncated=source.truncated,
                authors=list(source.authors),
                categories=list(source.categories),
                published_date=source.published_date,
            )
            for source in result.sources
        ],
        retrieval_mode=result.retrieval_mode,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
