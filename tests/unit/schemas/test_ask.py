from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from src.schemas.ask import AskRequest, AskSource, build_ask_response
from src.services.evidence import EvidenceSource
from src.services.rag import RAGGenerationResult


def source(
    label: str,
    *,
    rank: int,
    content: str,
    chunk_id: str,
    title: str = "Grounded Retrieval",
    section: str | None = "Results",
    truncated: bool = False,
) -> EvidenceSource:
    return EvidenceSource(
        label=label,
        retrieval_rank=rank,
        arxiv_id="2401.00001",
        chunk_id=chunk_id,
        chunk_index=rank - 1,
        paper_title=title,
        section_title=section,
        content=content,
        authors=("Ada Lovelace", "山田 太郎"),
        categories=("cs.IR", "cs.AI"),
        published_date=datetime(2024, 2, 3, 4, 5, tzinfo=UTC),
        rrf_score=0.03,
        bm25_rank=rank,
        vector_rank=rank + 1,
        bm25_score=4.2,
        vector_score=0.8,
        truncated=truncated,
    )


def generation_result(
    *sources: EvidenceSource,
    prompt_tokens: int | None = 120,
    completion_tokens: int | None = 18,
) -> RAGGenerationResult:
    return RAGGenerationResult(
        answer="The method combines retrieval signals [S1].",
        sources=tuple(sources),
        retrieval_mode="hybrid",
        model="test-model",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cited_labels=("[S1]",),
        estimated_prompt_tokens=115,
        token_counting="estimated",
    )


def test_valid_ask_request_preserves_question() -> None:
    request = AskRequest(question="  How does RAG work?  ")

    assert request.question == "  How does RAG work?  "


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_ask_request_is_rejected(question: str) -> None:
    with pytest.raises(ValidationError, match="question must not be blank"):
        AskRequest(question=question)


def test_maps_generation_result_to_public_contract_in_source_order() -> None:
    first = source("[S1]", rank=1, content="First evidence.", chunk_id="paper::chunk::000")
    second = source(
        "[S2]",
        rank=2,
        content="Second selected excerpt …",
        chunk_id="paper::chunk::001",
        title="Unicode research — 東京",
        section=None,
        truncated=True,
    )

    response = build_ask_response(generation_result(first, second))

    assert response.answer == "The method combines retrieval signals [S1]."
    assert response.model == "test-model"
    assert response.retrieval_mode == "hybrid"
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 18
    assert [item.citation for item in response.sources] == ["[S1]", "[S2]"]
    assert [item.chunk_id for item in response.sources] == [
        "paper::chunk::000",
        "paper::chunk::001",
    ]
    assert response.sources[0].arxiv_id == "2401.00001"
    assert response.sources[0].chunk_index == 0
    assert response.sources[0].title == "Grounded Retrieval"
    assert response.sources[0].section == "Results"
    assert response.sources[1].content == "Second selected excerpt …"
    assert response.sources[1].truncated is True
    assert response.sources[1].section is None


def test_returns_all_supplied_sources_even_when_answer_does_not_cite_them() -> None:
    result = generation_result(
        source("[S1]", rank=1, content="Cited.", chunk_id="one"),
        source("[S2]", rank=2, content="Uncited but supplied.", chunk_id="two"),
    )

    response = build_ask_response(result)

    assert result.cited_labels == ("[S1]",)
    assert [item.citation for item in response.sources] == ["[S1]", "[S2]"]


def test_nullable_usage_and_missing_optional_metadata_are_preserved() -> None:
    internal_source = source(
        "[S1]",
        rank=1,
        content="Evidence.",
        chunk_id="chunk",
        title="",
        section=None,
    )
    internal_source = EvidenceSource(
        **{
            **internal_source.__dict__,
            "authors": (),
            "categories": (),
            "published_date": None,
        }
    )

    response = build_ask_response(
        generation_result(internal_source, prompt_tokens=None, completion_tokens=None)
    )

    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.sources[0].title == ""
    assert response.sources[0].section is None
    assert response.sources[0].authors == []
    assert response.sources[0].categories == []
    assert response.sources[0].published_date is None


def test_json_serialization_is_public_and_excludes_internal_diagnostics() -> None:
    response = build_ask_response(
        generation_result(
            source(
                "[S1]",
                rank=1,
                content="Résumé evidence from 東京 🚀.",
                chunk_id="unicode",
            )
        )
    )

    payload = response.model_dump(mode="json")
    public_source = payload["sources"][0]

    assert public_source == {
        "citation": "[S1]",
        "retrieval_rank": 1,
        "arxiv_id": "2401.00001",
        "chunk_id": "unicode",
        "chunk_index": 0,
        "title": "Grounded Retrieval",
        "section": "Results",
        "content": "Résumé evidence from 東京 🚀.",
        "truncated": False,
        "authors": ["Ada Lovelace", "山田 太郎"],
        "categories": ["cs.IR", "cs.AI"],
        "published_date": "2024-02-03T04:05:00Z",
    }
    assert not {
        "rrf_score",
        "bm25_rank",
        "vector_rank",
        "bm25_score",
        "vector_score",
    } & public_source.keys()


def test_mapper_does_not_mutate_internal_result() -> None:
    result = generation_result(source("[S1]", rank=1, content="Evidence.", chunk_id="chunk"))
    before = deepcopy(result)

    build_ask_response(result)

    assert result == before


@pytest.mark.parametrize("citation", ["S1", "[S0]", "[S-1]", "[source1]"])
def test_public_citation_requires_response_local_label(citation: str) -> None:
    with pytest.raises(ValidationError):
        AskSource(
            citation=citation,
            retrieval_rank=1,
            arxiv_id="2401.00001",
            chunk_id="chunk",
            chunk_index=0,
            title="Title",
            content="Evidence",
            truncated=False,
        )
