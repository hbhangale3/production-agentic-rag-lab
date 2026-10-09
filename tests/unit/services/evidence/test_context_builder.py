from copy import deepcopy
from datetime import UTC, datetime

import pytest
from src.config import Settings
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.evidence import CharacterTokenEstimator, EvidenceContextBuilder


def make_hit(
    chunk_id: str,
    text: str,
    *,
    arxiv_id: str = "2401.00001",
    chunk_index: int = 0,
    title: str = "Reliable Retrieval",
    section: str | None = "Methods",
) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id,
        chunk_index=chunk_index,
        section_title=section,
        chunk_text=text,
        word_count=len(text.split()),
        paper_title=title,
        authors=["Ada Lovelace"],
        categories=["cs.IR"],
        published_date=datetime(2024, 1, 2, tzinfo=UTC),
        rrf_score=0.03,
        bm25_rank=1,
        vector_rank=2,
        bm25_score=4.2,
        vector_score=0.8,
    )


def make_result(*hits: HybridSearchHit) -> HybridSearchResult:
    return HybridSearchResult(
        query="how does retrieval work?",
        retrieval_mode="hybrid",
        count=len(hits),
        results=list(hits),
    )


def test_builds_ordered_citation_context_and_source_mapping() -> None:
    first = make_hit("paper-a::chunk::000", "First ranked evidence.")
    second = make_hit(
        "paper-b::chunk::003",
        "Second ranked evidence.",
        arxiv_id="2402.00002",
        chunk_index=3,
        title="Hybrid Search",
        section="Results",
    )

    context = EvidenceContextBuilder(max_context_tokens=1000).build(make_result(first, second))

    assert [source.label for source in context.sources] == ["[S1]", "[S2]"]
    assert [source.chunk_id for source in context.sources] == [
        "paper-a::chunk::000",
        "paper-b::chunk::003",
    ]
    assert context.text.index("[S1]") < context.text.index("[S2]")
    assert context.sources[1].arxiv_id == "2402.00002"
    assert context.sources[1].paper_title == "Hybrid Search"
    assert context.sources[1].section_title == "Results"
    assert context.sources[0].authors == ("Ada Lovelace",)
    assert context.sources[0].categories == ("cs.IR",)
    assert context.retrieval_mode == "hybrid"
    assert "estimated" in context.token_counting


def test_deduplicates_chunk_ids_and_normalized_exact_evidence() -> None:
    result = make_result(
        make_hit("duplicate-id", "Keep this passage."),
        make_hit("duplicate-id", "Different text with the same ID."),
        make_hit("different-id", "  Keep   this\npassage.  "),
        make_hit("unique-id", "Keep this distinct passage."),
    )

    context = EvidenceContextBuilder(max_context_tokens=1000).build(result)

    assert [source.chunk_id for source in context.sources] == ["duplicate-id", "unique-id"]
    assert [source.retrieval_rank for source in context.sources] == [1, 4]
    assert [source.label for source in context.sources] == ["[S1]", "[S2]"]


def test_keeps_distinct_chunks_from_the_same_paper() -> None:
    context = EvidenceContextBuilder(max_context_tokens=1000).build(
        make_result(
            make_hit("same-paper::chunk::000", "Introduction evidence.", chunk_index=0),
            make_hit("same-paper::chunk::001", "Methods evidence.", chunk_index=1),
        )
    )

    assert len(context.sources) == 2
    assert {source.arxiv_id for source in context.sources} == {"2401.00001"}
    assert [source.chunk_index for source in context.sources] == [0, 1]


def test_truncates_oversized_top_ranked_passage_within_budget() -> None:
    result = make_result(
        make_hit("large", "word " * 100),
        make_hit("later", "This lower-ranked passage must not leapfrog."),
    )
    builder = EvidenceContextBuilder(
        max_context_tokens=125,
        token_counter=CharacterTokenEstimator(characters_per_token=1),
    )

    context = builder.build(result)

    assert context.estimated_tokens <= 125
    assert len(context.sources) == 1
    assert context.sources[0].chunk_id == "large"
    assert context.sources[0].truncated is True
    assert context.sources[0].content.endswith(" …")
    assert "later" not in context.text


def test_insufficient_budget_returns_empty_context() -> None:
    context = EvidenceContextBuilder(
        max_context_tokens=1,
        token_counter=CharacterTokenEstimator(characters_per_token=1),
    ).build(make_result(make_hit("chunk", "Evidence")))

    assert context.text == ""
    assert context.sources == ()
    assert context.estimated_tokens == 0


def test_empty_retrieval_returns_structured_empty_context() -> None:
    result = HybridSearchResult(
        query="nothing found",
        retrieval_mode="bm25_fallback",
        count=0,
        results=[],
        degradation_reason="vector unavailable",
    )

    context = EvidenceContextBuilder(max_context_tokens=100).build(result)

    assert context.text == ""
    assert context.sources == ()
    assert context.query == "nothing found"
    assert context.retrieval_mode == "bm25_fallback"
    assert context.degradation_reason == "vector unavailable"


def test_malformed_optional_metadata_uses_safe_fallbacks() -> None:
    hit = make_hit("chunk", "Useful evidence.")
    malformed = hit.model_copy(
        update={
            "paper_title": None,
            "section_title": 42,
            "authors": None,
            "categories": ["cs.AI", "  ", None],
            "published_date": "not-a-date",
        }
    )

    context = EvidenceContextBuilder(max_context_tokens=1000).build(make_result(malformed))

    assert "[S1] Unknown title" in context.text
    assert "Section:" not in context.text
    assert context.sources[0].paper_title == ""
    assert context.sources[0].section_title is None
    assert context.sources[0].authors == ()
    assert context.sources[0].categories == ("cs.AI",)
    assert context.sources[0].published_date is None


def test_unicode_and_whitespace_are_normalized_deterministically() -> None:
    hit = make_hit("unicode", "  naïve\tRAG\nhandles   東京 and 🚀.  ")
    builder = EvidenceContextBuilder(max_context_tokens=1000)

    first = builder.build(make_result(hit))
    second = builder.build(make_result(hit))

    assert first == second
    assert first.sources[0].content == "naïve RAG handles 東京 and 🚀."
    assert "naïve RAG handles 東京 and 🚀." in first.text


def test_build_does_not_mutate_retrieval_result() -> None:
    result = make_result(make_hit("chunk", "  spaced   evidence  "))
    before = deepcopy(result.model_dump())

    EvidenceContextBuilder(max_context_tokens=1000).build(result)

    assert result.model_dump() == before


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_context_budget_must_be_a_positive_integer(budget: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EvidenceContextBuilder(max_context_tokens=budget)  # type: ignore[arg-type]


@pytest.mark.parametrize("characters_per_token", [0, -1, True, 1.5])
def test_character_estimator_requires_a_positive_integer(characters_per_token: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CharacterTokenEstimator(characters_per_token=characters_per_token)  # type: ignore[arg-type]


def test_builder_uses_configured_evidence_only_budget() -> None:
    settings = Settings(_env_file=None, debug=False, evidence_context_max_tokens=321)

    builder = EvidenceContextBuilder.from_settings(settings)

    assert builder.max_context_tokens == 321
