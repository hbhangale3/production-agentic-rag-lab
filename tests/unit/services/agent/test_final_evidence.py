import dataclasses
import math
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.agent import (
    AgentEvidenceCandidate,
    EvidenceRerankError,
    FinalEvidenceSelector,
    TransientLiveEvidenceChunk,
    merge_evidence,
    normalize_live_evidence,
    normalize_local_evidence,
)
from src.services.agent.final_evidence import to_evidence_inputs
from src.services.evidence import EvidenceContextBuilder

QUESTION = "What are the current healthcare issues in India and how can AI help solve them?"
PUBLISHED = datetime(2024, 5, 6, tzinfo=UTC)


def make_hit(chunk_id: str, text: str, *, arxiv_id: str = "2401.00001", index: int = 0, **scores) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id,
        chunk_index=index,
        section_title="  Methods ",
        chunk_text=text,
        word_count=max(len(text.split()), 0),
        paper_title="  Local   Paper ",
        authors=["Ada Lovelace", " "],
        categories=["cs.AI", "cs.CY"],
        published_date=PUBLISHED,
        rrf_score=scores.get("rrf_score", 0.03),
        bm25_rank=scores.get("bm25_rank", 1),
        vector_rank=scores.get("vector_rank", 2),
        bm25_score=scores.get("bm25_score", 4.2),
        vector_score=scores.get("vector_score", 0.8),
    )


def local_result(*hits: HybridSearchHit) -> HybridSearchResult:
    return HybridSearchResult(query="q", retrieval_mode="hybrid", count=len(hits), results=list(hits))


def make_chunk(arxiv_id: str = "2501.00001", index: int = 0, text: str | None = None) -> TransientLiveEvidenceChunk:
    return TransientLiveEvidenceChunk(
        arxiv_id=arxiv_id,
        chunk_id=f"live::{arxiv_id}::chunk::{index:03d}",
        chunk_index=index,
        paper_title="Live Paper",
        section_title="Results",
        text=text or f"live chunk {arxiv_id} {index}",
        word_count=4,
        authors=("Grace Hopper",),
        categories=("cs.CY",),
        published_date=PUBLISHED,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


class FakeEmbeddingProvider:
    def __init__(self, scores: dict[str, float] | None = None, *, default: float = 0.5, error=None, scale=1.0) -> None:
        self.scores = scores or {}
        self.default = default
        self.error = error
        self.scale = scale
        self.queries: list[str] = []
        self.passage_batches: list[list[str]] = []

    def initialize(self) -> None:
        raise AssertionError("the reranker must not manage model lifecycle")

    def embed_query(self, text: str) -> list[float]:
        if self.error:
            raise self.error
        self.queries.append(text)
        return [3.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_batches.append(list(texts))
        vectors = []
        for text in texts:
            score = next((value for key, value in self.scores.items() if key in text), self.default)
            vectors.append([score * self.scale, math.sqrt(1 - score * score) * self.scale])
        return vectors


def test_local_normalization_preserves_metadata_and_drops_retrieval_scores() -> None:
    result = local_result(
        make_hit("p::chunk::000", "  First   local\n passage. ", rrf_score=0.9),
        make_hit("p::chunk::003", "Second local passage.", index=3),
    )

    candidates = normalize_local_evidence(result)

    assert candidates == (
        AgentEvidenceCandidate(
            source_type="local",
            arxiv_id="2401.00001",
            chunk_id="p::chunk::000",
            chunk_index=0,
            paper_title="Local Paper",
            section_title="Methods",
            text="First local passage.",
            authors=("Ada Lovelace",),
            categories=("cs.AI", "cs.CY"),
            published_date=PUBLISHED,
            source_url=None,
            original_rank=1,
        ),
        dataclasses.replace(candidates[0], chunk_id="p::chunk::003", chunk_index=3, text="Second local passage.", original_rank=2),
    )
    assert normalize_local_evidence(result) == candidates
    field_names = {field.name for field in dataclasses.fields(AgentEvidenceCandidate)}
    assert not {"rrf_score", "bm25_score", "vector_score", "bm25_rank", "vector_rank"} & field_names
    assert all(candidate.relevance_score is None for candidate in candidates)


def test_local_normalization_skips_blank_hits_and_keeps_original_rank() -> None:
    result = local_result(make_hit("a", "   "), make_hit("b", "usable"), make_hit(" ", "no chunk id"))

    assert [(c.chunk_id, c.original_rank) for c in normalize_local_evidence(result)] == [("b", 2)]
    assert normalize_local_evidence(None) == ()
    assert normalize_local_evidence(local_result()) == ()


def test_live_normalization_preserves_provenance_and_source_url() -> None:
    chunks = (make_chunk("2501.00001", 4), make_chunk("2501.00002", 0))

    candidates = normalize_live_evidence(chunks)

    assert candidates[0] == AgentEvidenceCandidate(
        source_type="live_arxiv",
        arxiv_id="2501.00001",
        chunk_id="live::2501.00001::chunk::004",
        chunk_index=4,
        paper_title="Live Paper",
        section_title="Results",
        text="live chunk 2501.00001 4",
        authors=("Grace Hopper",),
        categories=("cs.CY",),
        published_date=PUBLISHED,
        source_url="https://arxiv.org/pdf/2501.00001",
        original_rank=1,
    )
    assert [(c.arxiv_id, c.original_rank) for c in candidates] == [("2501.00001", 1), ("2501.00002", 2)]
    assert normalize_live_evidence(None) == normalize_live_evidence(()) == ()


def test_local_and_live_candidates_share_one_type_and_embedding_format() -> None:
    local = normalize_local_evidence(local_result(make_hit("p::chunk::000", "Local text.")))[0]
    live = normalize_live_evidence((make_chunk(text="Live text."),))[0]
    untitled = dataclasses.replace(live, paper_title="", section_title=None)

    assert type(local) is type(live) is AgentEvidenceCandidate
    assert local.embedding_text == "Local Paper\nMethods\nLocal text."
    assert live.embedding_text == "Live Paper\nResults\nLive text."
    assert untitled.embedding_text == "Live text."
    for candidate in (local, live):
        assert "Ada" not in candidate.embedding_text and "Grace" not in candidate.embedding_text
        assert "arxiv.org" not in candidate.embedding_text and "cs." not in candidate.embedding_text
    with pytest.raises(dataclasses.FrozenInstanceError):
        local.text = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_type": "web"},
        {"arxiv_id": " "},
        {"chunk_id": ""},
        {"text": "  "},
        {"authors": ["Ada"]},
        {"categories": ["cs.AI"]},
        {"original_rank": 0},
        {"original_rank": True},
    ],
)
def test_candidate_rejects_invalid_fields(overrides: dict) -> None:
    valid = normalize_live_evidence((make_chunk(),))[0]

    with pytest.raises(ValueError):
        dataclasses.replace(valid, **overrides)


def test_merge_of_single_sources_and_both_sources_keeps_local_then_live_order() -> None:
    local = normalize_local_evidence(local_result(make_hit("l1", "local one"), make_hit("l2", "local two", index=1)))
    live = normalize_live_evidence((make_chunk(index=0), make_chunk(index=1)))

    assert merge_evidence(local, ()) == local
    assert merge_evidence((), live) == live
    assert merge_evidence((), ()) == ()
    assert merge_evidence(local, live) == (*local, *live)
    assert merge_evidence(local, live) == merge_evidence(local, live)


def test_merge_drops_exact_chunk_id_duplicates_keeping_the_first() -> None:
    local = normalize_local_evidence(local_result(make_hit("same", "first text"), make_hit("same", "second text")))

    assert [(c.chunk_id, c.text) for c in merge_evidence(local, ())] == [("same", "first text")]


def test_merge_prefers_local_for_equivalent_text_of_the_same_paper() -> None:
    local = normalize_local_evidence(local_result(make_hit("2401.00001::chunk::002", "Shared passage text.", index=2)))
    live = normalize_live_evidence(
        (
            make_chunk("2401.00001v3", 2, text="  shared   PASSAGE text. "),
            make_chunk("2401.00001", 5, text="A different chunk of the same paper."),
            make_chunk("2502.00009", 0, text="Shared passage text."),
        )
    )

    merged = merge_evidence(local, live)

    assert [(c.source_type, c.chunk_id) for c in merged] == [
        ("local", "2401.00001::chunk::002"),
        ("live_arxiv", "live::2401.00001::chunk::005"),
        ("live_arxiv", "live::2502.00009::chunk::000"),
    ]
    assert local == normalize_local_evidence(local_result(make_hit("2401.00001::chunk::002", "Shared passage text.", index=2)))


def test_merge_keeps_distinct_chunks_from_one_paper() -> None:
    live = normalize_live_evidence(tuple(make_chunk(index=index) for index in range(4)))

    assert merge_evidence((), live) == live


@pytest.mark.anyio
async def test_live_candidate_outranks_local_candidate_with_high_retrieval_scores() -> None:
    local = normalize_local_evidence(
        local_result(make_hit("l1", "local weak passage", rrf_score=0.99, bm25_score=99.0, vector_score=0.99))
    )
    live = normalize_live_evidence((make_chunk(text="live strong passage"),))
    provider = FakeEmbeddingProvider({"local weak": 0.1, "live strong": 0.95})

    final = await FinalEvidenceSelector(embedding_provider=provider).select(
        QUESTION, merge_evidence(local, live), max_sources=5
    )

    assert [(c.source_type, round(c.relevance_score, 6)) for c in final] == [("live_arxiv", 0.95), ("local", 0.1)]
    assert [c.original_rank for c in final] == [1, 1]


@pytest.mark.anyio
async def test_question_and_one_passage_batch_are_embedded_with_the_shared_format() -> None:
    local = normalize_local_evidence(local_result(make_hit("l1", "local passage")))
    live = normalize_live_evidence((make_chunk(text="live passage"),))
    provider = FakeEmbeddingProvider()

    await FinalEvidenceSelector(embedding_provider=provider).select(
        f"  {QUESTION}\n", merge_evidence(local, live), max_sources=5
    )

    assert provider.queries == [QUESTION]
    assert provider.passage_batches == [["Local Paper\nMethods\nlocal passage", "Live Paper\nResults\nlive passage"]]


@pytest.mark.anyio
async def test_scores_are_true_cosine_even_for_unnormalized_vectors() -> None:
    live = normalize_live_evidence((make_chunk(index=0, text="alpha"), make_chunk(index=1, text="beta")))
    provider = FakeEmbeddingProvider({"alpha": 0.3, "beta": 0.6}, scale=7.5)

    final = await FinalEvidenceSelector(embedding_provider=provider).select(QUESTION, live, max_sources=5)

    assert [round(c.relevance_score, 6) for c in final] == [0.6, 0.3]
    assert all(-1.0 <= c.relevance_score <= 1.0 for c in final)


@pytest.mark.anyio
async def test_ties_keep_merged_order_and_selection_is_deterministic() -> None:
    local = normalize_local_evidence(local_result(make_hit("l1", "one"), make_hit("l2", "two", index=1)))
    live = normalize_live_evidence((make_chunk(index=0), make_chunk(index=1)))
    merged = merge_evidence(local, live)
    selector = FinalEvidenceSelector(embedding_provider=FakeEmbeddingProvider(default=0.5))

    first = await selector.select(QUESTION, merged, max_sources=5)
    second = await selector.select(QUESTION, merged, max_sources=5)

    assert first == second
    assert [c.chunk_id for c in first] == [c.chunk_id for c in merged]


@pytest.mark.anyio
@pytest.mark.parametrize(("max_sources", "expected"), [(1, ["l3"]), (2, ["l3", "l1"]), (10, ["l3", "l1", "l2"])])
async def test_result_limit_is_enforced(max_sources: int, expected: list[str]) -> None:
    local = normalize_local_evidence(
        local_result(make_hit("l1", "one"), make_hit("l2", "two", index=1), make_hit("l3", "three", index=2))
    )
    provider = FakeEmbeddingProvider({"three": 0.9, "one": 0.5, "two": 0.2})

    final = await FinalEvidenceSelector(embedding_provider=provider).select(QUESTION, local, max_sources=max_sources)

    assert [c.chunk_id for c in final] == expected


@pytest.mark.anyio
async def test_zero_candidates_returns_empty_without_embedding() -> None:
    provider = FakeEmbeddingProvider()

    assert await FinalEvidenceSelector(embedding_provider=provider).select(QUESTION, (), max_sources=5) == ()
    assert provider.queries == [] and provider.passage_batches == []


@pytest.mark.anyio
async def test_embedding_work_is_capped_without_starving_either_source() -> None:
    local = normalize_local_evidence(local_result(*(make_hit(f"l{i}", f"local {i}", index=i) for i in range(10))))
    live = normalize_live_evidence(tuple(make_chunk(index=i) for i in range(10)))
    provider = FakeEmbeddingProvider()

    final = await FinalEvidenceSelector(embedding_provider=provider, max_candidates=6).select(
        QUESTION, merge_evidence(local, live), max_sources=20
    )

    assert len(provider.passage_batches) == 1 and len(provider.passage_batches[0]) == 6
    assert [c.chunk_id for c in final] == [
        "l0",
        "l1",
        "l2",
        "live::2501.00001::chunk::000",
        "live::2501.00001::chunk::001",
        "live::2501.00001::chunk::002",
    ]

    provider = FakeEmbeddingProvider()
    await FinalEvidenceSelector(embedding_provider=provider, max_candidates=6).select(QUESTION, local, max_sources=20)
    assert len(provider.passage_batches[0]) == 6


@pytest.mark.anyio
async def test_synchronous_embedding_is_offloaded_in_one_hop() -> None:
    live = normalize_live_evidence((make_chunk(),))
    provider = FakeEmbeddingProvider()
    selector = FinalEvidenceSelector(embedding_provider=provider)
    offload = AsyncMock(return_value=([1.0, 0.0], [[1.0, 0.0]]))

    with patch("src.services.agent.final_evidence.asyncio.to_thread", new=offload):
        final = await selector.select(QUESTION, live, max_sources=5)

    offload.assert_awaited_once_with(selector._embed, QUESTION, ["Live Paper\nResults\nlive chunk 2501.00001 0"])
    assert provider.queries == [] and provider.passage_batches == []
    assert final[0].relevance_score == 1.0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "vectors",
    [
        ([1.0, 0.0], []),
        ([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]]),
        ([1.0, 0.0], [[1.0, 0.0, 0.0]]),
        ([0.0, 0.0], [[1.0, 0.0]]),
        ([1.0, 0.0], [[0.0, 0.0]]),
        ([1.0, 0.0], [[math.nan, 0.0]]),
        ([], [[]]),
    ],
)
async def test_unusable_embeddings_are_rerank_errors_not_arbitrary_order(vectors) -> None:
    live = normalize_live_evidence((make_chunk(),))

    with patch("src.services.agent.final_evidence.asyncio.to_thread", new=AsyncMock(return_value=vectors)):
        with pytest.raises(EvidenceRerankError):
            await FinalEvidenceSelector(embedding_provider=FakeEmbeddingProvider()).select(QUESTION, live, max_sources=5)


@pytest.mark.anyio
async def test_embedding_failure_is_wrapped_without_details() -> None:
    live = normalize_live_evidence((make_chunk(),))
    provider = FakeEmbeddingProvider(error=RuntimeError("private embedding detail"))

    with pytest.raises(EvidenceRerankError) as caught:
        await FinalEvidenceSelector(embedding_provider=provider).select(QUESTION, live, max_sources=5)

    assert str(caught.value) == "evidence embedding failed"


@pytest.mark.anyio
@pytest.mark.parametrize(("question", "max_sources"), [("  ", 5), (QUESTION, 0), (QUESTION, True), (QUESTION, 2.5)])
async def test_invalid_selection_arguments_are_programming_errors(question: str, max_sources) -> None:
    with pytest.raises(ValueError):
        await FinalEvidenceSelector(embedding_provider=FakeEmbeddingProvider()).select(
            question, normalize_live_evidence((make_chunk(),)), max_sources=max_sources
        )


def test_invalid_candidate_cap_is_rejected() -> None:
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            FinalEvidenceSelector(embedding_provider=FakeEmbeddingProvider(), max_candidates=invalid)


def test_final_candidates_become_a_labelled_context_without_losing_provenance() -> None:
    live = normalize_live_evidence((make_chunk("2501.00001", 4, text="Live full text."),))
    local = normalize_local_evidence(local_result(make_hit("p::chunk::000", "Local text.")))
    final = (*live, *local)

    context = EvidenceContextBuilder(max_context_tokens=1000).build_from_sources(
        to_evidence_inputs(final), query="q", retrieval_mode="hybrid"
    )

    assert [(s.label, s.source_type, s.chunk_id) for s in context.sources] == [
        ("[S1]", "live_arxiv", "live::2501.00001::chunk::004"),
        ("[S2]", "local", "p::chunk::000"),
    ]
    live_source, local_source = context.sources
    assert live_source.source_url == "https://arxiv.org/pdf/2501.00001"
    assert (live_source.arxiv_id, live_source.chunk_index, live_source.paper_title) == ("2501.00001", 4, "Live Paper")
    assert (live_source.section_title, live_source.authors, live_source.categories) == (
        "Results",
        ("Grace Hopper",),
        ("cs.CY",),
    )
    assert live_source.published_date == PUBLISHED
    assert live_source.rrf_score is None and local_source.rrf_score is None
    assert local_source.source_url is None
    assert len({s.label for s in context.sources}) == 2
