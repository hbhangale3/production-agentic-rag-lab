from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.config import Settings
from src.dependencies import (
    get_hybrid_search_service,
    get_rag_generation_service,
    get_rag_response_cache,
)
from src.exceptions import LLMResponseError
from src.main import app
from src.schemas.ask import AskResponse
from src.schemas.hybrid_search import HybridSearchResult
from src.services.cache import RAGResponseCacheCoordinator, serialize_cached_response
from src.services.evidence import EvidenceSource
from src.services.rag import RAGGenerationResult


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, str, int]] = []
        self.set_result = True

    async def get(self, key: str) -> str | None:
        self.get_calls.append(key)
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        self.set_calls.append((key, value, ttl_seconds))
        if self.set_result:
            self.values[key] = value
        return self.set_result


class MutableFingerprint:
    def __init__(self, value: str | None = "a" * 64) -> None:
        self.value = value

    def get_fingerprint(self) -> str | None:
        return self.value


def generated_result() -> RAGGenerationResult:
    source = EvidenceSource(
        label="[S1]",
        retrieval_rank=1,
        arxiv_id="2401.00001",
        chunk_id="paper::chunk::000",
        chunk_index=0,
        paper_title="Paper",
        section_title="Results",
        content="Evidence.",
        authors=("Researcher",),
        categories=("cs.AI",),
        published_date=datetime(2024, 1, 2, tzinfo=UTC),
        rrf_score=0.03,
        bm25_rank=1,
        vector_rank=1,
        bm25_score=4.0,
        vector_score=0.8,
        truncated=False,
    )
    return RAGGenerationResult(
        answer="Grounded answer [S1].",
        sources=(source,),
        retrieval_mode="bm25_fallback",
        model="test-model",
        prompt_tokens=81,
        completion_tokens=9,
        cited_labels=("[S1]",),
        estimated_prompt_tokens=75,
        token_counting="estimated",
    )


def cached_response() -> AskResponse:
    return AskResponse(
        answer="Previously grounded [S1].",
        sources=[],
        retrieval_mode="vector_fallback",
        model="cached-model",
        prompt_tokens=55,
        completion_tokens=7,
    )


@pytest.fixture
def cache_setup():
    cache = MemoryCache()
    fingerprint = MutableFingerprint()
    settings = Settings(redis_enabled=True, rag_cache_ttl_seconds=321)
    coordinator = RAGResponseCacheCoordinator(
        cache=cache,
        settings=settings,
        fingerprint_provider=fingerprint,
    )
    hybrid = Mock()
    hybrid.search.return_value = HybridSearchResult(
        query="Question", retrieval_mode="hybrid", count=0, results=[]
    )
    rag = Mock()
    rag.generate = AsyncMock(return_value=generated_result())
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid
    app.dependency_overrides[get_rag_generation_service] = lambda: rag
    app.dependency_overrides[get_rag_response_cache] = lambda: coordinator
    yield cache, fingerprint, coordinator, hybrid, rag
    for dependency in (
        get_hybrid_search_service,
        get_rag_generation_service,
        get_rag_response_cache,
    ):
        app.dependency_overrides.pop(dependency, None)


@pytest.mark.anyio
async def test_valid_cache_hit_bypasses_all_rag_work(client, cache_setup) -> None:
    cache, _, coordinator, hybrid, rag = cache_setup
    lookup = await coordinator.lookup(question="Question")
    cache.get_calls.clear()
    cache.values[lookup.key] = serialize_cached_response(cached_response())

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert response.json() == cached_response().model_dump(mode="json")
    assert response.json()["retrieval_mode"] == "vector_fallback"
    assert response.json()["prompt_tokens"] == 55
    hybrid.search.assert_not_called()
    rag.generate.assert_not_awaited()
    assert len(cache.get_calls) == 1
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_cache_miss_runs_once_and_writes_exact_response(client, cache_setup) -> None:
    cache, _, _, hybrid, rag = cache_setup

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    hybrid.search.assert_called_once_with("Question", size=5)
    rag.generate.assert_awaited_once()
    assert len(cache.get_calls) == len(cache.set_calls) == 1
    key, payload, ttl = cache.set_calls[0]
    assert key == cache.get_calls[0]
    assert ttl == 321
    assert serialize_cached_response(AskResponse.model_validate(response.json())) == payload


@pytest.mark.anyio
async def test_repeated_identical_request_generates_only_once(client, cache_setup) -> None:
    cache, _, _, hybrid, rag = cache_setup

    first = await client.post("/api/v1/ask", json={"question": "Same question"})
    second = await client.post("/api/v1/ask", json={"question": "Same question"})

    assert first.json() == second.json()
    assert len(cache.get_calls) == 2
    assert len(cache.set_calls) == 1
    assert hybrid.search.call_count == 1
    assert rag.generate.await_count == 1


@pytest.mark.anyio
async def test_different_questions_do_not_share_cached_answer(client, cache_setup) -> None:
    cache, _, _, hybrid, rag = cache_setup

    await client.post("/api/v1/ask", json={"question": "Question A"})
    await client.post("/api/v1/ask", json={"question": "Question B"})

    assert cache.get_calls[0] != cache.get_calls[1]
    assert hybrid.search.call_count == 2
    assert rag.generate.await_count == 2


@pytest.mark.anyio
async def test_changed_corpus_fingerprint_does_not_reuse_entry(client, cache_setup) -> None:
    cache, fingerprint, _, hybrid, rag = cache_setup
    await client.post("/api/v1/ask", json={"question": "Question"})
    fingerprint.value = "b" * 64

    await client.post("/api/v1/ask", json={"question": "Question"})

    assert cache.get_calls[0] != cache.get_calls[1]
    assert hybrid.search.call_count == 2
    assert rag.generate.await_count == 2


@pytest.mark.anyio
async def test_corrupt_entry_is_replaced_after_normal_generation(client, cache_setup) -> None:
    cache, _, coordinator, hybrid, _ = cache_setup
    lookup = await coordinator.lookup(question="Question")
    cache.get_calls.clear()
    cache.values[lookup.key] = "not-json"

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert hybrid.search.call_count == 1
    assert len(cache.set_calls) == 1
    assert cache.values[lookup.key] != "not-json"


@pytest.mark.anyio
async def test_unavailable_fingerprint_bypasses_cache(client, cache_setup) -> None:
    cache, fingerprint, _, hybrid, rag = cache_setup
    fingerprint.value = None

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert cache.get_calls == cache.set_calls == []
    assert hybrid.search.call_count == 1
    assert rag.generate.await_count == 1


@pytest.mark.anyio
async def test_cache_set_unavailable_does_not_change_success(client, cache_setup) -> None:
    cache, _, _, _, _ = cache_setup
    cache.set_result = False

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer [S1]."
    assert len(cache.set_calls) == 1


@pytest.mark.anyio
async def test_cache_get_failure_is_a_fail_open_miss(client, cache_setup) -> None:
    cache, _, _, hybrid, rag = cache_setup
    cache.get = AsyncMock(side_effect=RuntimeError("cache unavailable"))

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert hybrid.search.call_count == 1
    assert rag.generate.await_count == 1


@pytest.mark.anyio
async def test_serialization_failure_does_not_change_success(client, cache_setup) -> None:
    cache, _, _, _, _ = cache_setup
    with patch(
        "src.services.cache.rag_response_cache.serialize_cached_response",
        side_effect=ValueError("safe test failure"),
    ):
        response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 200
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_rag_failure_never_writes_cache(client, cache_setup) -> None:
    cache, _, _, _, rag = cache_setup
    rag.generate.side_effect = LLMResponseError("bad output")

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 502
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_successful_insufficient_evidence_response_is_cached(client, cache_setup) -> None:
    cache, _, _, _, rag = cache_setup
    from src.exceptions import InsufficientEvidenceError

    rag.generate.side_effect = InsufficientEvidenceError("no evidence")

    first = await client.post("/api/v1/ask", json={"question": "Unknown"})
    second = await client.post("/api/v1/ask", json={"question": "Unknown"})

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert rag.generate.await_count == 1
    assert len(cache.set_calls) == 1


@pytest.mark.anyio
async def test_answer_affecting_model_change_uses_an_independent_key() -> None:
    cache = MemoryCache()
    fingerprint = MutableFingerprint()
    first = RAGResponseCacheCoordinator(
        cache=cache,
        settings=Settings(redis_enabled=True, groq_model="model-a"),
        fingerprint_provider=fingerprint,
    )
    second = RAGResponseCacheCoordinator(
        cache=cache,
        settings=Settings(redis_enabled=True, groq_model="model-b"),
        fingerprint_provider=fingerprint,
    )

    first_lookup = await first.lookup(question="Question")
    await first.store(first_lookup, cached_response())
    second_lookup = await second.lookup(question="Question")

    assert first_lookup.key != second_lookup.key
    assert second_lookup.response is None
