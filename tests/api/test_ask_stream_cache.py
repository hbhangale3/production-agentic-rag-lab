from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.config import Settings
from src.dependencies import (
    get_hybrid_search_service,
    get_rag_generation_service,
    get_rag_response_cache,
)
from src.exceptions import LLMRequestError
from src.main import app
from src.schemas.ask import AskResponse, AskSource
from src.services.cache import (
    RAGResponseCacheCoordinator,
    deserialize_cached_response,
    serialize_cached_response,
)
from src.services.evidence import EvidenceContextBuilder
from src.services.llm.base import LLMStreamEvent
from src.services.rag import RAGGenerationService

from tests.api.test_ask_cache import MemoryCache, MutableFingerprint, generated_result
from tests.api.test_ask_stream import FakeStreamingProvider, parse_sse, retrieval


def replay_response() -> AskResponse:
    return AskResponse(
        answer="Cached café result [S2] and 日本語 [S1].",
        sources=[
            AskSource(
                citation=f"[S{rank}]",
                retrieval_rank=rank,
                arxiv_id=f"2401.0000{rank}",
                chunk_id=f"paper-{rank}::chunk::000",
                chunk_index=0,
                title=f"Paper {rank}",
                section="Results",
                content=f"Evidence {rank} café.",
                truncated=False,
                authors=["Researcher"],
                categories=["cs.AI"],
            )
            for rank in (1, 2)
        ],
        retrieval_mode="vector_fallback",
        model="cached-model",
        prompt_tokens=73,
        completion_tokens=11,
    )


@pytest.fixture
def stream_cache_setup():
    cache = MemoryCache()
    fingerprint = MutableFingerprint()
    coordinator = RAGResponseCacheCoordinator(
        cache=cache,
        settings=Settings(redis_enabled=True, rag_cache_ttl_seconds=654),
        fingerprint_provider=fingerprint,
    )
    hybrid = Mock()
    hybrid.search.return_value = retrieval()
    provider = FakeStreamingProvider(
        (
            LLMStreamEvent(text="Grounded ", model="fake-model"),
            LLMStreamEvent(text="café [S1]."),
            LLMStreamEvent(model="fake-model", prompt_tokens=80, completion_tokens=9),
        )
    )
    rag = RAGGenerationService(
        llm_provider=provider,
        evidence_builder=EvidenceContextBuilder(max_context_tokens=1000),
    )
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid
    app.dependency_overrides[get_rag_generation_service] = lambda: rag
    app.dependency_overrides[get_rag_response_cache] = lambda: coordinator
    yield cache, fingerprint, coordinator, hybrid, provider, rag
    for dependency in (
        get_hybrid_search_service,
        get_rag_generation_service,
        get_rag_response_cache,
    ):
        app.dependency_overrides.pop(dependency, None)


@pytest.mark.anyio
async def test_stream_cache_hit_replays_exact_response_and_bypasses_rag(
    client, stream_cache_setup
) -> None:
    cache, _, coordinator, hybrid, provider, rag = stream_cache_setup
    rag.answer_validator.validate = Mock(side_effect=AssertionError("must not revalidate hit"))
    cached = replay_response()
    lookup = await coordinator.lookup(question="Question")
    cache.get_calls.clear()
    cache.values[lookup.key] = serialize_cached_response(cached)

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["metadata", "delta", "sources", "done"]
    assert events[0][1]["retrieval_mode"] == "vector_fallback"
    assert events[0][1]["source_count"] == 2
    assert events[0][1]["cache_status"] == "hit"
    assert "".join(data["text"] for name, data in events if name == "delta") == cached.answer
    assert events[-2][1]["sources"] == cached.model_dump(mode="json")["sources"]
    assert events[-1][1] == {
        "model": "cached-model",
        "prompt_tokens": 73,
        "completion_tokens": 11,
        "citations": ["[S2]", "[S1]"],
    }
    hybrid.search.assert_not_called()
    assert provider.calls == []
    rag.answer_validator.validate.assert_not_called()
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_stream_miss_uses_genuine_stream_and_stores_final_response(
    client, stream_cache_setup
) -> None:
    cache, _, _, hybrid, provider, _ = stream_cache_setup

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["metadata", "delta", "delta", "sources", "done"]
    assert len(provider.calls) == hybrid.search.call_count == 1
    assert len(cache.get_calls) == len(cache.set_calls) == 1
    key, payload, ttl = cache.set_calls[0]
    assert key == cache.get_calls[0]
    assert ttl == 654
    stored = deserialize_cached_response(payload)
    assert stored is not None
    assert stored.answer == "Grounded café [S1]."
    assert stored.model == "fake-model"
    assert stored.prompt_tokens == 80
    assert stored.completion_tokens == 9
    assert events[0][1]["cache_status"] == "miss"


@pytest.mark.anyio
async def test_repeated_stream_request_generates_once(client, stream_cache_setup) -> None:
    cache, _, _, hybrid, provider, _ = stream_cache_setup

    first = parse_sse((await client.post("/api/v1/ask/stream", json={"question": "Same"})).text)
    second = parse_sse((await client.post("/api/v1/ask/stream", json={"question": "Same"})).text)

    assert "".join(data["text"] for name, data in first if name == "delta") == "".join(
        data["text"] for name, data in second if name == "delta"
    )
    assert hybrid.search.call_count == len(provider.calls) == len(cache.set_calls) == 1


@pytest.mark.anyio
async def test_stream_questions_and_corpus_generations_are_isolated(
    client, stream_cache_setup
) -> None:
    cache, fingerprint, _, hybrid, provider, _ = stream_cache_setup
    await client.post("/api/v1/ask/stream", json={"question": "A"})
    await client.post("/api/v1/ask/stream", json={"question": "B"})
    fingerprint.value = "b" * 64
    await client.post("/api/v1/ask/stream", json={"question": "A"})

    assert len(set(cache.get_calls)) == 3
    assert hybrid.search.call_count == len(provider.calls) == 3


@pytest.mark.anyio
async def test_corrupt_stream_entry_is_replaced(client, stream_cache_setup) -> None:
    cache, _, coordinator, _, provider, _ = stream_cache_setup
    lookup = await coordinator.lookup(question="Question")
    cache.get_calls.clear()
    cache.values[lookup.key] = "corrupt"

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    assert parse_sse(response.text)[-1][0] == "done"
    assert len(provider.calls) == 1
    assert deserialize_cached_response(cache.values[lookup.key]) is not None


@pytest.mark.anyio
async def test_stream_cache_failures_and_fingerprint_bypass_preserve_success(
    client, stream_cache_setup
) -> None:
    cache, fingerprint, _, hybrid, provider, _ = stream_cache_setup
    cache.set_result = False
    first = await client.post("/api/v1/ask/stream", json={"question": "Write failure"})
    fingerprint.value = None
    before_gets = len(cache.get_calls)
    before_sets = len(cache.set_calls)
    second = await client.post("/api/v1/ask/stream", json={"question": "Bypass"})

    assert parse_sse(first.text)[-1][0] == parse_sse(second.text)[-1][0] == "done"
    assert parse_sse(first.text)[0][1]["cache_status"] == "miss"
    assert parse_sse(second.text)[0][1]["cache_status"] == "bypass"
    assert len(cache.get_calls) == before_gets
    assert len(cache.set_calls) == before_sets
    assert hybrid.search.call_count == len(provider.calls) == 2


@pytest.mark.anyio
async def test_stream_cache_read_and_serialization_failures_are_fail_open(
    client, stream_cache_setup
) -> None:
    cache, _, _, hybrid, provider, _ = stream_cache_setup
    cache.get = AsyncMock(side_effect=RuntimeError("unavailable"))
    with patch(
        "src.services.cache.rag_response_cache.serialize_cached_response",
        side_effect=ValueError("cannot serialize"),
    ):
        response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    events = parse_sse(response.text)
    assert events[0][1]["cache_status"] == "failure"
    assert events[-2:][0][0] == "sources"
    assert events[-1][0] == "done"
    assert hybrid.search.call_count == len(provider.calls) == 1
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_partial_provider_failure_and_grounding_failure_are_not_cached(
    client, stream_cache_setup
) -> None:
    cache, _, _, _, provider, _ = stream_cache_setup
    provider.events = (LLMStreamEvent(text="Partial [S1]."),)
    provider.error = LLMRequestError("failure")
    failed_provider = await client.post("/api/v1/ask/stream", json={"question": "Provider"})
    provider.error = None
    provider.events = (LLMStreamEvent(text="Unsupported uncited claim."),)
    failed_grounding = await client.post("/api/v1/ask/stream", json={"question": "Grounding"})

    assert parse_sse(failed_provider.text)[-1][0] == "error"
    assert parse_sse(failed_grounding.text)[-1][0] == "error"
    assert cache.set_calls == []


@pytest.mark.anyio
async def test_ask_cache_entry_is_replayed_by_stream(client, stream_cache_setup) -> None:
    _, _, _, hybrid, provider, _ = stream_cache_setup
    ask_rag = Mock()
    ask_rag.generate = AsyncMock(return_value=generated_result())
    app.dependency_overrides[get_rag_generation_service] = lambda: ask_rag
    non_stream = await client.post("/api/v1/ask", json={"question": "Shared"})
    hybrid.reset_mock()

    stream = await client.post("/api/v1/ask/stream", json={"question": "Shared"})

    events = parse_sse(stream.text)
    assert "".join(data["text"] for name, data in events if name == "delta") == non_stream.json()["answer"]
    assert events[-2][1]["sources"] == non_stream.json()["sources"]
    hybrid.search.assert_not_called()
    assert provider.calls == []


@pytest.mark.anyio
async def test_stream_cache_entry_is_returned_by_ask(client, stream_cache_setup) -> None:
    _, _, _, hybrid, provider, _ = stream_cache_setup
    streamed = await client.post("/api/v1/ask/stream", json={"question": "Shared"})
    events = parse_sse(streamed.text)
    hybrid.reset_mock()
    blocking_rag = Mock()
    blocking_rag.generate = AsyncMock(side_effect=AssertionError("cache hit must bypass generation"))
    app.dependency_overrides[get_rag_generation_service] = lambda: blocking_rag

    non_stream = await client.post("/api/v1/ask", json={"question": "Shared"})

    assert non_stream.status_code == 200
    assert non_stream.json()["answer"] == "".join(
        data["text"] for name, data in events if name == "delta"
    )
    assert non_stream.json()["sources"] == events[-2][1]["sources"]
    hybrid.search.assert_not_called()
    blocking_rag.generate.assert_not_awaited()
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_stream_model_change_does_not_replay_incompatible_entry(
    client, stream_cache_setup
) -> None:
    cache, fingerprint, _, hybrid, provider, _ = stream_cache_setup
    await client.post("/api/v1/ask/stream", json={"question": "Question"})
    changed = RAGResponseCacheCoordinator(
        cache=cache,
        settings=Settings(redis_enabled=True, groq_model="different-model"),
        fingerprint_provider=fingerprint,
    )
    app.dependency_overrides[get_rag_response_cache] = lambda: changed

    await client.post("/api/v1/ask/stream", json={"question": "Question"})

    assert cache.get_calls[0] != cache.get_calls[1]
    assert hybrid.search.call_count == len(provider.calls) == 2


@pytest.mark.anyio
async def test_validated_citation_free_insufficiency_is_cached_and_replayed(
    client, stream_cache_setup
) -> None:
    cache, _, _, hybrid, provider, _ = stream_cache_setup
    provider.events = (
        LLMStreamEvent(text="The retrieved evidence is insufficient to answer this question."),
        LLMStreamEvent(model="fake-model", prompt_tokens=77, completion_tokens=8),
    )

    first = await client.post("/api/v1/ask/stream", json={"question": "Unknown"})
    second = await client.post("/api/v1/ask/stream", json={"question": "Unknown"})

    assert parse_sse(first.text)[-1][1]["citations"] == []
    assert parse_sse(second.text)[-1][1]["citations"] == []
    assert hybrid.search.call_count == len(provider.calls) == len(cache.set_calls) == 1
