from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from src.config import Settings
from src.dependencies import (
    get_hybrid_search_service,
    get_observability_provider,
    get_rag_generation_service,
    get_rag_response_cache,
)
from src.exceptions import LLMRequestError
from src.main import app
from src.routers.ask import stream_ask_question
from src.schemas.ask import AskRequest
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.cache import RAGResponseCacheCoordinator
from src.services.evidence import EvidenceContextBuilder
from src.services.llm.base import LLMStreamEvent
from src.services.rag import RAGGenerationService

from tests.api.test_ask_cache import MemoryCache, MutableFingerprint
from tests.api.test_ask_stream import parse_sse

PRIVATE_QUESTION = "PRIVATE_QUESTION_SENTINEL PRIVATE_PROMPT_SENTINEL"
PRIVATE_EVIDENCE = "PRIVATE_EVIDENCE_SENTINEL"
PRIVATE_ANSWER = "PRIVATE_ANSWER_SENTINEL [S1]."


class RecordingObservation:
    def __init__(self, name: str, metadata=None) -> None:
        self.name = name
        self.metadata = [metadata or {}]
        self.error_types: list[str] = []
        self.children: list[RecordingObservation] = []
        self.end_count = 0

    def start_span(self, *, name: str, metadata=None):
        child = RecordingObservation(name, metadata)
        self.children.append(child)
        return child

    def update(self, *, metadata=None, error_type=None) -> None:
        if metadata:
            self.metadata.append(metadata)
        if error_type:
            self.error_types.append(error_type)

    def end(self) -> None:
        self.end_count += 1


class RecordingProvider:
    capture_content = False

    def __init__(self) -> None:
        self.roots: list[RecordingObservation] = []

    def start_trace(self, *, name: str, metadata=None):
        root = RecordingObservation(name, metadata)
        self.roots.append(root)
        return root


class FakeLLM:
    def __init__(self) -> None:
        self.answer = PRIVATE_ANSWER
        self.stream_parts = ("PRIVATE_ANSWER_", "SENTINEL [S1].")
        self.complete_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages, *, temperature=None, max_tokens=None):
        self.complete_calls += 1
        if self.complete_error:
            raise self.complete_error
        return SimpleNamespace(
            content=self.answer,
            model="observed-model",
            prompt_tokens=41,
            completion_tokens=7,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.stream_calls += 1
        for part in self.stream_parts:
            yield LLMStreamEvent(text=part, model="observed-model")
        if self.stream_error:
            raise self.stream_error
        yield LLMStreamEvent(model="observed-model", prompt_tokens=41, completion_tokens=7)

    async def close(self):
        return None


def retrieval() -> HybridSearchResult:
    hit = HybridSearchHit(
        chunk_id="paper::chunk::000",
        arxiv_id="2401.00001",
        chunk_index=0,
        section_title="Results",
        chunk_text=PRIVATE_EVIDENCE,
        word_count=1,
        paper_title="Private-free operational title",
        authors=["Researcher"],
        categories=["cs.AI"],
        rrf_score=0.03,
        bm25_rank=1,
        vector_rank=1,
    )
    return HybridSearchResult(query=PRIVATE_QUESTION, retrieval_mode="hybrid", count=1, results=[hit])


def all_observations(root: RecordingObservation):
    yield root
    for child in root.children:
        yield from all_observations(child)


def child_names(root: RecordingObservation) -> list[str]:
    return [child.name for child in root.children]


@pytest.fixture
def traced_services():
    cache = MemoryCache()
    coordinator = RAGResponseCacheCoordinator(
        cache=cache,
        settings=Settings(redis_enabled=True),
        fingerprint_provider=MutableFingerprint(),
    )
    hybrid = Mock()
    hybrid.search.return_value = retrieval()
    llm = FakeLLM()
    rag = RAGGenerationService(
        llm_provider=llm,
        evidence_builder=EvidenceContextBuilder(max_context_tokens=1000),
    )
    observability = RecordingProvider()
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid
    app.dependency_overrides[get_rag_generation_service] = lambda: rag
    app.dependency_overrides[get_rag_response_cache] = lambda: coordinator
    app.dependency_overrides[get_observability_provider] = lambda: observability
    yield cache, coordinator, hybrid, llm, rag, observability
    for dependency in (
        get_hybrid_search_service,
        get_rag_generation_service,
        get_rag_response_cache,
        get_observability_provider,
    ):
        app.dependency_overrides.pop(dependency, None)


@pytest.mark.anyio
async def test_ask_miss_records_full_safe_trace_without_private_content(
    client, traced_services
) -> None:
    _, _, _, _, _, recording = traced_services

    response = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})

    assert response.status_code == 200
    root = recording.roots[0]
    assert root.name == "rag.request"
    assert child_names(root) == [
        "rag.cache.lookup",
        "rag.retrieval",
        "rag.evidence",
        "rag.generation",
        "rag.grounding",
        "rag.cache.write",
    ]
    assert root.end_count == 1
    assert any(item.get("status") == "success" for item in root.metadata)
    exported = repr([(item.name, item.metadata, item.error_types) for item in all_observations(root)])
    for sentinel in (PRIVATE_QUESTION, "PRIVATE_PROMPT_SENTINEL", PRIVATE_EVIDENCE, PRIVATE_ANSWER):
        assert sentinel not in exported


@pytest.mark.anyio
async def test_ask_cache_hit_records_only_lookup_and_skips_expensive_work(
    client, traced_services
) -> None:
    _, _, hybrid, llm, _, recording = traced_services
    first = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})
    hybrid.reset_mock()
    recording.roots.clear()

    second = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})

    assert first.json() == second.json()
    root = recording.roots[0]
    assert child_names(root) == ["rag.cache.lookup"]
    assert any(item.get("cache_outcome") == "hit" for item in root.metadata)
    assert any("artifact_prompt_tokens" in item for item in root.metadata)
    hybrid.search.assert_not_called()
    assert llm.complete_calls == 1


@pytest.mark.anyio
async def test_cache_read_failure_is_traced_but_root_still_succeeds(client, traced_services) -> None:
    cache, _, _, _, _, recording = traced_services
    cache.get = AsyncMock(side_effect=RuntimeError("PRIVATE_CACHE_ERROR_SENTINEL"))

    response = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})

    root = recording.roots[0]
    lookup = root.children[0]
    assert response.status_code == 200
    assert lookup.error_types == ["cache_read_failure"]
    assert any(item.get("cache_outcome") == "failure" for item in root.metadata)
    assert any(item.get("status") == "success" for item in root.metadata)
    assert "PRIVATE_CACHE_ERROR_SENTINEL" not in repr(root.metadata)


@pytest.mark.anyio
async def test_cache_write_failure_is_child_error_but_root_success(client, traced_services) -> None:
    cache, _, _, _, _, recording = traced_services
    cache.set_result = False

    response = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})

    root = recording.roots[0]
    write = next(child for child in root.children if child.name == "rag.cache.write")
    assert response.status_code == 200
    assert write.error_types == ["cache_write_failure"]
    assert root.error_types == []
    assert any(item.get("status") == "success" for item in root.metadata)


@pytest.mark.anyio
async def test_empty_evidence_is_success_without_generation_or_grounding(
    client, traced_services
) -> None:
    _, _, hybrid, llm, _, recording = traced_services
    hybrid.search.return_value = HybridSearchResult(
        query=PRIVATE_QUESTION,
        retrieval_mode="hybrid",
        count=0,
        results=[],
    )

    response = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})

    root = recording.roots[0]
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert child_names(root) == [
        "rag.cache.lookup",
        "rag.retrieval",
        "rag.evidence",
        "rag.cache.write",
    ]
    assert any(item.get("valid_insufficiency") is True for item in root.metadata)
    assert llm.complete_calls == 0


@pytest.mark.anyio
async def test_generation_and_grounding_failures_are_safely_classified(
    client, traced_services
) -> None:
    _, _, _, llm, _, recording = traced_services
    llm.complete_error = LLMRequestError("PRIVATE_PROVIDER_ERROR_SENTINEL")
    generation_failure = await client.post("/api/v1/ask", json={"question": "Generation"})
    generation_root = recording.roots[-1]
    llm.complete_error = None
    llm.answer = "PRIVATE_UNCITED_ANSWER_SENTINEL"
    grounding_failure = await client.post("/api/v1/ask", json={"question": "Grounding"})
    grounding_root = recording.roots[-1]

    assert generation_failure.status_code == 503
    assert grounding_failure.status_code == 502
    assert "generation_failure" in generation_root.error_types
    assert "grounding_failure" in grounding_root.error_types
    assert generation_root.children[-1].name == "rag.generation"
    assert generation_root.children[-1].error_types == ["LLMRequestError"]
    grounding = next(child for child in grounding_root.children if child.name == "rag.grounding")
    assert grounding.error_types == ["GroundingValidationError"]
    assert "PRIVATE_PROVIDER_ERROR_SENTINEL" not in repr(generation_root.metadata)
    assert "PRIVATE_UNCITED_ANSWER_SENTINEL" not in repr(grounding_root.metadata)


@pytest.mark.anyio
async def test_successful_stream_has_one_generation_span_and_closes_at_done(
    client, traced_services
) -> None:
    _, _, _, llm, _, recording = traced_services

    response = await client.post("/api/v1/ask/stream", json={"question": PRIVATE_QUESTION})

    assert parse_sse(response.text)[-1][0] == "done"
    root = recording.roots[0]
    assert child_names(root).count("rag.generation") == 1
    assert child_names(root) == [
        "rag.cache.lookup",
        "rag.retrieval",
        "rag.evidence",
        "rag.generation",
        "rag.grounding",
        "rag.cache.write",
    ]
    assert root.end_count == 1
    assert llm.stream_calls == 1


@pytest.mark.anyio
async def test_stream_cache_hit_trace_has_no_fake_work(client, traced_services) -> None:
    _, _, hybrid, llm, _, recording = traced_services
    await client.post("/api/v1/ask/stream", json={"question": PRIVATE_QUESTION})
    recording.roots.clear()
    hybrid.reset_mock()

    response = await client.post("/api/v1/ask/stream", json={"question": PRIVATE_QUESTION})

    assert parse_sse(response.text)[-1][0] == "done"
    assert child_names(recording.roots[0]) == ["rag.cache.lookup"]
    assert recording.roots[0].end_count == 1
    hybrid.search.assert_not_called()
    assert llm.stream_calls == 1


@pytest.mark.anyio
async def test_stream_provider_and_grounding_failures_close_error_traces(
    client, traced_services
) -> None:
    cache, _, _, llm, _, recording = traced_services
    llm.stream_error = LLMRequestError("PRIVATE_PARTIAL_ERROR_SENTINEL")
    provider_failure = await client.post("/api/v1/ask/stream", json={"question": "Provider"})
    provider_root = recording.roots[-1]
    llm.stream_error = None
    llm.stream_parts = ("PRIVATE_UNCITED_STREAM_SENTINEL",)
    grounding_failure = await client.post("/api/v1/ask/stream", json={"question": "Grounding"})
    grounding_root = recording.roots[-1]

    assert parse_sse(provider_failure.text)[-1][0] == "error"
    assert parse_sse(grounding_failure.text)[-1][0] == "error"
    assert "generation_failure" in provider_root.error_types
    assert "grounding_failure" in grounding_root.error_types
    assert [child.name for child in provider_root.children].count("rag.generation") == 1
    assert not any(child.name == "rag.cache.write" for child in provider_root.children)
    assert not any(child.name == "rag.cache.write" for child in grounding_root.children)
    assert cache.set_calls == []
    exported = repr([root.metadata for root in (provider_root, grounding_root)])
    assert "PRIVATE_PARTIAL_ERROR_SENTINEL" not in exported
    assert "PRIVATE_UNCITED_STREAM_SENTINEL" not in exported


@pytest.mark.anyio
async def test_observability_provider_failure_does_not_change_rag_response(
    client, traced_services
) -> None:
    failing = Mock()
    failing.capture_content = False
    failing.start_trace.side_effect = RuntimeError("telemetry unavailable")
    app.dependency_overrides[get_observability_provider] = lambda: failing

    response = await client.post("/api/v1/ask", json={"question": PRIVATE_QUESTION})

    assert response.status_code == 200
    assert response.json()["answer"] == PRIVATE_ANSWER

    broken_observation = Mock()
    broken_observation.start_span.side_effect = RuntimeError("span unavailable")
    broken_observation.update.side_effect = RuntimeError("update unavailable")
    broken_observation.end.side_effect = RuntimeError("completion unavailable")
    failing.start_trace.side_effect = None
    failing.start_trace.return_value = broken_observation

    second = await client.post("/api/v1/ask", json={"question": "Another safe question"})

    assert second.status_code == 200
    assert second.json()["answer"] == PRIVATE_ANSWER


@pytest.mark.anyio
async def test_stream_generator_close_marks_trace_cancelled_without_cache_write(
    traced_services,
) -> None:
    cache, coordinator, hybrid, _, rag, recording = traced_services
    response = await stream_ask_question(
        request=AskRequest(question=PRIVATE_QUESTION),
        hybrid_service=hybrid,
        rag_service=rag,
        response_cache=coordinator,
        settings=Settings(redis_enabled=True),
        observability=recording,
    )
    iterator = response.body_iterator
    await anext(iterator)  # metadata
    await anext(iterator)  # first provider delta

    await iterator.aclose()

    root = recording.roots[0]
    generation = next(child for child in root.children if child.name == "rag.generation")
    assert "cancelled" in root.error_types
    assert "cancelled" in generation.error_types
    assert root.end_count == 1
    assert cache.set_calls == []
