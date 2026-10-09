import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.dependencies import get_hybrid_search_service, get_rag_generation_service
from src.exceptions import HybridSearchError, LLMRequestError, RAGPromptBudgetError
from src.main import app
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.evidence import EvidenceContextBuilder
from src.services.llm.base import LLMStreamEvent
from src.services.rag import RAGGenerationService


class FakeStreamingProvider:
    def __init__(self, events=(), error: Exception | None = None):
        self.events = events
        self.error = error
        self.calls = []

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append((messages, temperature, max_tokens))
        for event in self.events:
            yield event
        if self.error:
            raise self.error

    async def complete(self, messages, *, temperature=None, max_tokens=None):
        raise AssertionError("stream endpoint must not call complete")

    async def close(self):
        return None


def retrieval(*, empty=False):
    results = (
        []
        if empty
        else [
            HybridSearchHit(
                chunk_id="2401.00001::chunk::000",
                arxiv_id="2401.00001",
                chunk_index=0,
                section_title="Results",
                chunk_text="Evidence with Unicode café.",
                word_count=4,
                paper_title="Grounded Paper",
                authors=["Researcher"],
                categories=["cs.AI"],
                rrf_score=0.03,
                bm25_rank=1,
                vector_rank=1,
            )
        ]
    )
    return HybridSearchResult(query="Question", retrieval_mode="hybrid", count=len(results), results=results)


def parse_sse(body):
    parsed = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))
    return parsed


@pytest.fixture
def streaming_services():
    hybrid = Mock()
    hybrid.search.return_value = retrieval()
    provider = FakeStreamingProvider(
        (
            LLMStreamEvent(text="Grounded ", model="fake-model"),
            LLMStreamEvent(text="café 【"),
            LLMStreamEvent(text="S1】."),
            LLMStreamEvent(model="fake-model", prompt_tokens=80, completion_tokens=9),
        )
    )
    rag = RAGGenerationService(
        llm_provider=provider,
        evidence_builder=EvidenceContextBuilder(max_context_tokens=1000),
    )
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid
    app.dependency_overrides[get_rag_generation_service] = lambda: rag
    yield hybrid, provider, rag
    app.dependency_overrides.pop(get_hybrid_search_service, None)
    app.dependency_overrides.pop(get_rag_generation_service, None)


@pytest.mark.anyio
async def test_stream_endpoint_retrieves_once_and_emits_ordered_json_sse(client, streaming_services):
    hybrid, provider, _ = streaming_services

    async def execute(callable_):
        return callable_()

    with patch("src.routers.ask.run_in_threadpool", new=AsyncMock(side_effect=execute)) as pool:
        response = await client.post("/api/v1/ask/stream", json={"question": "Research?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "metadata"
    assert names[-2:] == ["sources", "done"]
    assert names[1:-2] == ["delta", "delta", "delta"]
    assert events[0][1]["retrieval_mode"] == "hybrid"
    assert events[0][1]["source_count"] == 1
    assert events[0][1]["cache_status"] == "bypass"
    assert "".join(data["text"] for name, data in events if name == "delta") == ("Grounded café [S1].")
    assert events[-2][1]["sources"][0]["content"] == "Evidence with Unicode café."
    assert events[-1][1] == {
        "model": "fake-model",
        "prompt_tokens": 80,
        "completion_tokens": 9,
        "citations": ["[S1]"],
    }
    pool.assert_awaited_once()
    hybrid.search.assert_called_once_with("Research?", size=5)
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_empty_evidence_is_successful_sse_without_provider(client, streaming_services):
    hybrid, provider, _ = streaming_services
    hybrid.search.return_value = retrieval(empty=True)

    response = await client.post("/api/v1/ask/stream", json={"question": "Unknown"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["metadata", "delta", "sources", "done"]
    assert events[0][1]["source_count"] == 0
    assert events[-2][1] == {"sources": []}
    assert events[-1][1]["model"] is None
    assert provider.calls == []


@pytest.mark.anyio
async def test_stream_failure_is_safe_terminal_error_without_done(client, streaming_services):
    _, provider, _ = streaming_services
    provider.events = (LLMStreamEvent(text="Partial secret-safe text"),)
    provider.error = LLMRequestError("credential raw-provider-body")

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["metadata", "delta", "error"]
    assert events[-1][1] == {
        "code": "generation_failed",
        "message": "Language model generation failed.",
    }
    assert "credential" not in response.text
    assert "raw-provider-body" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status_code"),
    [(HybridSearchError("private"), 503), (RAGPromptBudgetError("private"), 422)],
)
async def test_pre_stream_failures_retain_http_status(client, streaming_services, error, status_code):
    hybrid, _, rag = streaming_services
    if isinstance(error, HybridSearchError):
        hybrid.search.side_effect = error
    else:
        rag.prepare = Mock(side_effect=error)

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert "private" not in response.text
