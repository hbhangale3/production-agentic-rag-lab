import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.dependencies import get_hybrid_search_service, get_rag_generation_service
from src.main import app
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.evidence import EvidenceContextBuilder
from src.services.llm.base import LLMStreamEvent
from src.services.rag import RAGGenerationService


class MutableProvider:
    def __init__(self) -> None:
        self.answer = "Supported finding [S1]."
        self.stream_parts = ("Supported finding [S1].",)
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages, *, temperature=None, max_tokens=None):
        self.complete_calls += 1
        return SimpleNamespace(
            content=self.answer,
            model="fake-model",
            prompt_tokens=40,
            completion_tokens=8,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.stream_calls += 1
        for part in self.stream_parts:
            yield LLMStreamEvent(text=part, model="fake-model")
        yield LLMStreamEvent(model="fake-model", prompt_tokens=40, completion_tokens=8)

    async def close(self):
        return None


def retrieval() -> HybridSearchResult:
    hit = HybridSearchHit(
        chunk_id="paper::chunk::000",
        arxiv_id="2401.00001",
        chunk_index=0,
        section_title="Results",
        chunk_text="Selected evidence.",
        word_count=2,
        paper_title="Paper",
        authors=["Researcher"],
        categories=["cs.AI"],
        rrf_score=0.03,
        bm25_rank=1,
        vector_rank=1,
    )
    return HybridSearchResult(query="Question", retrieval_mode="hybrid", count=1, results=[hit])


def parse_sse(body: str):
    events = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


@pytest.fixture
def grounded_services():
    hybrid = Mock()
    hybrid.search.return_value = retrieval()
    provider = MutableProvider()
    rag = RAGGenerationService(
        llm_provider=provider,
        evidence_builder=EvidenceContextBuilder(max_context_tokens=1000),
    )
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid
    app.dependency_overrides[get_rag_generation_service] = lambda: rag
    yield provider
    app.dependency_overrides.pop(get_hybrid_search_service, None)
    app.dependency_overrides.pop(get_rag_generation_service, None)


@pytest.mark.anyio
async def test_non_streaming_valid_and_insufficiency_answers_succeed(client, grounded_services):
    provider = grounded_services
    valid = await client.post("/api/v1/ask", json={"question": "Question"})
    provider.answer = "The supplied sources do not provide enough information to answer this."
    insufficient = await client.post("/api/v1/ask", json={"question": "Question"})

    assert valid.status_code == insufficient.status_code == 200
    assert valid.json()["sources"][0]["citation"] == "[S1]"
    assert insufficient.json()["answer"].startswith("The supplied sources")
    assert provider.complete_calls == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        "",
        " \n ",
        "Unsupported [S99].",
        "Invalid [S0].",
        "Invalid [S].",
        "Invalid [Sabc].",
        "A substantive answer without attribution.",
        "Evidence is insufficient, but this treatment definitely works.",
    ],
)
async def test_non_streaming_grounding_failures_are_safe_502(client, grounded_services, answer):
    grounded_services.answer = answer

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Language model service returned an unusable response."}
    if answer.strip():
        assert answer not in response.text


@pytest.mark.anyio
async def test_streaming_citation_free_insufficiency_is_validated_success(client, grounded_services):
    grounded_services.stream_parts = ("The retrieved evidence is insufficient to answer this question.",)

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["metadata", "delta", "sources", "done"]
    assert events[-1][1]["citations"] == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "parts",
    [
        (),
        (" \n ",),
        ("Unknown [S99].",),
        ("Malformed [S].",),
        ("Invalid [S0].",),
        ("Substantive uncited answer.",),
    ],
)
async def test_streaming_grounding_failures_end_in_safe_error(client, grounded_services, parts):
    grounded_services.stream_parts = parts

    response = await client.post("/api/v1/ask/stream", json={"question": "Question"})

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "metadata"
    assert names[-1] == "error"
    assert "sources" not in names
    assert "done" not in names
    assert events[-1][1] == {
        "code": "grounding_validation_failed",
        "message": "The generated answer could not be validated against the supplied evidence.",
    }


@pytest.mark.anyio
async def test_provider_remains_usable_after_stream_validation_failure(client, grounded_services):
    provider = grounded_services
    provider.stream_parts = ("Uncited claim.",)
    failed = await client.post("/api/v1/ask/stream", json={"question": "First"})
    provider.stream_parts = ("Supported recovery [S1].",)
    recovered = await client.post("/api/v1/ask/stream", json={"question": "Second"})

    assert parse_sse(failed.text)[-1][0] == "error"
    assert parse_sse(recovered.text)[-1][0] == "done"
    assert provider.stream_calls == 2
