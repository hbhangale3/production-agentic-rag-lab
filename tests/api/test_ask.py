from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.dependencies import get_hybrid_search_service, get_rag_generation_service
from src.exceptions import (
    HybridSearchError,
    InsufficientEvidenceError,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    RAGPromptBudgetError,
)
from src.main import app
from src.schemas.hybrid_search import HybridSearchResult
from src.services.evidence import EvidenceContextBuilder, EvidenceSource
from src.services.rag import RAGGenerationResult, RAGGenerationService


class NoCallProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, temperature=None, max_tokens=None):
        self.calls += 1
        raise AssertionError("provider must not be called for empty evidence")

    async def close(self) -> None:
        return None


def empty_retrieval() -> HybridSearchResult:
    return HybridSearchResult(
        query="question",
        retrieval_mode="hybrid",
        count=0,
        results=[],
    )


def generation() -> RAGGenerationResult:
    published = datetime(2024, 1, 2, tzinfo=UTC)
    sources = tuple(
        EvidenceSource(
            label=f"[S{rank}]",
            retrieval_rank=rank,
            arxiv_id=f"2401.0000{rank}",
            chunk_id=f"paper-{rank}::chunk::000",
            chunk_index=0,
            paper_title=f"Paper {rank}",
            section_title="Results",
            content=f"Selected evidence {rank}.",
            authors=("Author",),
            categories=("cs.AI",),
            published_date=published,
            rrf_score=0.03,
            bm25_rank=rank,
            vector_rank=rank,
            bm25_score=4.0,
            vector_score=0.8,
            truncated=False,
        )
        for rank in (1, 2)
    )
    return RAGGenerationResult(
        answer="Grounded answer [S1].",
        sources=sources,
        retrieval_mode="hybrid",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=12,
        cited_labels=("[S1]",),
        estimated_prompt_tokens=95,
        token_counting="estimated",
    )


@pytest.fixture
def ask_services():
    hybrid = Mock()
    hybrid.search.return_value = empty_retrieval()
    rag = Mock()
    rag.generate = AsyncMock(return_value=generation())
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid
    app.dependency_overrides[get_rag_generation_service] = lambda: rag
    yield hybrid, rag
    app.dependency_overrides.pop(get_hybrid_search_service, None)
    app.dependency_overrides.pop(get_rag_generation_service, None)


@pytest.mark.anyio
async def test_grounded_ask_offloads_retrieval_and_maps_public_response(
    client, ask_services
) -> None:
    hybrid, rag = ask_services

    async def execute(callable_):
        return callable_()

    with patch("src.routers.ask.run_in_threadpool", new=AsyncMock(side_effect=execute)) as pool:
        response = await client.post("/api/v1/ask", json={"question": "Research question"})

    assert response.status_code == 200
    pool.assert_awaited_once()
    hybrid.search.assert_called_once_with("Research question", size=5)
    rag.generate.assert_awaited_once_with(
        question="Research question",
        retrieval_result=hybrid.search.return_value,
    )
    body = response.json()
    assert body["answer"] == "Grounded answer [S1]."
    assert body["retrieval_mode"] == "hybrid"
    assert body["model"] == "test-model"
    assert body["prompt_tokens"] == 100
    assert body["completion_tokens"] == 12
    assert [item["citation"] for item in body["sources"]] == ["[S1]", "[S2]"]
    assert [item["chunk_id"] for item in body["sources"]] == [
        "paper-1::chunk::000",
        "paper-2::chunk::000",
    ]


@pytest.mark.anyio
async def test_blank_question_is_rejected_before_services(client, ask_services) -> None:
    hybrid, rag = ask_services

    response = await client.post("/api/v1/ask", json={"question": "   "})

    assert response.status_code == 422
    hybrid.search.assert_not_called()
    rag.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_empty_evidence_returns_200_without_provider_call(client, ask_services) -> None:
    hybrid, _ = ask_services
    provider = NoCallProvider()
    rag = RAGGenerationService(
        llm_provider=provider,
        evidence_builder=EvidenceContextBuilder(max_context_tokens=1000),
    )
    app.dependency_overrides[get_rag_generation_service] = lambda: rag

    response = await client.post("/api/v1/ask", json={"question": "Unknown topic"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The available indexed evidence is insufficient to answer this question.",
        "sources": [],
        "retrieval_mode": "hybrid",
        "model": None,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    assert hybrid.search.call_count == 1
    assert provider.calls == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (HybridSearchError("backend-url secret"), 503, "Research retrieval is temporarily unavailable."),
        (RAGPromptBudgetError("full private prompt"), 422, "exceed the generation budget"),
        (LLMConfigurationError("GROQ_API_KEY secret"), 503, "not configured"),
        (LLMRequestError("provider secret response"), 503, "temporarily unavailable"),
        (LLMResponseError("malformed secret answer"), 502, "unusable response"),
    ],
)
async def test_domain_failures_map_to_safe_http_responses(
    client,
    ask_services,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    hybrid, rag = ask_services
    if isinstance(error, HybridSearchError):
        hybrid.search.side_effect = error
    else:
        rag.generate.side_effect = error

    response = await client.post("/api/v1/ask", json={"question": "Research question"})

    assert response.status_code == status_code
    assert detail in response.json()["detail"]
    serialized = response.text
    assert "secret" not in serialized
    assert "private prompt" not in serialized


@pytest.mark.anyio
async def test_worker_scoped_services_are_reused_across_requests(client, ask_services) -> None:
    hybrid, rag = ask_services

    first = await client.post("/api/v1/ask", json={"question": "First"})
    second = await client.post("/api/v1/ask", json={"question": "Second"})

    assert first.status_code == second.status_code == 200
    assert hybrid.search.call_count == 2
    assert rag.generate.await_count == 2


@pytest.mark.anyio
async def test_missing_runtime_llm_configuration_is_safe_503(client, ask_services) -> None:
    app.dependency_overrides.pop(get_rag_generation_service, None)
    app.state.rag_generation_service = None

    response = await client.post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Language model service is not configured."}
