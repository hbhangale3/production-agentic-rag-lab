import asyncio
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.config import Settings
from src.schemas.parsed_pdf import ParsedPDF
from src.services.agent import AgentExecutionEvent, AgentExecutionMetadata, AgentExecutionStatus, TerminalReason
from src.services.agent.live_documents import LiveDocumentProcessingService
from src.services.agent.live_search import ArxivLiveSearchService
from src.services.agent.service import AgentService
from src.services.agent.status import (
    CACHE_HIT_STATUS,
    SEMANTIC_OUTCOME_MESSAGES,
    status_message,
    to_public_status,
)
from src.services.agent.wiring import LazyPDFParser, agent_graph_config_from_settings, make_agent_service
from src.services.observability import NoOpObservabilityProvider
from src.services.prompts import LocalPromptResolver

S = AgentExecutionStatus


def event(status: AgentExecutionStatus, **metadata) -> AgentExecutionEvent:
    return AgentExecutionEvent(status=status, sequence=0, metadata=AgentExecutionMetadata(**metadata))


def test_every_execution_status_has_deterministic_safe_text() -> None:
    messages = {status: status_message(event(status)) for status in S}

    assert all(isinstance(message, str) and message and message != "Working" for message in messages.values())
    assert messages[S.GUARDRAIL_PASSED] == "Guardrail passed"
    assert messages[S.EVIDENCE_INSUFFICIENT] == "Evidence insufficient"
    assert messages[S.QUERY_REWRITTEN] == "Query rewritten"
    assert messages[S.LIVE_FALLBACK_STARTED] == "Live arXiv fallback used"
    assert messages[S.GROUNDING_PASSED] == "Grounding validation passed"
    assert messages[S.GRAPH_FAILED] == "The request could not be completed"
    assert {status_message(event(status)) for status in S} == set(messages.values())
    for message in messages.values():
        assert not any(term in message.lower() for term in ("because", "reason", "score", "error:", "exception"))


@pytest.mark.parametrize(
    ("status", "metadata", "expected"),
    [
        (S.LOCAL_RETRIEVAL_STARTED, {"retrieval_attempt": 1}, "Searching the local corpus"),
        (S.LOCAL_RETRIEVAL_STARTED, {"retrieval_attempt": 2}, "Retrying local retrieval"),
        (S.LOCAL_RETRIEVAL_COMPLETED, {"retrieval_attempt": 1, "source_count": 5}, "Local retrieval completed"),
        (S.LOCAL_RETRIEVAL_COMPLETED, {"retrieval_attempt": 2, "source_count": 5}, "Local retry completed"),
        (S.LIVE_FALLBACK_COMPLETED, {"candidate_count": 5}, "Live arXiv search found 5 candidate papers"),
        (S.LIVE_FALLBACK_COMPLETED, {"candidate_count": 1}, "Live arXiv search found 1 candidate paper"),
        (S.LIVE_FALLBACK_COMPLETED, {"candidate_count": 0}, "Live arXiv search found 0 candidate papers"),
        (S.LIVE_SELECTION_COMPLETED, {"selected_count": 2}, "2 live papers selected"),
        (S.LIVE_SELECTION_COMPLETED, {"selected_count": 1}, "1 live paper selected"),
        (S.LIVE_DOCUMENT_PROCESSING_COMPLETED, {"processed_count": 1}, "Live document processing completed (1 paper usable)"),
        (S.EVIDENCE_RERANK_COMPLETED, {"final_source_count": 5}, "Evidence reranked (5 sources selected)"),
        (S.GENERATION_STARTED, {"generation_attempt": 1}, "Generating the answer"),
        (S.GENERATION_STARTED, {"generation_attempt": 2}, "Regenerating the answer"),
        (S.GENERATION_COMPLETED, {"generation_attempt": 1}, "Answer generated"),
        (S.GENERATION_COMPLETED, {"generation_attempt": 2}, "Answer regenerated"),
    ],
)
def test_count_and_attempt_aware_status_text(status, metadata: dict, expected: str) -> None:
    assert status_message(event(status, **metadata)) == expected


def test_public_status_carries_only_code_and_message() -> None:
    status = to_public_status(event(S.EVIDENCE_INSUFFICIENT, evidence_score=12, source_count=5, retrieval_attempt=1))

    assert status.model_dump() == {"code": "evidence_insufficient", "message": "Evidence insufficient"}
    assert "12" not in status.model_dump_json()
    assert CACHE_HIT_STATUS.model_dump() == {"code": "cache_hit", "message": "Validated cached answer found"}


def test_semantic_outcome_messages_are_fixed_and_cover_exactly_the_semantic_terminals() -> None:
    assert set(SEMANTIC_OUTCOME_MESSAGES) == {
        TerminalReason.OUT_OF_SCOPE,
        TerminalReason.INSUFFICIENT_EVIDENCE,
        TerminalReason.GROUNDING_FAILED,
    }
    assert "enough reliable evidence" in SEMANTIC_OUTCOME_MESSAGES[TerminalReason.INSUFFICIENT_EVIDENCE]
    assert "grounding validation" in SEMANTIC_OUTCOME_MESSAGES[TerminalReason.GROUNDING_FAILED]
    assert len(set(SEMANTIC_OUTCOME_MESSAGES.values())) == 3


def wired_service(**overrides) -> tuple[AgentService, dict]:
    settings = Settings(_env_file=None, debug=False, **overrides)
    parts = {
        "llm_provider": Mock(),
        "hybrid_search_service": Mock(),
        "embedding_provider": Mock(),
        "rag_generation_service": Mock(),
        "cache": Mock(),
    }
    service = make_agent_service(settings=settings, observability=NoOpObservabilityProvider(), **parts)
    return service, parts


@pytest.mark.anyio
async def test_wiring_reuses_worker_services_and_adds_only_bounded_live_services() -> None:
    service, parts = wired_service(
        rag_retrieval_size=7,
        agent_request_timeout_seconds=90,
        agent_live_arxiv_timeout_seconds=9,
        agent_live_arxiv_max_retries=0,
        agent_live_paper_timeout_seconds=200,
        langfuse_timeout_seconds=3,
    )
    deps = service.dependencies

    assert deps.llm_provider is parts["llm_provider"]
    assert deps.hybrid_search_service is parts["hybrid_search_service"]
    assert deps.embedding_provider is parts["embedding_provider"]
    assert deps.rag_generation_service is parts["rag_generation_service"]
    assert deps.evidence_context_builder is parts["rag_generation_service"].evidence_builder
    assert service.response_cache.cache is parts["cache"]
    assert isinstance(deps.live_search_service, ArxivLiveSearchService)
    assert isinstance(deps.live_document_processor, LiveDocumentProcessingService)
    live_client = deps.live_search_service.arxiv_client
    assert deps.live_document_processor.acquisition_service.arxiv_client is live_client
    assert (live_client.max_retries, live_client.max_results) == (0, 5)
    assert live_client._client.timeout.read == 9
    assert isinstance(deps.live_document_processor.pdf_parser, LazyPDFParser)
    assert deps.live_document_processor.paper_timeout_seconds == 200
    assert deps.llm_telemetry.provider_name == "groq" and deps.llm_telemetry.cost_rates is None
    assert service.graph_config.retrieval_size == 7
    assert agent_graph_config_from_settings(Settings(_env_file=None)).retrieval_size == 5
    assert (service.request_timeout_seconds, service.prompt_timeout_seconds) == (90, 3)
    assert isinstance(service.prompt_resolver, LocalPromptResolver)
    assert service.response_cache.enabled is False
    await service.aclose()
    assert live_client._client.is_closed


def test_wiring_does_not_import_docling_or_load_models() -> None:
    for module in [name for name in sys.modules if name == "docling" or name.startswith("docling.")]:
        del sys.modules[module]

    wired_service()

    assert "docling" not in sys.modules


@pytest.mark.anyio
async def test_lazy_parser_is_created_once_on_first_use_and_parses_one_pdf_at_a_time() -> None:
    running = 0
    peak = 0

    class SlowParser:
        async def parse_pdf(self, path) -> ParsedPDF:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return ParsedPDF(raw_text=str(path))

    parser = LazyPDFParser(Settings(_env_file=None))
    with patch.object(LazyPDFParser, "_make_parser", return_value=SlowParser()) as make_parser:
        assert make_parser.call_count == 0
        results = await asyncio.gather(*(parser.parse_pdf(f"paper-{index}.pdf") for index in range(4)))

    assert make_parser.call_count == 1
    assert peak == 1
    assert [result.raw_text for result in results] == [f"paper-{index}.pdf" for index in range(4)]


@pytest.mark.anyio
async def test_lazy_parser_propagates_parser_errors_and_stays_usable() -> None:
    inner = Mock()
    inner.parse_pdf = AsyncMock(side_effect=[RuntimeError("bad pdf"), ParsedPDF(raw_text="ok")])
    parser = LazyPDFParser(Settings(_env_file=None))

    with patch.object(LazyPDFParser, "_make_parser", return_value=inner):
        with pytest.raises(RuntimeError):
            await parser.parse_pdf("a.pdf")
        assert (await parser.parse_pdf("b.pdf")).raw_text == "ok"


@pytest.mark.parametrize(
    "overrides",
    [
        {"agent_request_timeout_seconds": 0},
        {"agent_request_timeout_seconds": 901},
        {"agent_live_paper_timeout_seconds": 0},
        {"agent_live_paper_timeout_seconds": 601},
        {"agent_live_arxiv_timeout_seconds": 0},
        {"agent_live_arxiv_max_retries": 4},
    ],
)
def test_agent_runtime_settings_are_bounded(overrides: dict) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)
    defaults = Settings(_env_file=None)
    assert (defaults.agent_request_timeout_seconds, defaults.agent_live_arxiv_timeout_seconds) == (540.0, 15.0)
    assert defaults.agent_live_paper_timeout_seconds == 240.0
    # Two papers at the per-paper limit must still fit inside the request deadline.
    assert 2 * defaults.agent_live_paper_timeout_seconds < defaults.agent_request_timeout_seconds
    assert defaults.agent_live_arxiv_max_retries == 1
