import asyncio
import dataclasses
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.exceptions import ArxivPDFDownloadError, IncompleteGenerationError, PDFNoTextError, PDFParserError
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.schemas.parsed_pdf import ParsedPDF
from src.services.agent import (
    AGENT_OBSERVATION_KEY,
    AGENT_PROMPT_DEFINITIONS,
    AgentErrorCategory,
    AgentEvidenceCandidate,
    AgentExecutionStatus,
    AgentGraphConfig,
    AgentGraphDependencies,
    AnswerGroundingPromptBuilder,
    AnswerGroundingResult,
    EvidenceGrade,
    LiveArxivPaper,
    LiveArxivSearchResult,
    LiveDocumentProcessingResult,
    LiveDocumentProcessingService,
    LivePaperAcquisitionService,
    LiveSearchError,
    TerminalReason,
    TransientLiveEvidenceChunk,
    build_agent_graph,
    create_initial_agent_state,
    local_agent_prompt_bundle,
    resolve_agent_prompt_bundle,
)
from src.services.agent.query_rewriter import MAX_REWRITTEN_QUERY_CHARACTERS
from src.services.arxiv.client import ArxivClient
from src.services.chunking import PaperChunkingService
from src.services.evidence import EvidenceContextBuilder
from src.services.llm.base import LLMCompletion
from src.services.observability import LLMTelemetry, TokenCostRates
from src.services.prompts import ResolvedPrompt, local_prompt
from src.services.rag import RAG_SYSTEM_PROMPT, RAGGenerationService
from src.services.search.hybrid_service import HybridSearchService

S = AgentExecutionStatus
EVIDENCE_TEXT = "private evidence text about clinical NLP"
INDIA_QUESTION = "What are the current healthcare issues in India and how can AI help solve them?"
REWRITTEN = "India healthcare challenges artificial intelligence health equity"
REWRITE_JSON = f'{{"query":"{REWRITTEN}"}}'
SINGLE_ATTEMPT = AgentGraphConfig(max_local_retrieval_attempts=1, live_fallback_enabled=False)
LOCAL_ONLY = AgentGraphConfig(live_fallback_enabled=False)

GUARDRAIL_PASSED = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GUARDRAIL_PASSED]
RETRIEVED = [*GUARDRAIL_PASSED, S.LOCAL_RETRIEVAL_STARTED, S.LOCAL_RETRIEVAL_COMPLETED]
RERANKED = [S.EVIDENCE_RERANK_STARTED, S.EVIDENCE_RERANK_COMPLETED]
GENERATED = [*RERANKED, S.GENERATION_STARTED, S.GENERATION_COMPLETED]
GROUNDED = [S.GROUNDING_STARTED, S.GROUNDING_PASSED]
UNGROUNDED = [S.GROUNDING_STARTED, S.GROUNDING_FAILED]
REGENERATED = [S.GENERATION_STARTED, S.GENERATION_COMPLETED]
GENERATION_TAIL = [*GENERATED, *GROUNDED, S.GRAPH_COMPLETED]
REGENERATE_PASS_TAIL = [*GENERATED, *UNGROUNDED, *REGENERATED, *GROUNDED, S.GRAPH_COMPLETED]
REGENERATE_FAIL_TAIL = [*GENERATED, *UNGROUNDED, *REGENERATED, *UNGROUNDED, S.GRAPH_COMPLETED]
RERANK_FAILED_TAIL = [S.EVIDENCE_RERANK_STARTED, S.GRAPH_FAILED]
GENERATION_FAILED_TAIL = [*RERANKED, S.GENERATION_STARTED, S.GRAPH_FAILED]
LOCAL_SUFFICIENT = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_SUFFICIENT]
SUFFICIENT_PATH = [*LOCAL_SUFFICIENT, *GENERATION_TAIL]
INSUFFICIENT_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT, S.GRAPH_COMPLETED]
GRADER_FAILED_PATH = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.GRAPH_FAILED]
REJECTED_PATH = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GUARDRAIL_REJECTED, S.GRAPH_COMPLETED]
GUARDRAIL_FAILED_PATH = [S.GRAPH_STARTED, S.GUARDRAIL_STARTED, S.GRAPH_FAILED]
RETRIEVAL_FAILED_PATH = [*GUARDRAIL_PASSED, S.LOCAL_RETRIEVAL_STARTED, S.GRAPH_FAILED]
FIRST_INSUFFICIENT = [*RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT]
REWRITTEN_PATH = [*FIRST_INSUFFICIENT, S.QUERY_REWRITE_STARTED, S.QUERY_REWRITTEN]
SECOND_RETRIEVED = [*REWRITTEN_PATH, S.LOCAL_RETRIEVAL_STARTED, S.LOCAL_RETRIEVAL_COMPLETED]
RETRY_SUFFICIENT_PATH = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_SUFFICIENT, *GENERATION_TAIL]
RETRY_INSUFFICIENT_PATH = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT, S.GRAPH_COMPLETED]
REWRITE_FAILED_PATH = [*FIRST_INSUFFICIENT, S.QUERY_REWRITE_STARTED, S.GRAPH_FAILED]
SECOND_RETRIEVAL_FAILED_PATH = [*REWRITTEN_PATH, S.LOCAL_RETRIEVAL_STARTED, S.GRAPH_FAILED]
SECOND_GRADER_FAILED_PATH = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.GRAPH_FAILED]
LOCAL_EXHAUSTED = [*SECOND_RETRIEVED, S.EVIDENCE_GRADING_STARTED, S.EVIDENCE_INSUFFICIENT]
LIVE_COMPLETED_PATH = [*LOCAL_EXHAUSTED, S.LIVE_FALLBACK_STARTED, S.LIVE_FALLBACK_COMPLETED, S.GRAPH_COMPLETED]
LIVE_FAILED_PATH = [*LOCAL_EXHAUSTED, S.LIVE_FALLBACK_STARTED, S.GRAPH_FAILED]
LIVE_SEARCHED = [*LOCAL_EXHAUSTED, S.LIVE_FALLBACK_STARTED, S.LIVE_FALLBACK_COMPLETED]
LIVE_SELECTED = [*LIVE_SEARCHED, S.LIVE_SELECTION_STARTED, S.LIVE_SELECTION_COMPLETED]
LIVE_DOCUMENT_TAIL = [S.LIVE_DOCUMENT_PROCESSING_STARTED, S.LIVE_DOCUMENT_PROCESSING_COMPLETED, S.GRAPH_COMPLETED]
LIVE_PROCESSED = [*LIVE_SELECTED, S.LIVE_DOCUMENT_PROCESSING_STARTED, S.LIVE_DOCUMENT_PROCESSING_COMPLETED]
LIVE_EVIDENCE_PATH = [*LIVE_PROCESSED, *GENERATION_TAIL]
LIVE_ZERO_SELECTED_PATH = [*LIVE_SELECTED, S.GRAPH_COMPLETED]
LIVE_SELECTION_FAILED_PATH = [*LIVE_SEARCHED, S.LIVE_SELECTION_STARTED, S.GRAPH_FAILED]
LIVE_DOCUMENTS_FAILED_PATH = [*LIVE_SELECTED, S.LIVE_DOCUMENT_PROCESSING_STARTED, S.GRAPH_FAILED]
SELECT_FIRST = '{"selected_arxiv_ids":["2501.00001"]}'
SELECT_TWO = '{"selected_arxiv_ids":["2501.00001","2501.00002"]}'
SELECT_NONE = '{"selected_arxiv_ids":[]}'
CHUNK_TEXT = "private live chunk text"
ANSWER = "Private generated synthesis [S1]."
REGENERATED_ANSWER = "Private regenerated synthesis [S1]."
GROUNDING_SYSTEM_PROMPT = AnswerGroundingPromptBuilder.SYSTEM_PROMPT


class Truncated(str):
    """A scripted answer the provider reports as cut off at the completion-token limit."""


class Stopped(str):
    """A scripted answer the provider reports as normally finished."""


class FakeLLMProvider:
    """Scripted classifier completions in call order (guardrail, grader, rewrite, grader, selection).

    Answer generation and answer grounding are recognised by their system
    prompts and served from ``answer`` and ``grounding`` so that scripts stay
    about routing decisions. A tuple scripts successive calls; its last item repeats.
    """

    def __init__(
        self,
        *responses: str | Exception,
        answer: str | Exception | tuple = ANSWER,
        grounding: str | Exception | tuple = '{"score":90}',
    ) -> None:
        self.responses = list(responses or ('{"score":90}', '{"score":85}'))
        self.answers = answer if isinstance(answer, tuple) else (answer,)
        self.groundings = grounding if isinstance(grounding, tuple) else (grounding,)
        self.calls: list[tuple] = []
        self.generation_calls: list[tuple] = []
        self.grounding_calls: list[tuple] = []

    async def complete(self, messages, **kwargs):
        system = messages[0].content
        if system.startswith(RAG_SYSTEM_PROMPT):
            response = self.answers[min(len(self.generation_calls), len(self.answers) - 1)]
            self.generation_calls.append((messages, kwargs))
        elif system.startswith(GROUNDING_SYSTEM_PROMPT):
            response = self.groundings[min(len(self.grounding_calls), len(self.groundings) - 1)]
            self.grounding_calls.append((messages, kwargs))
        else:
            response = self.responses[len(self.calls)]
            self.calls.append((messages, kwargs))
        if isinstance(response, Exception):
            raise response
        finish_reason = "length" if isinstance(response, Truncated) else "stop" if isinstance(response, Stopped) else None
        return LLMCompletion(
            content=response,
            model="fake-model",
            prompt_tokens=111,
            completion_tokens=22,
            finish_reason=finish_reason,
        )


class FakeEmbeddingProvider:
    """Cosine to the query is the value of the first ``scores`` key found in a passage, else ``default``."""

    def __init__(self, scores: dict[str, float] | None = None, *, default: float = 0.5, error=None) -> None:
        self.scores = scores or {}
        self.default = default
        self.error = error
        self.queries: list[str] = []
        self.passage_batches: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        if self.error:
            raise self.error
        self.queries.append(text)
        return [1.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_batches.append(list(texts))
        vectors = []
        for text in texts:
            score = next((value for key, value in self.scores.items() if key in text), self.default)
            vectors.append([score, math.sqrt(1 - score * score)])
        return vectors


def make_hit(chunk_id: str = "c1", text: str = EVIDENCE_TEXT) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=chunk_id,
        arxiv_id="2401.00001",
        chunk_index=0,
        chunk_text=text,
        word_count=len(text.split()),
        paper_title="Private Paper Title",
        rrf_score=0.5,
    )


def retrieval_result(mode: str = "hybrid", hits: list[HybridSearchHit] | None = None) -> HybridSearchResult:
    results = [make_hit()] if hits is None else hits
    return HybridSearchResult(query="question", retrieval_mode=mode, count=len(results), results=results)


def make_live_paper(arxiv_id: str = "2501.00001") -> LiveArxivPaper:
    return LiveArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Private Live Title {arxiv_id}",
        abstract="private live abstract",
        authors=("Private Author",),
        categories=("cs.CY",),
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        updated_date=None,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def live_result(*arxiv_ids: str) -> LiveArxivSearchResult:
    return LiveArxivSearchResult(query="live query", candidates=tuple(make_live_paper(i) for i in arxiv_ids))


def make_chunk(paper: LiveArxivPaper, index: int = 0) -> TransientLiveEvidenceChunk:
    return TransientLiveEvidenceChunk(
        arxiv_id=paper.arxiv_id,
        chunk_id=f"live::{paper.arxiv_id}::chunk::{index:03d}",
        chunk_index=index,
        paper_title=paper.title,
        section_title="Private Section",
        text=f"{CHUNK_TEXT} {paper.arxiv_id[-1]} {index}",
        word_count=6,
        authors=paper.authors,
        categories=paper.categories,
        published_date=paper.published_date,
        pdf_url=paper.pdf_url,
    )


def fake_processor(*, unusable=(), failing=(), chunks_per_paper=1, error=None, extra=()):
    """Processor fake: papers in ``failing`` error, ``unusable`` yield nothing, the rest yield chunks."""

    async def process(papers, *, question, max_chunks_per_paper):
        if error:
            raise error
        usable = [paper for paper in papers if paper.arxiv_id not in {*unusable, *failing}]
        return LiveDocumentProcessingResult(
            chunks=(*(make_chunk(paper, index) for paper in usable for index in range(chunks_per_paper)), *extra),
            selected_count=len(papers),
            processed_count=len(usable),
            unusable_count=sum(paper.arxiv_id in unusable for paper in papers),
            failed_count=sum(paper.arxiv_id in failing for paper in papers),
        )

    processor = Mock()
    processor.process = AsyncMock(side_effect=process)
    return processor


def dependencies(
    *,
    llm=None,
    result=None,
    results=None,
    retrieval_error=None,
    max_context_tokens=1000,
    live=None,
    live_error=None,
    processor=None,
    embedding=None,
):
    """``results`` scripts successive search calls; an exception item is raised."""
    live_service = Mock()
    live_service.search = AsyncMock(
        return_value=live_result("2501.00001") if live is None else live, side_effect=live_error
    )
    provider = llm or FakeLLMProvider()
    retrieval = Mock(spec=HybridSearchService)
    retrieval.search.return_value = result or retrieval_result()
    retrieval.search.side_effect = results if results is not None else retrieval_error
    context_builder = EvidenceContextBuilder(max_context_tokens=max_context_tokens)
    deps = AgentGraphDependencies(
        llm_provider=provider,
        hybrid_search_service=retrieval,
        evidence_context_builder=context_builder,
        embedding_provider=embedding or FakeEmbeddingProvider(),
        rag_generation_service=RAGGenerationService(
            llm_provider=provider,
            evidence_builder=context_builder,
            max_completion_tokens=64,
            token_safety_margin=16,
        ),
        live_search_service=live_service,
        live_document_processor=processor or fake_processor(),
    )
    return deps, provider, retrieval


async def run(deps, question: str = "AI question", config: AgentGraphConfig | None = None):
    return await build_agent_graph(dependencies=deps, config=config).ainvoke(create_initial_agent_state(question))


def statuses(result) -> list[AgentExecutionStatus]:
    return [event.status for event in result["execution_events"]]


def assert_no_downstream_progress(result) -> None:
    assert result["rewritten_query"] is None
    assert result["generated_answer"] is None
    assert result["generation_result"] is None
    assert result["grounding_passed"] is None
    assert result["grounding_result"] is None
    assert result["grounding_attempts"] == 0
    assert result["live_fallback_used"] is False
    assert result["live_search_result"] is None
    assert result["live_selected_papers"] is None
    assert result["transient_live_evidence"] is None
    assert result["live_evidence"] is None
    assert result["final_evidence"] is None


@pytest.mark.anyio
async def test_sufficient_path_grades_after_one_retrieval_and_completes() -> None:
    expected = retrieval_result()
    deps, provider, retrieval = dependencies(result=expected)
    question = "How can AI improve healthcare access?"

    result = await run(deps, question, AgentGraphConfig(retrieval_size=7))

    assert statuses(result) == SUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(14))
    assert len(provider.calls) == 2
    assert len(provider.generation_calls) == 1
    retrieval.search.assert_called_once_with(question, size=7)
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] == expected
    assert result["guardrail_score"] == 90
    assert result["guardrail_passed"] is True
    assert result["evidence_sufficient"] is True
    assert result["evidence_grade"] == EvidenceGrade(score=85, sufficient=True, source_count=1)
    assert result["current_query"] == question
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert [candidate.source_type for candidate in result["final_evidence"]] == ["local"]
    assert result["generated_answer"] == ANSWER
    assert result["generation_result"].cited_labels == ("[S1]",)
    assert len(provider.grounding_calls) == 1
    assert result["grounding_passed"] is True
    assert result["grounding_result"] == AnswerGroundingResult(score=90, passed=True)
    assert result["grounding_attempts"] == 1
    assert result["rewritten_query"] is None
    assert result["live_fallback_used"] is False
    assert result["live_search_result"] is None
    assert result["transient_live_evidence"] is None
    deps.live_search_service.search.assert_not_awaited()


@pytest.mark.anyio
async def test_single_attempt_config_completes_insufficient_without_rewrite_or_retry() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":25}'))

    result = await run(deps, INDIA_QUESTION, SINGLE_ATTEMPT)

    assert statuses(result) == INSUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(8))
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=25, sufficient=False, source_count=1)
    assert result["current_query"] == INDIA_QUESTION
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    deps.live_search_service.search.assert_not_awaited()
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_evidence_event_metadata_is_bounded_and_operational() -> None:
    deps, _, _ = dependencies(result=retrieval_result(hits=[make_hit("c1", "first"), make_hit("c2", "second")]))

    result = await run(deps)

    started, graded = result["execution_events"][5:7]
    assert started.metadata.model_dump(exclude_none=True) == {"retrieval_attempt": 1, "source_count": 2}
    assert graded.metadata.model_dump(exclude_none=True) == {
        "retrieval_attempt": 1,
        "source_count": 2,
        "evidence_sufficient": True,
        "evidence_score": 85,
    }
    assert result["execution_events"][4].status == S.LOCAL_RETRIEVAL_COMPLETED
    assert S.GRAPH_COMPLETED not in statuses(result)[:7]
    assert statuses(result).count(S.GRAPH_COMPLETED) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("threshold", "score", "sufficient"),
    [(60, 59, False), (60, 60, True), (80, 75, False), (70, 75, True)],
)
async def test_routing_uses_configured_evidence_threshold(threshold: int, score: int, sufficient: bool) -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider('{"score":90}', f'{{"score":{score}}}'))

    result = await run(
        deps,
        config=AgentGraphConfig(
            evidence_sufficiency_threshold=threshold,
            max_local_retrieval_attempts=1,
            live_fallback_enabled=False,
        ),
    )

    assert result["evidence_sufficient"] is sufficient
    assert result["evidence_grade"].score == score
    assert statuses(result) == (SUFFICIENT_PATH if sufficient else INSUFFICIENT_PATH)


@pytest.mark.anyio
async def test_grader_evaluates_original_question_while_retrieval_uses_current_query() -> None:
    deps, provider, retrieval = dependencies()
    state = create_initial_agent_state(INDIA_QUESTION)
    state["current_query"] = "India healthcare AI applications"

    result = await build_agent_graph(dependencies=deps).ainvoke(state)

    retrieval.search.assert_called_once_with("India healthcare AI applications", size=5)
    grader_user_message = provider.calls[1][0][1].content
    assert INDIA_QUESTION in grader_user_message
    assert "India healthcare AI applications" not in grader_user_message
    assert result["original_question"] == INDIA_QUESTION
    assert result["current_query"] == "India healthcare AI applications"


@pytest.mark.anyio
async def test_grader_receives_bounded_deterministic_evidence_context() -> None:
    hits = [make_hit("c1", "alpha " * 40), make_hit("c1", "duplicate chunk id"), make_hit("c3", "omega " * 400)]
    deps, provider, _ = dependencies(result=retrieval_result(hits=hits), max_context_tokens=120)

    result = await run(deps)

    grader_user_message = provider.calls[1][0][1].content
    assert len(grader_user_message) < 120 * 4 + 200
    assert "duplicate chunk id" not in grader_user_message
    assert grader_user_message.index("[S1]") < grader_user_message.index("[S2]")
    assert "omega " * 400 not in grader_user_message
    assert result["evidence_grade"].source_count == 2
    assert provider.calls[1][1] == {"temperature": 0.0, "max_tokens": 1024}


@pytest.mark.anyio
async def test_empty_retrieval_is_insufficient_without_grader_llm_call() -> None:
    deps, provider, retrieval = dependencies(result=retrieval_result(hits=[]))

    result = await run(deps, config=SINGLE_ATTEMPT)

    assert statuses(result) == INSUFFICIENT_PATH
    assert len(provider.calls) == 1
    retrieval.search.assert_called_once()
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=0, sufficient=False, source_count=0)
    assert result["execution_events"][6].metadata.evidence_score == 0
    assert result["execution_events"][6].metadata.source_count == 0
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "grader_response",
    [RuntimeError("private grader detail"), "not json", '{"score":73,"reason":"private grader detail"}'],
)
async def test_grader_failure_is_distinct_from_insufficient_evidence(grader_response) -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', grader_response))

    result = await run(deps)

    assert statuses(result) == GRADER_FAILED_PATH
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert "private grader detail" not in repr(result["execution_events"])
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_context_builder_failure_is_controlled_evidence_grading_failure() -> None:
    expected = retrieval_result()
    deps, provider, retrieval = dependencies(result=expected)
    builder = Mock(spec=EvidenceContextBuilder)
    builder.build.side_effect = RuntimeError("private context builder detail")
    deps = dataclasses.replace(deps, evidence_context_builder=builder)

    result = await run(deps)

    assert statuses(result) == GRADER_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert [event.sequence for event in result["execution_events"]] == list(range(7))
    builder.build.assert_called_once_with(expected)
    assert len(provider.calls) == 1
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] == expected
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert result["execution_events"][5].metadata.model_dump(exclude_none=True) == {"retrieval_attempt": 1}
    serialized = repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private context builder detail" not in serialized
    assert "private context builder detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )
    assert_no_downstream_progress(result)


@pytest.mark.anyio
async def test_guardrail_rejection_is_normal_completion_and_skips_retrieval_and_grading() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":10}'))

    result = await run(deps, "What is a dog?")

    assert statuses(result) == REJECTED_PATH
    assert len(provider.calls) == 1
    assert result["guardrail_score"] == 10
    assert result["guardrail_passed"] is False
    assert result["retrieval_attempts"] == 0
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.OUT_OF_SCOPE
    assert result["error_category"] is None
    retrieval.search.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("guardrail_response", [RuntimeError("private"), "not json"])
async def test_guardrail_failure_is_internal_and_safe(guardrail_response) -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider(guardrail_response))

    result = await run(deps)

    assert statuses(result) == GUARDRAIL_FAILED_PATH
    assert len(provider.calls) == 1
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.GUARDRAIL_FAILURE
    assert result["guardrail_passed"] is None
    assert result["retrieval_attempts"] == 0
    assert result["evidence_sufficient"] is None
    retrieval.search.assert_not_called()
    assert "private" not in repr(result["execution_events"])


@pytest.mark.anyio
async def test_retrieval_failure_is_not_insufficient_evidence_and_skips_grading() -> None:
    deps, provider, retrieval = dependencies(retrieval_error=RuntimeError("private retrieval detail"))

    result = await run(deps)

    assert statuses(result) == RETRIEVAL_FAILED_PATH
    assert len(provider.calls) == 1
    assert result["retrieval_attempts"] == 1
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.RETRIEVAL_FAILED
    assert result["error_category"] == AgentErrorCategory.RETRIEVAL_FAILURE
    retrieval.search.assert_called_once()
    assert "private retrieval detail" not in repr(result["execution_events"])


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["hybrid", "bm25_fallback", "vector_fallback"])
async def test_every_valid_hybrid_mode_is_graded(mode: str) -> None:
    expected = retrieval_result(mode)
    deps, _, _ = dependencies(result=expected)

    result = await run(deps)

    assert result["local_retrieval_result"] == expected
    assert statuses(result) == SUFFICIENT_PATH


@pytest.mark.anyio
async def test_sync_retrieval_is_offloaded_without_timing_assertions() -> None:
    expected = retrieval_result()
    deps, _, retrieval = dependencies(result=expected)

    real_to_thread = asyncio.to_thread

    async def to_thread(function, *args, **kwargs):
        if function is retrieval.search:
            return expected
        return await real_to_thread(function, *args, **kwargs)

    with patch("src.services.agent.graph.asyncio.to_thread", new=AsyncMock(side_effect=to_thread)) as offload:
        result = await run(deps, "AI question")

    offload.assert_any_await(retrieval.search, "AI question", size=5)
    assert [call.args[0] for call in offload.await_args_list].count(retrieval.search) == 1
    retrieval.search.assert_not_called()
    assert result["local_retrieval_result"] == expected


@pytest.mark.anyio
@pytest.mark.parametrize(("threshold", "passed"), [(85, True), (86, False)])
async def test_routing_uses_configured_guardrail_threshold(threshold: int, passed: bool) -> None:
    deps, _, retrieval = dependencies(llm=FakeLLMProvider('{"score":85}', '{"score":85}'))

    result = await run(deps, config=AgentGraphConfig(guardrail_threshold=threshold))

    assert result["guardrail_passed"] is passed
    assert retrieval.search.call_count == (1 if passed else 0)
    assert result["terminal_reason"] == (None if passed else TerminalReason.OUT_OF_SCOPE)


def test_graph_compilation_has_conditional_m08_topology() -> None:
    deps, _, _ = dependencies()

    graph = build_agent_graph(dependencies=deps).get_graph()

    assert set(graph.nodes) == {
        "__start__",
        "graph_start",
        "guardrail",
        "out_of_scope",
        "local_retrieval",
        "evidence_grading",
        "query_rewrite",
        "live_arxiv_search",
        "live_paper_selection",
        "live_document_processing",
        "evidence_rerank",
        "answer_generation",
        "answer_grounding",
        "grounding_failed",
        "insufficient_evidence",
        "graph_complete",
        "__end__",
    }
    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "graph_start", False),
        ("graph_start", "guardrail", False),
        ("guardrail", "local_retrieval", True),
        ("guardrail", "out_of_scope", True),
        ("guardrail", "__end__", True),
        ("out_of_scope", "__end__", False),
        ("local_retrieval", "evidence_grading", True),
        ("local_retrieval", "__end__", True),
        ("evidence_grading", "evidence_rerank", True),
        ("evidence_grading", "query_rewrite", True),
        ("evidence_grading", "__end__", True),
        ("query_rewrite", "local_retrieval", True),
        ("query_rewrite", "__end__", True),
        ("evidence_grading", "live_arxiv_search", True),
        ("evidence_grading", "insufficient_evidence", True),
        ("live_arxiv_search", "live_paper_selection", True),
        ("live_paper_selection", "live_document_processing", True),
        ("live_paper_selection", "insufficient_evidence", True),
        ("live_paper_selection", "__end__", True),
        ("live_document_processing", "evidence_rerank", True),
        ("evidence_rerank", "answer_generation", True),
        ("evidence_rerank", "insufficient_evidence", True),
        ("evidence_rerank", "__end__", True),
        ("answer_generation", "answer_grounding", True),
        ("answer_grounding", "graph_complete", True),
        ("answer_grounding", "answer_generation", True),
        ("answer_grounding", "grounding_failed", True),
        ("answer_grounding", "__end__", True),
        ("grounding_failed", "__end__", False),
        ("answer_generation", "insufficient_evidence", True),
        ("answer_generation", "__end__", True),
        ("live_document_processing", "insufficient_evidence", True),
        ("live_document_processing", "__end__", True),
        ("live_arxiv_search", "insufficient_evidence", True),
        ("live_arxiv_search", "__end__", True),
        ("insufficient_evidence", "__end__", False),
        ("graph_complete", "__end__", False),
    }


def assert_no_downstream_progress_after_rewrite(result) -> None:
    assert result["generated_answer"] is None
    assert result["grounding_passed"] is None
    assert result["live_fallback_used"] is False
    assert result["live_search_result"] is None
    assert result["live_selected_papers"] is None
    assert result["transient_live_evidence"] is None
    assert result["live_evidence"] is None
    assert result["final_evidence"] is None


def retry_dependencies(*grader_and_rewrite: str | Exception, results=None):
    """Guardrail passes and attempt 1 grades 25; the remaining responses are scripted."""
    first, second = retrieval_result(hits=[make_hit("a1", "attempt one evidence")]), retrieval_result(
        hits=[make_hit("b1", "attempt two evidence"), make_hit("b2", "more attempt two evidence")]
    )
    llm = FakeLLMProvider('{"score":90}', '{"score":25}', *grader_and_rewrite)
    deps, provider, retrieval = dependencies(llm=llm, results=[first, second] if results is None else results)
    return deps, provider, retrieval, first, second


def exhausted_dependencies(*, live=None, live_error=None, selection: str | Exception = SELECT_FIRST, processor=None):
    """Both local attempts grade insufficient, so the default config reaches live fallback."""
    first, second = retrieval_result(hits=[make_hit("a1", "attempt one evidence")]), retrieval_result(
        hits=[make_hit("b1", "attempt two evidence")]
    )
    llm = FakeLLMProvider('{"score":90}', '{"score":25}', REWRITE_JSON, '{"score":40}', selection)
    deps, provider, retrieval = dependencies(
        llm=llm, results=[first, second], live=live, live_error=live_error, processor=processor
    )
    return deps, provider, retrieval, second


def user_payload(call) -> str:
    return call[0][1].content


@pytest.mark.anyio
async def test_retry_then_sufficient_uses_rewritten_query_and_grades_original_question() -> None:
    deps, provider, retrieval, _, second = retry_dependencies(REWRITE_JSON, '{"score":88}')

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == RETRY_SUFFICIENT_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(20))
    assert [call.args[0] for call in retrieval.search.call_args_list] == [INDIA_QUESTION, REWRITTEN]
    assert len(provider.calls) == 4
    guardrail_call, first_grade, rewrite, second_grade = provider.calls
    assert user_payload(rewrite).startswith("UNTRUSTED_REWRITE_INPUT_JSON\n")
    for grade_call in (first_grade, second_grade):
        assert user_payload(grade_call).startswith("UNTRUSTED_GRADING_INPUT_JSON\n")
        assert INDIA_QUESTION in user_payload(grade_call)
        assert REWRITTEN not in user_payload(grade_call)
    assert "attempt one evidence" in user_payload(first_grade)
    assert "attempt two evidence" in user_payload(second_grade)
    assert "attempt one evidence" not in user_payload(second_grade)
    assert result["original_question"] == INDIA_QUESTION
    assert result["current_query"] == REWRITTEN
    assert result["rewritten_query"] == REWRITTEN
    assert result["retrieval_attempts"] == 2
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is True
    assert result["evidence_grade"] == EvidenceGrade(score=88, sufficient=True, source_count=2)
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    deps.live_search_service.search.assert_not_awaited()
    deps.live_document_processor.process.assert_not_awaited()
    assert result["live_fallback_used"] is False
    assert result["transient_live_evidence"] is None
    assert len(provider.generation_calls) == 1
    assert [candidate.chunk_id for candidate in result["final_evidence"]] == ["b1", "b2"]
    assert {candidate.source_type for candidate in result["final_evidence"]} == {"local"}
    generation_prompt = user_payload(provider.generation_calls[0])
    assert INDIA_QUESTION in generation_prompt
    assert "attempt two evidence" in generation_prompt
    assert "attempt one evidence" not in generation_prompt
    assert result["generated_answer"] == ANSWER
    assert result["grounding_passed"] is True
    assert result["grounding_attempts"] == 1


@pytest.mark.anyio
async def test_exhausted_with_fallback_disabled_is_insufficient_evidence_without_live_search() -> None:
    deps, provider, retrieval, _, second = retry_dependencies(REWRITE_JSON, '{"score":40}')

    result = await run(deps, INDIA_QUESTION, LOCAL_ONLY)

    assert statuses(result) == RETRY_INSUFFICIENT_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 4
    assert statuses(result).count(S.QUERY_REWRITTEN) == 1
    assert result["retrieval_attempts"] == 2
    assert result["rewritten_query"] == REWRITTEN
    assert result["current_query"] == REWRITTEN
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=40, sufficient=False, source_count=2)
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    deps.live_search_service.search.assert_not_awaited()
    assert_no_downstream_progress_after_rewrite(result)


@pytest.mark.anyio
async def test_attempt_metadata_distinguishes_first_and_second_attempts() -> None:
    deps, _, _, _, _ = retry_dependencies(REWRITE_JSON, '{"score":88}')

    result = await run(deps)

    metadata = [
        (event.status, event.metadata.model_dump(exclude_none=True)) for event in result["execution_events"][3:13]
    ]
    assert metadata == [
        (S.LOCAL_RETRIEVAL_STARTED, {"retrieval_attempt": 1}),
        (S.LOCAL_RETRIEVAL_COMPLETED, {"retrieval_attempt": 1, "source_count": 1}),
        (S.EVIDENCE_GRADING_STARTED, {"retrieval_attempt": 1, "source_count": 1}),
        (
            S.EVIDENCE_INSUFFICIENT,
            {"retrieval_attempt": 1, "source_count": 1, "evidence_sufficient": False, "evidence_score": 25},
        ),
        (S.QUERY_REWRITE_STARTED, {"retrieval_attempt": 1, "next_retrieval_attempt": 2}),
        (S.QUERY_REWRITTEN, {"retrieval_attempt": 1, "next_retrieval_attempt": 2}),
        (S.LOCAL_RETRIEVAL_STARTED, {"retrieval_attempt": 2}),
        (S.LOCAL_RETRIEVAL_COMPLETED, {"retrieval_attempt": 2, "source_count": 2}),
        (S.EVIDENCE_GRADING_STARTED, {"retrieval_attempt": 2, "source_count": 2}),
        (
            S.EVIDENCE_SUFFICIENT,
            {"retrieval_attempt": 2, "source_count": 2, "evidence_sufficient": True, "evidence_score": 88},
        ),
    ]


@pytest.mark.anyio
async def test_empty_first_retrieval_is_rewritten_and_retried_under_default_config() -> None:
    empty = retrieval_result(hits=[])
    deps, provider, retrieval = dependencies(
        llm=FakeLLMProvider('{"score":90}', REWRITE_JSON), results=[empty, empty]
    )

    result = await run(deps, config=LOCAL_ONLY)

    assert statuses(result) == RETRY_INSUFFICIENT_PATH
    assert len(provider.calls) == 2
    assert retrieval.search.call_count == 2
    assert result["retrieval_attempts"] == 2
    assert result["evidence_grade"] == EvidenceGrade(score=0, sufficient=False, source_count=0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "rewrite_response",
    [
        RuntimeError("private rewrite detail"),
        "private rewrite detail, not json",
        '{"query":"new query","reason":"private rewrite detail"}',
        '{"query":"   "}',
        '{"query":"  ai   QUESTION "}',
        f'{{"query":"{"x" * (MAX_REWRITTEN_QUERY_CHARACTERS + 1)}"}}',
    ],
)
async def test_rewrite_failure_is_controlled_and_skips_second_retrieval(rewrite_response) -> None:
    deps, provider, retrieval, first, _ = retry_dependencies(rewrite_response)

    result = await run(deps, "AI question")

    assert statuses(result) == REWRITE_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert len(provider.calls) == 3
    retrieval.search.assert_called_once()
    assert result["retrieval_attempts"] == 1
    assert result["rewritten_query"] is None
    assert result["current_query"] == "AI question"
    assert result["local_retrieval_result"] == first
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"].score == 25
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.QUERY_REWRITE_FAILURE
    serialized = repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private rewrite detail" not in serialized
    assert "new query" not in serialized


@pytest.mark.anyio
async def test_second_retrieval_failure_counts_as_attempt_two_and_skips_grading() -> None:
    first = retrieval_result()
    deps, provider, retrieval, _, _ = retry_dependencies(
        REWRITE_JSON, results=[first, RuntimeError("private retrieval detail")]
    )

    result = await run(deps)

    assert statuses(result) == SECOND_RETRIEVAL_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 3
    assert result["retrieval_attempts"] == 2
    assert result["current_query"] == REWRITTEN
    assert result["local_retrieval_result"] is None
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.RETRIEVAL_FAILED
    assert result["error_category"] == AgentErrorCategory.RETRIEVAL_FAILURE
    assert "private retrieval detail" not in repr(result["execution_events"])


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["provider", "parser", "context_builder"])
async def test_second_grading_failure_does_not_restore_first_grade_or_retry(failure: str) -> None:
    grader_response = {"provider": RuntimeError("private grader detail"), "parser": "private grader detail"}.get(
        failure, '{"score":99}'
    )
    deps, provider, retrieval, _, second = retry_dependencies(REWRITE_JSON, grader_response)
    if failure == "context_builder":
        real = deps.evidence_context_builder
        builder = Mock(spec=EvidenceContextBuilder)
        builder.build.side_effect = [real.build(retrieval_result()), RuntimeError("private grader detail")]
        deps = dataclasses.replace(deps, evidence_context_builder=builder)

    result = await run(deps)

    assert statuses(result) == SECOND_GRADER_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert retrieval.search.call_count == 2
    assert statuses(result).count(S.QUERY_REWRITE_STARTED) == 1
    assert len(provider.calls) == (3 if failure == "context_builder" else 4)
    assert result["retrieval_attempts"] == 2
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is None
    assert result["evidence_grade"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_GRADING_FAILURE
    assert "private grader detail" not in repr(result["execution_events"])


@pytest.mark.anyio
async def test_first_attempt_grade_is_cleared_before_second_attempt_is_graded() -> None:
    deps, _, _, first, second = retry_dependencies(REWRITE_JSON, '{"score":88}')
    graph = build_agent_graph(dependencies=deps)

    snapshots = {}
    async for state in graph.astream(create_initial_agent_state("AI question"), stream_mode="values"):
        if state["execution_events"]:
            snapshots[len(state["execution_events"])] = state

    after_first_grade = snapshots[len(FIRST_INSUFFICIENT)]
    assert after_first_grade["local_retrieval_result"] == first
    assert after_first_grade["evidence_sufficient"] is False
    assert after_first_grade["evidence_grade"].score == 25

    after_rewrite = snapshots[len(REWRITTEN_PATH)]
    assert after_rewrite["current_query"] == REWRITTEN
    assert after_rewrite["retrieval_attempts"] == 1
    assert after_rewrite["local_retrieval_result"] is None
    assert after_rewrite["evidence_sufficient"] is None
    assert after_rewrite["evidence_grade"] is None

    after_second_retrieval = snapshots[len(SECOND_RETRIEVED)]
    assert after_second_retrieval["retrieval_attempts"] == 2
    assert after_second_retrieval["local_retrieval_result"] == second
    assert after_second_retrieval["evidence_sufficient"] is None
    assert after_second_retrieval["evidence_grade"] is None

    final = snapshots[len(RETRY_SUFFICIENT_PATH)]
    assert final["evidence_grade"] == EvidenceGrade(score=88, sufficient=True, source_count=2)


@pytest.mark.anyio
async def test_total_attempts_follow_configuration_and_never_exceed_it() -> None:
    llm = FakeLLMProvider(
        '{"score":90}',
        '{"score":10}',
        '{"query":"first rewrite"}',
        '{"score":10}',
        '{"query":"second rewrite"}',
        '{"score":10}',
    )
    deps, provider, retrieval = dependencies(llm=llm)

    result = await run(deps, config=AgentGraphConfig(max_local_retrieval_attempts=3, live_fallback_enabled=False))

    assert [call.args[0] for call in retrieval.search.call_args_list] == [
        "AI question",
        "first rewrite",
        "second rewrite",
    ]
    assert len(provider.calls) == 6
    assert result["retrieval_attempts"] == 3
    assert result["rewritten_query"] == "second rewrite"
    assert result["evidence_sufficient"] is False
    assert statuses(result)[-1] == S.GRAPH_COMPLETED
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE


@pytest.mark.anyio
@pytest.mark.parametrize(("attempts", "maximum"), [(2, 2), (3, 2), (1, 1)])
async def test_retrieval_node_refuses_to_exceed_configured_attempts(attempts: int, maximum: int) -> None:
    deps, provider, retrieval = dependencies()
    state = create_initial_agent_state("AI question")
    state["retrieval_attempts"] = attempts
    config = AgentGraphConfig(max_local_retrieval_attempts=maximum)

    result = await build_agent_graph(dependencies=deps, config=config).ainvoke(state)

    retrieval.search.assert_not_called()
    assert statuses(result) == [*GUARDRAIL_PASSED, S.GRAPH_FAILED]
    assert len(provider.calls) == 1
    assert result["retrieval_attempts"] == attempts
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.INTERNAL_FAILURE


@pytest.mark.anyio
async def test_live_candidates_are_selected_processed_and_stored_as_transient_evidence() -> None:
    candidates = live_result("2501.00001", "2501.00002", "2501.00003")
    deps, provider, retrieval, second = exhausted_dependencies(live=candidates, selection=SELECT_TWO)

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == LIVE_EVIDENCE_PATH
    assert [event.sequence for event in result["execution_events"]] == list(range(26))
    assert S.GRAPH_COMPLETED not in statuses(result)[:-1]
    deps.live_search_service.search.assert_awaited_once_with(
        REWRITTEN, max_results=5, exclude_arxiv_ids=("2401.00001",)
    )
    assert [call.args[0] for call in retrieval.search.call_args_list] == [INDIA_QUESTION, REWRITTEN]
    assert len(provider.calls) == 5
    selection_payload = user_payload(provider.calls[4])
    assert selection_payload.startswith("UNTRUSTED_SELECTION_INPUT_JSON\n")
    assert INDIA_QUESTION in selection_payload
    assert REWRITTEN not in selection_payload
    selected = candidates.candidates[:2]
    deps.live_document_processor.process.assert_awaited_once_with(
        selected, question=INDIA_QUESTION, max_chunks_per_paper=8
    )
    assert result["original_question"] == INDIA_QUESTION
    assert result["current_query"] == REWRITTEN
    assert result["retrieval_attempts"] == 2
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] == candidates
    assert result["live_selected_papers"] == selected
    assert result["transient_live_evidence"] == (make_chunk(selected[0]), make_chunk(selected[1]))
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["evidence_grade"] == EvidenceGrade(score=40, sufficient=False, source_count=1)
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert result["live_evidence"] is None
    assert [(candidate.source_type, candidate.chunk_id) for candidate in result["final_evidence"]] == [
        ("local", "b1"),
        ("live_arxiv", "live::2501.00001::chunk::000"),
        ("live_arxiv", "live::2501.00002::chunk::000"),
    ]
    assert len(provider.generation_calls) == 1
    assert result["generated_answer"] == ANSWER
    assert [source.label for source in result["generation_result"].sources] == ["[S1]", "[S2]", "[S3]"]
    assert [source.source_type for source in result["generation_result"].sources] == [
        "local",
        "live_arxiv",
        "live_arxiv",
    ]
    assert result["grounding_passed"] is True
    assert len(provider.grounding_calls) == 1
    assert result["execution_events"][19:23] == result["execution_events"][-7:-3]
    assert [event.metadata.model_dump(exclude_none=True) for event in result["execution_events"][19:23]] == [
        {"local_candidate_count": 1, "live_candidate_count": 2, "merged_candidate_count": 3},
        {"local_candidate_count": 1, "live_candidate_count": 2, "merged_candidate_count": 3, "final_source_count": 3},
        {"generation_attempt": 1, "final_source_count": 3},
        {"generation_attempt": 1, "final_source_count": 3, "prompt_tokens": 111, "completion_tokens": 22},
    ]
    metadata = [event.metadata.model_dump(exclude_none=True) for event in result["execution_events"][13:19]]
    assert metadata == [
        {"live_fallback_used": True, "max_results": 5},
        {"live_fallback_used": True, "max_results": 5, "candidate_count": 3},
        {"candidate_count": 3},
        {"candidate_count": 3, "selected_count": 2},
        {"selected_count": 2},
        {"selected_count": 2, "processed_count": 2, "failed_count": 0, "transient_chunk_count": 2},
    ]


@pytest.mark.anyio
async def test_zero_selected_papers_is_insufficient_evidence_without_processing() -> None:
    candidates = live_result("2501.00001", "2501.00002")
    deps, provider, _, second = exhausted_dependencies(live=candidates, selection=SELECT_NONE)

    result = await run(deps)

    assert statuses(result) == LIVE_ZERO_SELECTED_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    assert len(provider.calls) == 5
    deps.live_document_processor.process.assert_not_awaited()
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] == candidates
    assert result["live_selected_papers"] == ()
    assert result["transient_live_evidence"] is None
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    assert result["execution_events"][16].metadata.selected_count == 0


@pytest.mark.anyio
async def test_only_fabricated_selection_ids_is_treated_as_zero_selection() -> None:
    deps, _, _, _ = exhausted_dependencies(selection='{"selected_arxiv_ids":["9999.99999"]}')

    result = await run(deps)

    assert statuses(result) == LIVE_ZERO_SELECTED_PATH
    deps.live_document_processor.process.assert_not_awaited()
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE


@pytest.mark.anyio
@pytest.mark.parametrize(
    "selection",
    [
        RuntimeError("private selection detail"),
        "private selection detail, not json",
        '{"selected_arxiv_ids":["2501.00001"],"reason":"private selection detail"}',
        '{"selected_arxiv_ids":"2501.00001"}',
    ],
)
async def test_selection_failure_is_distinct_from_zero_selection(selection) -> None:
    deps, provider, _, _ = exhausted_dependencies(selection=selection)

    result = await run(deps)

    assert statuses(result) == LIVE_SELECTION_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert len(provider.calls) == 5
    deps.live_document_processor.process.assert_not_awaited()
    assert result["live_selected_papers"] is None
    assert result["transient_live_evidence"] is None
    assert result["live_search_result"] is not None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.LIVE_SELECTION_FAILURE
    assert "private selection detail" not in repr(
        [event.model_dump(mode="json") for event in result["execution_events"]]
    )


@pytest.mark.anyio
async def test_partial_document_success_keeps_usable_evidence_and_counts_failure() -> None:
    candidates = live_result("2501.00001", "2501.00002")
    deps, _, _, _ = exhausted_dependencies(
        live=candidates, selection=SELECT_TWO, processor=fake_processor(failing={"2501.00002"})
    )

    result = await run(deps)

    assert statuses(result) == LIVE_EVIDENCE_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    assert result["transient_live_evidence"] == (make_chunk(candidates.candidates[0]),)
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert result["execution_events"][18].metadata.model_dump(exclude_none=True) == {
        "selected_count": 2,
        "processed_count": 1,
        "failed_count": 1,
        "transient_chunk_count": 1,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("unusable", "failing"),
    [({"2501.00001", "2501.00002"}, set()), ({"2501.00001"}, {"2501.00002"})],
)
async def test_no_usable_content_is_insufficient_evidence_not_failure(unusable: set, failing: set) -> None:
    deps, _, _, _ = exhausted_dependencies(
        live=live_result("2501.00001", "2501.00002"),
        selection=SELECT_TWO,
        processor=fake_processor(unusable=unusable, failing=failing),
    )

    result = await run(deps)

    assert statuses(result) == [*LIVE_SELECTED, *LIVE_DOCUMENT_TAIL]
    assert S.GRAPH_FAILED not in statuses(result)
    assert result["transient_live_evidence"] == ()
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    assert result["execution_events"][18].metadata.transient_chunk_count == 0
    assert result["execution_events"][18].metadata.failed_count == len(failing)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "processor",
    [
        fake_processor(failing={"2501.00001", "2501.00002"}),
        fake_processor(error=RuntimeError("private processing detail")),
    ],
)
async def test_all_document_execution_failures_fail_the_graph(processor) -> None:
    deps, _, _, second = exhausted_dependencies(
        live=live_result("2501.00001", "2501.00002"), selection=SELECT_TWO, processor=processor
    )

    result = await run(deps)

    assert statuses(result) == LIVE_DOCUMENTS_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert result["transient_live_evidence"] == ()
    assert result["local_retrieval_result"] == second
    assert result["live_search_result"].count == 2
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.LIVE_DOCUMENT_PROCESSING_FAILURE
    assert "private processing detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )


def real_processor(outcomes: dict[str, str]) -> LiveDocumentProcessingService:
    """The real processing service over a fake client and parser scripted per arXiv ID."""

    async def download_pdf(paper, *, cache_dir, max_bytes):
        if outcomes[paper.arxiv_id] == "download_error":
            raise ArxivPDFDownloadError("private document detail")
        path = Path(cache_dir) / f"{paper.arxiv_id}.pdf"
        path.write_bytes(b"%PDF-1.7\nfake")
        return path

    async def parse_pdf(pdf_path):
        outcome = outcomes[Path(pdf_path).stem]
        if outcome == "no_text":
            raise PDFNoTextError(f"private document detail {pdf_path}")
        if outcome == "parser_error":
            raise PDFParserError("private document detail")
        return ParsedPDF(raw_text="usable full text " * 20)

    client = Mock(spec=ArxivClient)
    client.download_pdf = AsyncMock(side_effect=download_pdf)
    parser = Mock()
    parser.parse_pdf = AsyncMock(side_effect=parse_pdf)
    return LiveDocumentProcessingService(
        acquisition_service=LivePaperAcquisitionService(arxiv_client=client),
        pdf_parser=parser,
        chunking_service=PaperChunkingService(target_words=50, overlap_words=10, min_words=10),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("first", "second", "failed_count"),
    [("no_text", "no_text", 0), ("no_text", "parser_error", 1), ("no_text", "download_error", 1)],
)
async def test_papers_without_extractable_text_are_insufficient_evidence_not_failure(
    first: str, second: str, failed_count: int
) -> None:
    deps, _, _, second_local = exhausted_dependencies(
        live=live_result("2501.00001", "2501.00002"),
        selection=SELECT_TWO,
        processor=real_processor({"2501.00001": first, "2501.00002": second}),
    )

    result = await run(deps)

    assert statuses(result) == [*LIVE_SELECTED, *LIVE_DOCUMENT_TAIL]
    assert S.GRAPH_FAILED not in statuses(result)
    assert result["transient_live_evidence"] == ()
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    assert result["local_retrieval_result"] == second_local
    assert result["execution_events"][18].metadata.model_dump(exclude_none=True) == {
        "selected_count": 2,
        "processed_count": 0,
        "failed_count": failed_count,
        "transient_chunk_count": 0,
    }
    assert "private document detail" not in repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private document detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )


@pytest.mark.anyio
@pytest.mark.parametrize(("first", "second"), [("parser_error", "parser_error"), ("parser_error", "download_error")])
async def test_genuine_parser_and_download_failures_for_every_paper_still_fail_the_graph(
    first: str, second: str
) -> None:
    deps, _, _, _ = exhausted_dependencies(
        live=live_result("2501.00001", "2501.00002"),
        selection=SELECT_TWO,
        processor=real_processor({"2501.00001": first, "2501.00002": second}),
    )

    result = await run(deps)

    assert statuses(result) == LIVE_DOCUMENTS_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert result["transient_live_evidence"] == ()
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.LIVE_DOCUMENT_PROCESSING_FAILURE
    assert "private document detail" not in repr([event.model_dump(mode="json") for event in result["execution_events"]])


@pytest.mark.anyio
@pytest.mark.parametrize(("second", "failed_count"), [("no_text", 0), ("parser_error", 1)])
async def test_usable_paper_survives_a_no_text_or_failed_sibling(second: str, failed_count: int) -> None:
    deps, _, _, _ = exhausted_dependencies(
        live=live_result("2501.00001", "2501.00002"),
        selection=SELECT_TWO,
        processor=real_processor({"2501.00001": "usable", "2501.00002": second}),
    )

    result = await run(deps)

    assert statuses(result) == LIVE_EVIDENCE_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    assert {chunk.arxiv_id for chunk in result["transient_live_evidence"]} == {"2501.00001"}
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    completed = result["execution_events"][18].metadata
    assert (completed.selected_count, completed.processed_count, completed.failed_count) == (2, 1, failed_count)


@pytest.mark.anyio
async def test_live_processing_bounds_are_enforced_by_the_graph() -> None:
    candidates = live_result(*(f"2501.{index:05d}" for index in range(1, 6)))
    everything = '{"selected_arxiv_ids":["2501.00001","2501.00002","2501.00003","2501.00004","2501.00005"]}'
    stray = make_chunk(make_live_paper("2501.00005"))
    deps, _, _, _ = exhausted_dependencies(
        live=candidates, selection=everything, processor=fake_processor(chunks_per_paper=7, extra=(stray,))
    )
    config = AgentGraphConfig(live_pdf_max_papers=2, live_max_chunks_per_paper=3)

    result = await run(deps, config=config)

    assert result["live_search_result"].count == 5
    assert [paper.arxiv_id for paper in result["live_selected_papers"]] == ["2501.00001", "2501.00002"]
    deps.live_document_processor.process.assert_awaited_once()
    assert deps.live_document_processor.process.await_args.args[0] == candidates.candidates[:2]
    assert deps.live_document_processor.process.await_args.kwargs["max_chunks_per_paper"] == 3
    chunks = result["transient_live_evidence"]
    assert [(chunk.arxiv_id, chunk.chunk_index) for chunk in chunks] == [
        ("2501.00001", 0),
        ("2501.00001", 1),
        ("2501.00001", 2),
        ("2501.00002", 0),
        ("2501.00002", 1),
        ("2501.00002", 2),
    ]
    assert len(chunks) <= config.live_pdf_max_papers * config.live_max_chunks_per_paper
    assert result["execution_events"][18].metadata.transient_chunk_count == 6


@pytest.mark.anyio
async def test_live_search_uses_configured_bound_even_if_service_returns_more() -> None:
    deps, _, _, _ = exhausted_dependencies(live=live_result("2501.00001", "2501.00002", "2501.00003"))

    result = await run(deps, config=AgentGraphConfig(live_arxiv_max_results=2))

    assert deps.live_search_service.search.await_args.kwargs["max_results"] == 2
    assert [paper.arxiv_id for paper in result["live_search_result"].candidates] == ["2501.00001", "2501.00002"]
    assert result["execution_events"][14].metadata.candidate_count == 2


@pytest.mark.anyio
async def test_single_attempt_config_falls_back_live_with_unrewritten_query() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":25}', SELECT_FIRST))

    result = await run(deps, INDIA_QUESTION, AgentGraphConfig(max_local_retrieval_attempts=1))

    assert statuses(result) == [
        *FIRST_INSUFFICIENT,
        S.LIVE_FALLBACK_STARTED,
        S.LIVE_FALLBACK_COMPLETED,
        S.LIVE_SELECTION_STARTED,
        S.LIVE_SELECTION_COMPLETED,
        S.LIVE_DOCUMENT_PROCESSING_STARTED,
        S.LIVE_DOCUMENT_PROCESSING_COMPLETED,
        *GENERATION_TAIL,
    ]
    assert deps.live_search_service.search.await_args.args == (INDIA_QUESTION,)
    assert len(provider.calls) == 3
    retrieval.search.assert_called_once()
    assert result["rewritten_query"] is None
    assert result["live_fallback_used"] is True


@pytest.mark.anyio
async def test_live_search_with_zero_candidates_is_insufficient_evidence_not_failure() -> None:
    empty = live_result()
    deps, _, retrieval, second = exhausted_dependencies(live=empty)

    result = await run(deps)

    assert statuses(result) == LIVE_COMPLETED_PATH
    assert S.GRAPH_FAILED not in statuses(result)
    deps.live_search_service.search.assert_awaited_once()
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] == empty
    assert result["live_search_result"].count == 0
    assert result["execution_events"][14].metadata.candidate_count == 0
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    assert retrieval.search.call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("error", [LiveSearchError("private live detail"), RuntimeError("private live detail")])
async def test_live_search_failure_is_distinct_from_zero_candidates(error: Exception) -> None:
    deps, provider, retrieval, second = exhausted_dependencies(live_error=error)

    result = await run(deps)

    assert statuses(result) == LIVE_FAILED_PATH
    assert S.GRAPH_COMPLETED not in statuses(result)
    deps.live_search_service.search.assert_awaited_once()
    assert result["live_fallback_used"] is True
    assert result["live_search_result"] is None
    assert result["local_retrieval_result"] == second
    assert result["evidence_sufficient"] is False
    assert result["retrieval_attempts"] == 2
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.LIVE_SEARCH_FAILURE
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 4
    assert "private live detail" not in repr([event.model_dump(mode="json") for event in result["execution_events"]])
    assert "private live detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )


@pytest.mark.anyio
async def test_live_search_is_skipped_when_first_attempt_is_sufficient() -> None:
    deps, provider, _ = dependencies()

    result = await run(deps)

    assert statuses(result) == SUFFICIENT_PATH
    assert len(provider.calls) == 2
    deps.live_search_service.search.assert_not_awaited()
    deps.live_document_processor.process.assert_not_awaited()
    assert result["live_fallback_used"] is False
    assert result["live_search_result"] is None
    assert result["live_selected_papers"] is None
    assert result["transient_live_evidence"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "llm",
    [
        FakeLLMProvider('{"score":1}'),
        FakeLLMProvider("invalid"),
        FakeLLMProvider('{"score":90}', "invalid"),
        FakeLLMProvider('{"score":90}', '{"score":25}', "invalid"),
    ],
)
async def test_live_search_is_never_reached_from_earlier_terminal_paths(llm: FakeLLMProvider) -> None:
    deps, _, _ = dependencies(llm=FakeLLMProvider(*llm.responses))

    result = await run(deps)

    deps.live_search_service.search.assert_not_awaited()
    deps.live_document_processor.process.assert_not_awaited()
    assert result["live_fallback_used"] is False
    assert S.LIVE_FALLBACK_STARTED not in statuses(result)
    assert S.LIVE_SELECTION_STARTED not in statuses(result)


def test_enabled_live_fallback_requires_a_live_search_service() -> None:
    deps, _, _ = dependencies()
    without_live = dataclasses.replace(deps, live_search_service=None, live_document_processor=None)

    with pytest.raises(ValueError, match="live_search_service is required"):
        build_agent_graph(dependencies=without_live)
    with pytest.raises(ValueError, match="live_document_processor is required"):
        build_agent_graph(dependencies=dataclasses.replace(deps, live_document_processor=None))

    graph = build_agent_graph(dependencies=without_live, config=LOCAL_ONLY).get_graph()
    assert not {"live_arxiv_search", "live_paper_selection", "live_document_processing"} & set(graph.nodes)
    assert "insufficient_evidence" in graph.nodes


def final_ids(result) -> list[tuple[str, str]]:
    return [(candidate.source_type, candidate.chunk_id) for candidate in result["final_evidence"]]


@pytest.mark.anyio
async def test_live_evidence_can_outrank_a_local_result_with_a_high_retrieval_score() -> None:
    strong_local = make_hit("b1", "local evidence about healthcare AI methods").model_copy(
        update={"rrf_score": 0.99, "bm25_score": 99.0, "vector_score": 0.99, "bm25_rank": 1, "vector_rank": 1}
    )
    first = retrieval_result(hits=[make_hit("a1", "attempt one evidence")])
    second = retrieval_result(hits=[strong_local])
    embedding = FakeEmbeddingProvider({"local evidence": 0.2, CHUNK_TEXT: 0.9})
    llm = FakeLLMProvider('{"score":90}', '{"score":25}', REWRITE_JSON, '{"score":40}', SELECT_FIRST)
    deps, provider, _ = dependencies(llm=llm, results=[first, second], embedding=embedding)

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == LIVE_EVIDENCE_PATH
    assert final_ids(result) == [("live_arxiv", "live::2501.00001::chunk::000"), ("local", "b1")]
    assert [round(candidate.relevance_score, 6) for candidate in result["final_evidence"]] == [0.9, 0.2]
    sources = result["generation_result"].sources
    assert [(source.label, source.source_type) for source in sources] == [("[S1]", "live_arxiv"), ("[S2]", "local")]
    assert sources[0].source_url == "https://arxiv.org/pdf/2501.00001"
    assert sources[0].rrf_score is None
    assert sources[1].source_url is None
    prompt = user_payload(provider.generation_calls[0])
    assert prompt.index("[S1] Private Live Title 2501.00001") < prompt.index("[S2] Private Paper Title")
    assert embedding.queries == [INDIA_QUESTION]
    assert len(embedding.passage_batches) == 1
    assert embedding.passage_batches[0] == [
        "Private Paper Title\nlocal evidence about healthcare AI methods",
        f"Private Live Title 2501.00001\nPrivate Section\n{CHUNK_TEXT} 1 0",
    ]
    assert REWRITTEN not in embedding.queries


@pytest.mark.anyio
async def test_final_evidence_is_bounded_and_ordered_by_common_relevance() -> None:
    hits = [make_hit(f"c{index}", f"local passage number {index}") for index in range(1, 5)]
    embedding = FakeEmbeddingProvider({"number 1": 0.1, "number 2": 0.8, "number 3": 0.4, "number 4": 0.8})
    deps, provider, _ = dependencies(result=retrieval_result(hits=hits), embedding=embedding)

    result = await run(deps, config=AgentGraphConfig(final_evidence_max_sources=3))

    assert final_ids(result) == [("local", "c2"), ("local", "c4"), ("local", "c3")]
    assert [candidate.original_rank for candidate in result["final_evidence"]] == [2, 4, 3]
    assert [source.chunk_id for source in result["generation_result"].sources] == ["c2", "c4", "c3"]
    assert [source.label for source in result["generation_result"].sources] == ["[S1]", "[S2]", "[S3]"]
    assert "number 1" not in user_payload(provider.generation_calls[0])
    assert result["execution_events"][8].metadata.model_dump(exclude_none=True) == {
        "local_candidate_count": 4,
        "live_candidate_count": 0,
        "merged_candidate_count": 4,
        "final_source_count": 3,
    }


@pytest.mark.anyio
async def test_cross_source_duplicate_content_keeps_the_local_representation() -> None:
    shared = "shared passage found locally and live"
    paper = make_live_paper("2401.00001")
    duplicate = dataclasses.replace(make_chunk(paper), text="  Shared   passage found LOCALLY and live ")
    distinct = dataclasses.replace(make_chunk(paper, 1), text="a different chunk of the same paper")
    processor = Mock()
    processor.process = AsyncMock(
        return_value=LiveDocumentProcessingResult(chunks=(duplicate, distinct), selected_count=1, processed_count=1)
    )
    first = retrieval_result(hits=[make_hit("a1", "attempt one evidence")])
    second = retrieval_result(hits=[make_hit("b1", shared)])
    llm = FakeLLMProvider(
        '{"score":90}', '{"score":25}', REWRITE_JSON, '{"score":40}', '{"selected_arxiv_ids":["2401.00001"]}'
    )
    deps, _, _ = dependencies(llm=llm, results=[first, second], live=live_result("2401.00001"), processor=processor)

    result = await run(deps)

    assert final_ids(result) == [("local", "b1"), ("live_arxiv", "live::2401.00001::chunk::001")]
    assert result["transient_live_evidence"] == (duplicate, distinct)
    assert result["execution_events"][20].metadata.model_dump(exclude_none=True) == {
        "local_candidate_count": 1,
        "live_candidate_count": 2,
        "merged_candidate_count": 2,
        "final_source_count": 2,
    }


@pytest.mark.anyio
async def test_source_specific_state_is_preserved_alongside_final_evidence() -> None:
    candidates = live_result("2501.00001", "2501.00002")
    deps, _, _, second = exhausted_dependencies(live=candidates, selection=SELECT_TWO)

    result = await run(deps)

    assert result["local_retrieval_result"] == second
    assert result["evidence_grade"] == EvidenceGrade(score=40, sufficient=False, source_count=1)
    assert result["evidence_sufficient"] is False
    assert result["live_search_result"] == candidates
    assert result["live_selected_papers"] == candidates.candidates
    assert result["transient_live_evidence"] == tuple(make_chunk(paper) for paper in candidates.candidates)
    assert all(isinstance(candidate, AgentEvidenceCandidate) for candidate in result["final_evidence"])
    assert result["generation_result"].answer == result["generated_answer"] == ANSWER
    assert (result["generation_result"].model, result["generation_result"].prompt_tokens) == ("fake-model", 111)
    assert result["generation_result"].completion_tokens == 22
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    assert result["grounding_passed"] is True
    assert result["grounding_attempts"] == 1


@pytest.mark.anyio
async def test_no_usable_final_evidence_is_insufficient_without_generation() -> None:
    hit = make_hit("c1", "   ")
    deps, provider, _ = dependencies(result=retrieval_result(hits=[hit]))
    grader = Mock()
    grader.grade = AsyncMock(return_value=EvidenceGrade(score=90, sufficient=True, source_count=1))

    with patch("src.services.agent.graph.EvidenceSufficiencyGrader", return_value=grader):
        result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *RERANKED, S.GRAPH_COMPLETED]
    assert S.GRAPH_FAILED not in statuses(result)
    assert result["final_evidence"] == ()
    assert provider.generation_calls == []
    assert deps.embedding_provider.queries == []
    assert result["generated_answer"] is None
    assert result["generation_result"] is None
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None
    assert result["execution_events"][8].metadata.final_source_count == 0


@pytest.mark.anyio
async def test_evidence_that_cannot_fit_the_context_budget_is_insufficient_not_failure() -> None:
    deps, provider, _ = dependencies()
    tiny = dataclasses.replace(deps, evidence_context_builder=EvidenceContextBuilder(max_context_tokens=1))
    grader = Mock()
    grader.grade = AsyncMock(return_value=EvidenceGrade(score=90, sufficient=True, source_count=1))

    with patch("src.services.agent.graph.EvidenceSufficiencyGrader", return_value=grader):
        result = await run(tiny)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *RERANKED, S.GENERATION_STARTED, S.GRAPH_COMPLETED]
    assert provider.generation_calls == []
    assert result["generated_answer"] is None
    assert result["terminal_reason"] == TerminalReason.INSUFFICIENT_EVIDENCE
    assert result["error_category"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("live_path", [False, True])
async def test_rerank_failure_is_distinct_from_insufficient_evidence(live_path: bool) -> None:
    embedding = FakeEmbeddingProvider(error=RuntimeError("private embedding detail"))
    if live_path:
        deps, provider, _, _ = exhausted_dependencies()
        deps = dataclasses.replace(deps, embedding_provider=embedding)
        expected = [*LIVE_PROCESSED, *RERANK_FAILED_TAIL]
    else:
        deps, provider, _ = dependencies(embedding=embedding)
        expected = [*LOCAL_SUFFICIENT, *RERANK_FAILED_TAIL]

    result = await run(deps)

    assert statuses(result) == expected
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert provider.generation_calls == []
    assert result["final_evidence"] is None
    assert result["generated_answer"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.EVIDENCE_RERANK_FAILURE
    assert "private embedding detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        "Private generation detail cites an unknown source [S9].",
        "Private generation detail cites nothing at all.",
        "Private generation detail uses a malformed label [S0].",
        "Private generation detail uses a malformed label [S 1].",
    ],
)
async def test_structurally_invalid_answer_is_a_generation_failure_not_a_grounding_failure(answer) -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":85}', answer=answer))

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATION_FAILED_TAIL]
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert S.GROUNDING_STARTED not in statuses(result) and S.GROUNDING_FAILED not in statuses(result)
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (1, 0)
    assert result["generated_answer"] is None
    assert result["generation_result"] is None
    assert (result["grounding_passed"], result["grounding_attempts"]) == (None, 0)
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.ANSWER_VALIDATION_FAILURE
    serialized = repr({key: value for key, value in result.items() if key != "local_retrieval_result"})
    assert "rivate generation detail" not in serialized


@pytest.mark.anyio
async def test_generation_provider_failure_is_controlled() -> None:
    answer = RuntimeError("private generation detail")
    deps, provider, _ = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":85}', answer=answer))

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATION_FAILED_TAIL]
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert S.GROUNDING_STARTED not in statuses(result)
    assert len(provider.generation_calls) == 1
    assert result["final_evidence"] is not None
    assert result["generated_answer"] is None
    assert result["generation_result"] is None
    assert result["grounding_passed"] is None
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.GENERATION_FAILURE
    serialized = repr({key: value for key, value in result.items() if key != "local_retrieval_result"})
    assert "rivate generation detail" not in serialized


@pytest.mark.anyio
async def test_citation_free_insufficiency_answer_is_a_valid_generation() -> None:
    answer = "The available evidence is insufficient to answer the question."
    deps, _, _ = dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":85}', answer=answer))

    result = await run(deps)

    assert statuses(result) == SUFFICIENT_PATH
    assert result["generated_answer"] == answer
    assert result["generation_result"].cited_labels == ()
    assert result["terminal_reason"] is None


@pytest.mark.anyio
async def test_generation_runs_once_and_reuses_the_existing_rag_prompt_and_limits() -> None:
    deps, provider, _ = dependencies()

    result = await run(deps, "How is NLP used in clinical decision support?")

    assert len(provider.generation_calls) == 1
    messages, kwargs = provider.generation_calls[0]
    assert messages[0].content == RAG_SYSTEM_PROMPT
    assert "BEGIN USER QUESTION (untrusted data)\nHow is NLP used in clinical decision support?" in messages[1].content
    assert f"[S1] Private Paper Title\narXiv ID: 2401.00001\nChunk ID: c1\nEvidence: {EVIDENCE_TEXT}" in messages[1].content
    assert kwargs == {"temperature": 0.1, "max_tokens": 64}
    assert result["generation_result"].retrieval_mode == "hybrid"
    assert statuses(result).count(S.GENERATION_STARTED) == 1


LOW, HIGH = '{"score":30}', '{"score":88}'


def generation_user_prompt(call) -> str:
    return call[0][1].content


def grounding_payload(call) -> dict:
    return json.loads(call[0][1].content.split("\n", 1)[1])


@pytest.mark.anyio
async def test_first_grounding_pass_completes_without_regeneration() -> None:
    deps, provider, retrieval = dependencies(llm=FakeLLMProvider(grounding=HIGH))

    result = await run(deps, "How is NLP used in clinical decision support?")

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, *GROUNDED, S.GRAPH_COMPLETED]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (1, 1)
    assert provider.grounding_calls[0][1] == {"temperature": 0.0, "max_tokens": 1024}
    assert result["grounding_attempts"] == 1
    assert result["grounding_passed"] is True
    assert result["grounding_result"] == AnswerGroundingResult(score=88, passed=True)
    assert result["generated_answer"] == ANSWER
    assert result["terminal_reason"] is None
    assert result["error_category"] is None
    grounding_events = result["execution_events"][11:13]
    assert [event.metadata.model_dump(exclude_none=True) for event in grounding_events] == [
        {"grounding_attempt": 1, "final_source_count": 1},
        {"grounding_attempt": 1, "grounding_score": 88, "grounding_passed": True},
    ]


@pytest.mark.anyio
async def test_grounding_grades_the_answer_against_the_sources_sent_to_generation() -> None:
    hits = [
        make_hit("c1", "first " + "alpha " * 40),
        make_hit("c2", "second " + "beta " * 40),
        make_hit("c3", "third " + "omega " * 400),
        make_hit("c4", "fourth passage never sent"),
        make_hit("c5", "fifth passage never sent"),
    ]
    deps, provider, _ = dependencies(result=retrieval_result(hits=hits), max_context_tokens=260)
    question = "How is NLP used in clinical decision support?"

    result = await run(deps, question)

    assert len(result["final_evidence"]) == 5
    sent_to_generation = result["generation_result"].sources
    assert [source.label for source in sent_to_generation] == ["[S1]", "[S2]", "[S3]"]
    assert sent_to_generation[2].truncated is True
    payload = grounding_payload(provider.grounding_calls[0])
    assert payload["original_question"] == question
    assert payload["generated_answer"] == ANSWER
    assert [source["label"] for source in payload["sources"]] == ["[S1]", "[S2]", "[S3]"]
    assert [source["content"] for source in payload["sources"]] == [source.content for source in sent_to_generation]
    assert "never sent" not in provider.grounding_calls[0][0][1].content
    assert result["execution_events"][11].metadata.final_source_count == 3
    assert result["execution_events"][9].metadata.final_source_count == 5


@pytest.mark.anyio
async def test_failed_grounding_regenerates_once_from_the_same_evidence_then_passes() -> None:
    llm = FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH))
    deps, provider, retrieval = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *REGENERATE_PASS_TAIL]
    assert [event.sequence for event in result["execution_events"]] == list(range(18))
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 2)
    assert len(provider.calls) == 2
    retrieval.search.assert_called_once()
    assert len(deps.embedding_provider.passage_batches) == 1
    assert result["grounding_attempts"] == 2
    assert result["grounding_passed"] is True
    assert result["grounding_result"] == AnswerGroundingResult(score=88, passed=True)
    assert result["generated_answer"] == REGENERATED_ANSWER
    assert result["generation_result"].answer == REGENERATED_ANSWER
    assert ANSWER not in repr({key: value for key, value in result.items() if key != "execution_events"})
    assert result["terminal_reason"] is None
    assert result["error_category"] is None

    first_prompt, second_prompt = (generation_user_prompt(call) for call in provider.generation_calls)
    assert second_prompt.startswith(first_prompt)
    assert "BEGIN PREVIOUS ANSWER (untrusted data)\n" + ANSWER in second_prompt
    assert "PREVIOUS ANSWER" not in first_prompt
    assert provider.generation_calls[0][0][0] == provider.generation_calls[1][0][0]
    assert provider.generation_calls[0][1] == provider.generation_calls[1][1]
    first_grade, second_grade = (grounding_payload(call) for call in provider.grounding_calls)
    assert first_grade["generated_answer"] == ANSWER
    assert second_grade["generated_answer"] == REGENERATED_ANSWER
    assert first_grade["sources"] == second_grade["sources"]

    tail = [(event.status, event.metadata.model_dump(exclude_none=True)) for event in result["execution_events"][9:17]]
    assert tail == [
        (S.GENERATION_STARTED, {"generation_attempt": 1, "final_source_count": 1}),
        (
            S.GENERATION_COMPLETED,
            {"generation_attempt": 1, "final_source_count": 1, "prompt_tokens": 111, "completion_tokens": 22},
        ),
        (S.GROUNDING_STARTED, {"grounding_attempt": 1, "final_source_count": 1}),
        (S.GROUNDING_FAILED, {"grounding_attempt": 1, "grounding_score": 30, "grounding_passed": False}),
        (S.GENERATION_STARTED, {"generation_attempt": 2, "final_source_count": 1}),
        (
            S.GENERATION_COMPLETED,
            {"generation_attempt": 2, "final_source_count": 1, "prompt_tokens": 111, "completion_tokens": 22},
        ),
        (S.GROUNDING_STARTED, {"grounding_attempt": 2, "final_source_count": 1}),
        (S.GROUNDING_PASSED, {"grounding_attempt": 2, "grounding_score": 88, "grounding_passed": True}),
    ]


@pytest.mark.anyio
async def test_grounding_failing_twice_ends_as_grounding_failed_without_a_third_attempt() -> None:
    llm = FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, '{"score":45}', HIGH))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *REGENERATE_FAIL_TAIL]
    assert S.GRAPH_FAILED not in statuses(result)
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 2)
    assert result["grounding_attempts"] == 2
    assert result["grounding_passed"] is False
    assert result["grounding_result"] == AnswerGroundingResult(score=45, passed=False)
    assert result["generated_answer"] == REGENERATED_ANSWER
    assert result["terminal_reason"] == TerminalReason.GROUNDING_FAILED
    assert result["error_category"] is None


@pytest.mark.anyio
async def test_single_grounding_attempt_config_fails_without_regeneration() -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(grounding=LOW))

    result = await run(deps, config=AgentGraphConfig(max_grounding_attempts=1))

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, *UNGROUNDED, S.GRAPH_COMPLETED]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (1, 1)
    assert result["grounding_attempts"] == 1
    assert result["grounding_passed"] is False
    assert result["terminal_reason"] == TerminalReason.GROUNDING_FAILED
    assert result["error_category"] is None


@pytest.mark.anyio
async def test_grounding_attempts_follow_configuration_and_never_exceed_it() -> None:
    llm = FakeLLMProvider(answer=("First [S1].", "Second [S1].", "Third [S1].", "Fourth [S1]."), grounding=LOW)
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps, config=AgentGraphConfig(max_grounding_attempts=3))

    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (3, 3)
    assert result["grounding_attempts"] == 3
    assert result["generated_answer"] == "Third [S1]."
    assert result["terminal_reason"] == TerminalReason.GROUNDING_FAILED
    assert "BEGIN PREVIOUS ANSWER (untrusted data)\nSecond [S1]." in generation_user_prompt(provider.generation_calls[2])
    assert "First [S1]." not in generation_user_prompt(provider.generation_calls[2])


@pytest.mark.anyio
@pytest.mark.parametrize(("threshold", "passed"), [(88, True), (89, False)])
async def test_routing_uses_configured_grounding_threshold(threshold: int, passed: bool) -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(grounding=HIGH))
    config = AgentGraphConfig(answer_grounding_threshold=threshold, max_grounding_attempts=1)

    result = await run(deps, config=config)

    assert result["grounding_passed"] is passed
    assert result["terminal_reason"] == (None if passed else TerminalReason.GROUNDING_FAILED)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "grounding",
    [RuntimeError("private grounding detail"), "private grounding detail, not json", '{"score":87,"why":"private grounding detail"}'],
)
async def test_grounding_grader_execution_failure_is_not_a_low_score(grounding) -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(grounding=grounding))

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, S.GROUNDING_STARTED, S.GRAPH_FAILED]
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (1, 1)
    assert result["grounding_attempts"] == 1
    assert result["grounding_passed"] is None
    assert result["grounding_result"] is None
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.ANSWER_GROUNDING_FAILURE
    assert "private grounding detail" not in repr(
        {key: value for key, value in result.items() if key != "local_retrieval_result"}
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "regenerated",
    [
        "Private regeneration detail with unknown label [S9].",
        "Private regeneration detail with malformed label [S0].",
        "Private regeneration detail without any citation.",
    ],
)
async def test_structurally_invalid_regeneration_is_a_generation_failure(regenerated) -> None:
    llm = FakeLLMProvider(answer=(ANSWER, regenerated), grounding=(LOW, HIGH))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, *UNGROUNDED, S.GENERATION_STARTED, S.GRAPH_FAILED]
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 1)
    assert (result["grounding_attempts"], result["grounding_passed"]) == (1, False)
    assert result["generated_answer"] is None
    assert result["generation_result"] is None
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.ANSWER_VALIDATION_FAILURE
    serialized = repr({key: value for key, value in result.items() if key != "local_retrieval_result"})
    assert "rivate regeneration detail" not in serialized
    assert ANSWER not in serialized


@pytest.mark.anyio
async def test_regeneration_provider_failure_is_generation_failure_without_second_grounding() -> None:
    regenerated = RuntimeError("private regeneration detail")
    llm = FakeLLMProvider(answer=(ANSWER, regenerated), grounding=(LOW, HIGH))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, *UNGROUNDED, S.GENERATION_STARTED, S.GRAPH_FAILED]
    assert S.GRAPH_COMPLETED not in statuses(result)
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 1)
    assert result["grounding_attempts"] == 1
    assert result["grounding_passed"] is False
    assert result["generated_answer"] is None
    assert result["generation_result"] is None
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.GENERATION_FAILURE
    serialized = repr({key: value for key, value in result.items() if key != "local_retrieval_result"})
    assert "rivate regeneration detail" not in serialized
    assert ANSWER not in serialized


@pytest.mark.anyio
async def test_regeneration_on_the_live_path_reuses_evidence_and_makes_no_new_retrieval_or_rerank() -> None:
    candidates = live_result("2501.00001", "2501.00002")
    first = retrieval_result(hits=[make_hit("a1", "attempt one evidence")])
    second = retrieval_result(hits=[make_hit("b1", "attempt two evidence")])
    llm = FakeLLMProvider(
        '{"score":90}',
        '{"score":25}',
        REWRITE_JSON,
        '{"score":40}',
        SELECT_TWO,
        answer=(ANSWER, REGENERATED_ANSWER),
        grounding=(LOW, HIGH),
    )
    deps, provider, retrieval = dependencies(llm=llm, results=[first, second], live=candidates)

    result = await run(deps, INDIA_QUESTION)

    assert statuses(result) == [*LIVE_PROCESSED, *REGENERATE_PASS_TAIL]
    assert retrieval.search.call_count == 2
    assert len(provider.calls) == 5
    assert statuses(result).count(S.QUERY_REWRITTEN) == 1
    deps.live_search_service.search.assert_awaited_once()
    deps.live_document_processor.process.assert_awaited_once()
    assert len(deps.embedding_provider.queries) == 1
    assert len(deps.embedding_provider.passage_batches) == 1
    assert statuses(result).count(S.EVIDENCE_RERANK_STARTED) == 1
    assert result["retrieval_attempts"] == 2

    first_prompt, second_prompt = (generation_user_prompt(call) for call in provider.generation_calls)
    evidence_block = first_prompt[first_prompt.index("BEGIN RETRIEVED EVIDENCE") : first_prompt.index("END RETRIEVED EVIDENCE")]
    assert evidence_block in second_prompt
    assert second_prompt.count("[S1] ") == first_prompt.count("[S1] ") == 1
    first_grade, second_grade = (grounding_payload(call) for call in provider.grounding_calls)
    assert first_grade["sources"] == second_grade["sources"]
    assert [source["label"] for source in second_grade["sources"]] == ["[S1]", "[S2]", "[S3]"]
    assert first_grade["original_question"] == second_grade["original_question"] == INDIA_QUESTION
    assert [source.source_type for source in result["generation_result"].sources] == ["local", "live_arxiv", "live_arxiv"]
    assert result["generated_answer"] == REGENERATED_ANSWER
    assert result["grounding_passed"] is True


@pytest.mark.anyio
async def test_citation_free_insufficiency_answer_can_pass_grounding() -> None:
    answer = "The available evidence is insufficient to answer the question."
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=answer, grounding=HIGH))

    result = await run(deps)

    assert statuses(result) == SUFFICIENT_PATH
    assert result["generation_result"].cited_labels == ()
    assert grounding_payload(provider.grounding_calls[0])["generated_answer"] == answer
    assert result["grounding_passed"] is True
    assert result["generated_answer"] == answer
    assert result["terminal_reason"] is None
    assert len(provider.generation_calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(("attempts", "maximum"), [(2, 2), (1, 1), (5, 2)])
async def test_grounding_node_refuses_to_exceed_configured_attempts(attempts: int, maximum: int) -> None:
    deps, provider, _ = dependencies()
    state = create_initial_agent_state("AI question")
    state["grounding_attempts"] = attempts
    state["generated_answer"] = "Seeded previous answer [S1]."
    config = AgentGraphConfig(max_grounding_attempts=maximum)

    result = await build_agent_graph(dependencies=deps, config=config).ainvoke(state)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, S.GRAPH_FAILED]
    assert provider.grounding_calls == []
    assert len(provider.generation_calls) == 1
    assert result["grounding_attempts"] == attempts
    assert result["terminal_reason"] == TerminalReason.INTERNAL_ERROR
    assert result["error_category"] == AgentErrorCategory.INTERNAL_FAILURE


@pytest.mark.anyio
async def test_stale_grounding_decision_is_cleared_when_the_answer_is_regenerated() -> None:
    llm = FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH))
    deps, _, _ = dependencies(llm=llm)
    graph = build_agent_graph(dependencies=deps)

    snapshots = {}
    async for state in graph.astream(create_initial_agent_state("AI question"), stream_mode="values"):
        if state["execution_events"]:
            snapshots[len(state["execution_events"])] = state

    after_first_grade = snapshots[len(LOCAL_SUFFICIENT) + len(GENERATED) + 2]
    assert after_first_grade["generated_answer"] == ANSWER
    assert after_first_grade["grounding_passed"] is False
    assert after_first_grade["grounding_result"].score == 30

    after_regeneration = snapshots[len(LOCAL_SUFFICIENT) + len(GENERATED) + 4]
    assert after_regeneration["generated_answer"] == REGENERATED_ANSWER
    assert after_regeneration["grounding_attempts"] == 1
    assert after_regeneration["grounding_passed"] is None
    assert after_regeneration["grounding_result"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "grounding",
    [
        HIGH,
        (LOW, HIGH),
        (LOW, LOW),
        '{"score":88,"reasoning":"secret grounding rationale"}',
        RuntimeError("secret grounding rationale"),
    ],
)
async def test_grounding_and_regeneration_events_are_deterministic_and_content_free(grounding) -> None:
    question = "How is NLP used in clinical decision support?"

    async def events():
        llm = FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=grounding)
        deps, _, _ = dependencies(llm=llm)
        result = await run(deps, question)
        return [event.model_dump(mode="json") for event in result["execution_events"]]

    first, second = await events(), await events()

    assert first == second
    serialized = repr(first)
    for forbidden in (
        question,
        EVIDENCE_TEXT,
        "Private Paper Title",
        ANSWER,
        REGENERATED_ANSWER,
        "Private generated",
        "Private regenerated",
        "UNTRUSTED_GROUNDING_INPUT_JSON",
        "answer-grounding grader",
        "PREVIOUS ANSWER",
        "insufficiently grounded",
        '{"score"',
        "secret grounding rationale",
        "local-v1",
        "fake-model",
    ):
        assert forbidden not in serialized


class RecordingObservation:
    def __init__(self, name: str = "agent.request", kind: str = "span", metadata=None) -> None:
        self.name, self.kind = name, kind
        self.metadata = [metadata or {}]
        self.error_types: list[str] = []
        self.generation_records: list[dict] = []
        self.children: list[RecordingObservation] = []
        self.end_count = 0

    def start_span(self, *, name, metadata=None):
        child = RecordingObservation(name, "span", metadata)
        self.children.append(child)
        return child

    def start_generation(self, *, name, model=None, metadata=None):
        child = RecordingObservation(name, "generation", metadata)
        self.children.append(child)
        return child

    def record_generation(self, **kwargs) -> None:
        self.generation_records.append(kwargs)

    def update(self, *, metadata=None, error_type=None) -> None:
        if metadata:
            self.metadata.append(metadata)
        if error_type:
            self.error_types.append(error_type)

    def end(self) -> None:
        self.end_count += 1

    @property
    def generations(self) -> list["RecordingObservation"]:
        return [child for child in self.children if child.kind == "generation"]


async def run_observed(deps, observation, question: str = "AI question", config: AgentGraphConfig | None = None):
    graph = build_agent_graph(dependencies=deps, config=config)
    return await graph.ainvoke(
        create_initial_agent_state(question), config={"configurable": {AGENT_OBSERVATION_KEY: observation}}
    )


PROMPT_IDENTITIES = local_agent_prompt_bundle().identities


@pytest.mark.anyio
async def test_every_llm_call_on_the_local_path_is_a_generation_observation_with_prompt_identity() -> None:
    deps, _, _ = dependencies()
    deps = dataclasses.replace(deps, llm_telemetry=LLMTelemetry(provider_name="test-provider"))
    root = RecordingObservation()

    result = await run_observed(deps, root, "How is NLP used in clinical decision support?")

    assert statuses(result) == SUFFICIENT_PATH
    assert [generation.name for generation in root.generations] == [
        "agent.guardrail",
        "agent.evidence_grader",
        "rag.generation",
        "agent.answer_grounding",
    ]
    assert [child.name for child in root.children if child.kind == "span"] == ["rag.evidence", "rag.grounding"]
    expected = [
        PROMPT_IDENTITIES.guardrail,
        PROMPT_IDENTITIES.evidence_grader,
        PROMPT_IDENTITIES.generation,
        PROMPT_IDENTITIES.answer_grounding,
    ]
    for generation, identity in zip(root.generations, expected, strict=True):
        started = generation.metadata[0]
        assert {key: started[key] for key in identity.as_metadata()} == identity.as_metadata()
        assert generation.generation_records == [
            {"model": "fake-model", "input_tokens": 111, "output_tokens": 22, "cost": None}
        ]
        finished = generation.metadata[1]
        assert (finished["model"], finished["prompt_tokens"], finished["completion_tokens"]) == ("fake-model", 111, 22)
        assert finished["latency_ms"] >= 0
        assert "cost_usd" not in finished
        assert generation.end_count == 1 and generation.error_types == []
    # Classifier calls use the graph's telemetry; generation uses its own service's.
    assert [generation.metadata[0].get("provider") for generation in root.generations] == [
        "test-provider",
        "test-provider",
        None,
        "test-provider",
    ]
    assert root.generations[1].metadata[0]["retrieval_attempt"] == 1
    assert root.generations[2].metadata[0]["generation_attempt"] == 1
    assert root.generations[3].metadata[0]["grounding_attempt"] == 1


@pytest.mark.anyio
async def test_regeneration_is_a_separate_generation_observation_with_its_own_identity() -> None:
    llm = FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH))
    deps, _, _ = dependencies(llm=llm)
    root = RecordingObservation()

    result = await run_observed(deps, root)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *REGENERATE_PASS_TAIL]
    names = [generation.name for generation in root.generations]
    assert names == [
        "agent.guardrail",
        "agent.evidence_grader",
        "rag.generation",
        "agent.answer_grounding",
        "rag.regeneration",
        "agent.answer_grounding",
    ]
    first, regenerated = root.generations[2], root.generations[4]
    assert first.metadata[0]["generation_attempt"] == 1
    assert first.metadata[0]["prompt_name"] == "rag-answer-generation"
    assert regenerated.metadata[0]["generation_attempt"] == 2
    assert regenerated.metadata[0]["prompt_name"] == "rag-answer-regeneration"
    assert regenerated.metadata[0]["prompt_fingerprint"] == PROMPT_IDENTITIES.regeneration.fingerprint
    assert regenerated.metadata[0]["base_prompt_name"] == "rag-answer-generation"
    assert regenerated.metadata[0]["base_prompt_fingerprint"] == PROMPT_IDENTITIES.generation.fingerprint
    assert "base_prompt_name" not in first.metadata[0]
    assert first is not regenerated and first.end_count == regenerated.end_count == 1
    assert [root.generations[index].metadata[0]["grounding_attempt"] for index in (3, 5)] == [1, 2]


@pytest.mark.anyio
async def test_live_path_observes_rewrite_and_selection_and_records_known_cost() -> None:
    deps, _, _, _ = exhausted_dependencies()
    rates = TokenCostRates(input_per_million=1.0, output_per_million=2.0)
    deps = dataclasses.replace(deps, llm_telemetry=LLMTelemetry(provider_name="test-provider", cost_rates=rates))
    root = RecordingObservation()

    result = await run_observed(deps, root, INDIA_QUESTION)

    assert statuses(result) == LIVE_EVIDENCE_PATH
    assert [generation.name for generation in root.generations] == [
        "agent.guardrail",
        "agent.evidence_grader",
        "agent.query_rewrite",
        "agent.evidence_grader",
        "agent.live_selector",
        "rag.generation",
        "agent.answer_grounding",
    ]
    assert [root.generations[index].metadata[0]["retrieval_attempt"] for index in (1, 3)] == [1, 2]
    assert root.generations[2].metadata[0]["prompt_name"] == "agent-query-rewrite"
    assert root.generations[4].metadata[0]["prompt_name"] == "agent-live-paper-selector"
    rewrite_cost = root.generations[2].generation_records[0]["cost"]
    assert rewrite_cost.total_cost == pytest.approx(111 * 1.0 / 1e6 + 22 * 2.0 / 1e6)
    assert root.generations[2].metadata[1]["cost_usd"] == pytest.approx(rewrite_cost.total_cost)
    assert root.generations[5].generation_records[0]["cost"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("grounding", [HIGH, (LOW, HIGH), RuntimeError("secret telemetry detail")])
async def test_generation_observations_never_contain_question_evidence_prompt_or_answer(grounding) -> None:
    llm = FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=grounding)
    deps, _, _ = dependencies(llm=llm)
    root = RecordingObservation()
    question = "PRIVATE_QUESTION_SENTINEL about clinical NLP?"

    await run_observed(deps, root, question)

    recorded = repr([(child.name, child.metadata, child.generation_records, child.error_types) for child in root.children])
    for forbidden in (
        question,
        "PRIVATE_QUESTION_SENTINEL",
        EVIDENCE_TEXT,
        "Private Paper Title",
        ANSWER,
        REGENERATED_ANSWER,
        "UNTRUSTED_",
        "You are",
        "BEGIN USER QUESTION",
        "secret telemetry detail",
        '{"score"',
    ):
        assert forbidden not in recorded
    if isinstance(grounding, Exception):
        assert root.generations[-1].name == "agent.answer_grounding"
        assert root.generations[-1].error_types == ["RuntimeError"]
        assert root.generations[-1].generation_records == []


@pytest.mark.anyio
async def test_observability_failures_and_absence_do_not_change_agent_results() -> None:
    def comparable(result) -> dict:
        return {key: value for key, value in result.items() if key != "generation_result"}

    baseline_deps, _, _ = dependencies(llm=FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH)))
    baseline = await run(baseline_deps)

    broken = Mock()
    broken.start_generation.side_effect = RuntimeError("telemetry down")
    broken.start_span.side_effect = RuntimeError("telemetry down")
    half_broken = Mock()
    for child in (half_broken.start_generation.return_value, half_broken.start_span.return_value):
        child.record_generation.side_effect = RuntimeError("telemetry down")
        child.update.side_effect = RuntimeError("telemetry down")
        child.end.side_effect = RuntimeError("telemetry down")

    for observation in (broken, half_broken, RecordingObservation(), None):
        deps, _, _ = dependencies(llm=FakeLLMProvider(answer=(ANSWER, REGENERATED_ANSWER), grounding=(LOW, HIGH)))
        result = await run_observed(deps, observation)
        assert comparable(result) == comparable(baseline)
        assert result["generation_result"].answer == REGENERATED_ANSWER


@pytest.mark.anyio
async def test_state_records_local_prompt_identities_by_default_without_template_text() -> None:
    deps, _, _ = dependencies()

    result = await run(deps)
    rejected = await run(dependencies(llm=FakeLLMProvider('{"score":10}'))[0])

    assert result["prompt_identities"] == PROMPT_IDENTITIES
    assert rejected["prompt_identities"] == PROMPT_IDENTITIES
    assert create_initial_agent_state("AI question")["prompt_identities"] is None
    assert "You are" not in repr(result["prompt_identities"])


class MarkerLLM:
    """Answers by what each call's user payload is, so managed system prompts can differ freely."""

    def __init__(self) -> None:
        self.system_prompts: dict[str, list[str]] = {}

    async def complete(self, messages, **kwargs):
        system, user = messages[0].content, messages[1].content
        for marker, stage, content in (
            ("UNTRUSTED_QUESTION_JSON", "guardrail", '{"score":90}'),
            ("UNTRUSTED_GRADING_INPUT_JSON", "evidence_grader", '{"score":25}'),
            ("UNTRUSTED_REWRITE_INPUT_JSON", "query_rewrite", REWRITE_JSON),
            ("UNTRUSTED_SELECTION_INPUT_JSON", "live_selector", SELECT_FIRST),
            ("UNTRUSTED_GROUNDING_INPUT_JSON", "answer_grounding", '{"score":30}'),
            ("BEGIN PREVIOUS ANSWER", "regeneration", REGENERATED_ANSWER),
            ("BEGIN USER QUESTION", "generation", ANSWER),
        ):
            if marker in user:
                self.system_prompts.setdefault(stage, []).append(system)
                self.last_user = {**getattr(self, "last_user", {}), stage: user}
                return LLMCompletion(content=content, model="fake-model", prompt_tokens=1, completion_tokens=1)
        raise AssertionError("unexpected LLM call")


class ManagedResolver:
    async def resolve(self, definition, *, label="production"):
        return ResolvedPrompt(
            name=definition.name,
            content=f"MANAGED[{definition.name}] " + " ".join(definition.required_markers),
            version="langfuse-v5",
            label=label,
            source="langfuse",
        )


@pytest.mark.anyio
async def test_graph_executes_with_exactly_the_resolved_prompt_bundle_it_was_built_with() -> None:
    bundle = await resolve_agent_prompt_bundle(ManagedResolver())
    llm = MarkerLLM()
    first = retrieval_result(hits=[make_hit("a1", "attempt one evidence")])
    second = retrieval_result(hits=[make_hit("b1", "attempt two evidence")])
    deps, _, _ = dependencies(llm=FakeLLMProvider(), results=[first, second])
    deps = dataclasses.replace(
        deps,
        llm_provider=llm,
        rag_generation_service=RAGGenerationService(
            llm_provider=llm,
            evidence_builder=deps.evidence_context_builder,
            max_completion_tokens=64,
            token_safety_margin=16,
        ),
    )
    root = RecordingObservation()

    graph = build_agent_graph(dependencies=deps, prompts=bundle)
    result = await graph.ainvoke(
        create_initial_agent_state(INDIA_QUESTION), config={"configurable": {AGENT_OBSERVATION_KEY: root}}
    )

    assert statuses(result) == [*LIVE_PROCESSED, *REGENERATE_FAIL_TAIL]
    assert result["terminal_reason"] == TerminalReason.GROUNDING_FAILED
    for stage in ("guardrail", "evidence_grader", "query_rewrite", "live_selector", "generation", "answer_grounding"):
        name = AGENT_PROMPT_DEFINITIONS[stage].name
        assert set(llm.system_prompts[stage]) == {getattr(bundle, stage).content}
        assert f"MANAGED[{name}]" in llm.system_prompts[stage][0]
    assert set(llm.system_prompts["regeneration"]) == {bundle.generation.content}
    assert llm.last_user["regeneration"].endswith(bundle.regeneration.content)
    assert result["prompt_identities"] == bundle.identities
    assert {identity["version"] for identity in bundle.identities.as_cache_payload().values()} == {"langfuse-v5"}
    assert result["prompt_identities"] != PROMPT_IDENTITIES
    observed = {generation.name: generation.metadata[0] for generation in root.generations}
    assert observed["agent.guardrail"]["prompt_version"] == "langfuse-v5"
    assert observed["agent.guardrail"]["prompt_fingerprint"] == bundle.identities.guardrail.fingerprint
    assert observed["rag.generation"]["prompt_fingerprint"] == bundle.identities.generation.fingerprint
    assert observed["rag.regeneration"]["prompt_fingerprint"] == bundle.identities.regeneration.fingerprint
    assert "MANAGED[" not in repr([(g.metadata, g.generation_records) for g in root.generations])
    assert deps.rag_generation_service.prompt_builder.identity == PROMPT_IDENTITIES.generation


@pytest.mark.anyio
async def test_partially_managed_bundle_keeps_local_prompts_for_the_rest() -> None:
    local = local_agent_prompt_bundle()
    managed_guardrail = ResolvedPrompt(
        name="agent-guardrail",
        content='Managed guardrail. Return {"score": <0-100>}.',
        version="langfuse-v2",
        label="production",
        source="langfuse",
    )
    bundle = dataclasses.replace(local, guardrail=managed_guardrail)
    llm = MarkerLLM()
    deps, _, _ = dependencies()
    deps = dataclasses.replace(deps, llm_provider=llm)

    result = await build_agent_graph(dependencies=deps, config=SINGLE_ATTEMPT, prompts=bundle).ainvoke(
        create_initial_agent_state("AI question")
    )

    assert llm.system_prompts["guardrail"] == [managed_guardrail.content]
    assert llm.system_prompts["evidence_grader"] == [local.evidence_grader.content]
    assert result["prompt_identities"].guardrail.version == "langfuse-v2"
    assert result["prompt_identities"].evidence_grader == PROMPT_IDENTITIES.evidence_grader
    assert local_prompt(AGENT_PROMPT_DEFINITIONS["guardrail"]) == local.guardrail


def scenario_dependencies():
    """One (dependencies, config) pair per distinct graph path."""
    return [
        (dependencies()[0], None),
        (dependencies(llm=FakeLLMProvider('{"score":90}', '{"score":5}'))[0], SINGLE_ATTEMPT),
        (dependencies(result=retrieval_result(hits=[]))[0], SINGLE_ATTEMPT),
        (dependencies(llm=FakeLLMProvider('{"score":1}'))[0], None),
        (dependencies(llm=FakeLLMProvider("invalid"))[0], None),
        (dependencies(retrieval_error=RuntimeError("down"))[0], None),
        (dependencies(llm=FakeLLMProvider('{"score":90}', "invalid"))[0], None),
        (dependencies(llm=FakeLLMProvider('{"score":90}', RuntimeError("down")))[0], None),
        (retry_dependencies(REWRITE_JSON, '{"score":88}')[0], None),
        (retry_dependencies(REWRITE_JSON, '{"score":40}', SELECT_FIRST)[0], None),
        (retry_dependencies("invalid")[0], None),
        (retry_dependencies(RuntimeError("down"))[0], None),
        (retry_dependencies(REWRITE_JSON, "invalid")[0], None),
        (retry_dependencies(REWRITE_JSON, results=[retrieval_result(), RuntimeError("down")])[0], None),
        (retry_dependencies(REWRITE_JSON, '{"score":40}')[0], LOCAL_ONLY),
        (exhausted_dependencies(live=live_result())[0], None),
        (exhausted_dependencies(live_error=LiveSearchError("down"))[0], None),
        (exhausted_dependencies(selection=SELECT_NONE)[0], None),
        (exhausted_dependencies(selection="invalid")[0], None),
        (exhausted_dependencies(processor=fake_processor(unusable={"2501.00001"}))[0], None),
        (exhausted_dependencies(processor=fake_processor(failing={"2501.00001"}))[0], None),
        (exhausted_dependencies(processor=fake_processor(error=RuntimeError("down")))[0], None),
        (dependencies(embedding=FakeEmbeddingProvider(error=RuntimeError("down")))[0], None),
        (dependencies(llm=FakeLLMProvider(answer=RuntimeError("down")))[0], None),
        (dependencies(llm=FakeLLMProvider(answer="No citation here."))[0], None),
        (dependencies(llm=FakeLLMProvider(grounding=LOW))[0], None),
        (dependencies(llm=FakeLLMProvider(grounding=(LOW, HIGH)))[0], None),
        (dependencies(llm=FakeLLMProvider(grounding="invalid"))[0], None),
        (dependencies(llm=FakeLLMProvider(grounding=(LOW, RuntimeError("down"))))[0], None),
        (dependencies(llm=FakeLLMProvider(answer=(ANSWER, RuntimeError("down")), grounding=LOW))[0], None),
        (dependencies(llm=FakeLLMProvider(grounding=LOW))[0], AgentGraphConfig(max_grounding_attempts=1)),
    ]


@pytest.mark.anyio
async def test_every_path_has_exactly_one_terminal_event_and_contiguous_sequences() -> None:
    for deps, config in scenario_dependencies():
        result = await run(deps, config=config)
        path = statuses(result)
        assert path.count(S.GRAPH_COMPLETED) + path.count(S.GRAPH_FAILED) == 1
        assert path[-1] in {S.GRAPH_COMPLETED, S.GRAPH_FAILED}
        assert [event.sequence for event in result["execution_events"]] == list(range(len(path)))
        assert path.count(S.LOCAL_RETRIEVAL_STARTED) <= 2
        assert path.count(S.QUERY_REWRITE_STARTED) <= 1
        assert path.count(S.LIVE_FALLBACK_STARTED) <= 1
        assert path.count(S.LIVE_SELECTION_STARTED) <= 1
        assert path.count(S.LIVE_DOCUMENT_PROCESSING_STARTED) <= 1
        assert path.count(S.EVIDENCE_RERANK_STARTED) <= 1
        assert path.count(S.GENERATION_STARTED) <= 2
        assert path.count(S.GROUNDING_STARTED) <= 2
        assert result["grounding_attempts"] == path.count(S.GROUNDING_STARTED) <= 2
        assert (result["generated_answer"] is not None) == (result["generation_result"] is not None)
        if S.GROUNDING_PASSED in path:
            assert path[-3:] == [S.GROUNDING_STARTED, S.GROUNDING_PASSED, S.GRAPH_COMPLETED]
            assert result["grounding_passed"] is True
        assert result["retrieval_attempts"] <= 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "responses",
    [
        ('{"score":85}',),
        ('{"score":85,"reasoning":"secret rationale"}',),
        ('{"score":20}', REWRITE_JSON, '{"score":88}'),
        ('{"score":20}', REWRITE_JSON, '{"score":30}', SELECT_FIRST),
        ('{"score":20}', REWRITE_JSON, '{"score":30}', '{"selected_arxiv_ids":["2501.00001"],"why":"secret rationale"}'),
        ('{"score":20}', f'{{"query":"{REWRITTEN}","rationale":"secret rationale"}}'),
    ],
)
async def test_events_are_deterministic_and_content_free(responses: tuple[str, ...]) -> None:
    question = "How is NLP used in clinical decision support?"

    async def events():
        deps, _, _ = dependencies(llm=FakeLLMProvider('{"score":95}', *responses))
        result = await run(deps, question)
        return [event.model_dump(mode="json") for event in result["execution_events"]]

    first, second = await events(), await events()

    assert first == second
    assert first[2]["metadata"]["guardrail_score"] == 95
    assert first[4]["metadata"]["source_count"] == 1
    serialized = repr(first)
    for forbidden in (
        question,
        REWRITTEN,
        EVIDENCE_TEXT,
        "Private Paper Title",
        "Private Live Title",
        "private live abstract",
        "Private Author",
        "cs.CY",
        "arxiv.org",
        "2501.00001",
        "live query",
        CHUNK_TEXT,
        "Private Section",
        "UNTRUSTED_SELECTION_INPUT_JSON",
        "research-paper selector",
        ANSWER,
        "Private generated synthesis",
        "research assistant",
        "BEGIN RETRIEVED EVIDENCE",
        "fake-model",
        "relevance_score",
        "UNTRUSTED_QUESTION_JSON",
        "UNTRUSTED_GRADING_INPUT_JSON",
        "UNTRUSTED_REWRITE_INPUT_JSON",
        "evidence-sufficiency grader",
        "retrieval-query rewriter",
        *responses,
        "secret rationale",
    ):
        assert forbidden not in serialized


# --- Completion finish reason and bounded length recovery ---

PARTIAL = "Private partial synthesis cut off mid sentence [S1] and then"
PARTIAL_REGENERATION = "Private partial regeneration cut off [S1] and"
RECOVERED_ANSWER = "Private recovered complete synthesis [S1]."
RECOVERY = S.GENERATION_LENGTH_RECOVERY_STARTED
NORMAL_MAX_TOKENS, RECOVERY_MAX_TOKENS = 64, 128


def serialized_state(result) -> str:
    return repr({key: value for key, value in result.items() if key != "local_retrieval_result"})


def max_tokens(provider) -> list[int]:
    return [kwargs["max_tokens"] for _, kwargs in provider.generation_calls]


@pytest.mark.anyio
async def test_normal_stop_finish_reason_follows_the_existing_path() -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=Stopped(ANSWER)))

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *GENERATED, *GROUNDED, S.GRAPH_COMPLETED]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (1, 1)
    assert max_tokens(provider) == [NORMAL_MAX_TOKENS]
    assert result["generation_result"].finish_reason == "stop"
    assert (result["generated_answer"], result["grounding_passed"], result["terminal_reason"]) == (ANSWER, True, None)


@pytest.mark.anyio
async def test_missing_finish_reason_keeps_the_existing_behaviour() -> None:
    deps, provider, _ = dependencies(llm=FakeLLMProvider(answer=ANSWER))

    result = await run(deps)

    assert RECOVERY not in statuses(result)
    assert len(provider.generation_calls) == 1
    assert result["generation_result"].finish_reason is None
    assert (result["generated_answer"], result["grounding_passed"]) == (ANSWER, True)


@pytest.mark.anyio
async def test_length_truncation_is_recovered_once_from_scratch_with_a_larger_budget() -> None:
    llm = FakeLLMProvider(answer=(Truncated(PARTIAL), Stopped(RECOVERED_ANSWER)))
    deps, provider, retrieval = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [
        *LOCAL_SUFFICIENT,
        *RERANKED,
        S.GENERATION_STARTED,
        RECOVERY,
        S.GENERATION_COMPLETED,
        *GROUNDED,
        S.GRAPH_COMPLETED,
    ]
    assert [event.sequence for event in result["execution_events"]] == list(range(len(result["execution_events"])))
    first, second = provider.generation_calls
    # Same question, evidence, labels, and prompt: nothing of the partial answer is appended or sent back.
    assert first[0] == second[0]
    assert PARTIAL not in repr(second[0])
    assert max_tokens(provider) == [NORMAL_MAX_TOKENS, RECOVERY_MAX_TOKENS]
    # Only the complete answer was ever graded.
    assert len(provider.grounding_calls) == 1
    assert RECOVERED_ANSWER in repr(provider.grounding_calls[0][0])
    assert PARTIAL not in repr(provider.grounding_calls)
    assert retrieval.search.call_count == 1
    assert result["generated_answer"] == result["generation_result"].answer == RECOVERED_ANSWER
    assert result["generation_result"].finish_reason == "stop"
    assert (result["grounding_attempts"], result["grounding_passed"], result["terminal_reason"]) == (1, True, None)
    assert PARTIAL not in serialized_state(result)


@pytest.mark.anyio
async def test_length_truncation_twice_is_a_controlled_generation_failure() -> None:
    llm = FakeLLMProvider(answer=(Truncated(PARTIAL), Truncated(PARTIAL + " more")))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *RERANKED, S.GENERATION_STARTED, RECOVERY, S.GRAPH_FAILED]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 0)
    assert max_tokens(provider) == [NORMAL_MAX_TOKENS, RECOVERY_MAX_TOKENS]
    assert result["generated_answer"] is None and result["generation_result"] is None
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.GENERATION_INCOMPLETE
    assert "rivate partial" not in serialized_state(result)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "recovered",
    ["Private recovery cites nothing at all.", "Private recovery cites an unknown source [S9]."],
)
async def test_length_recovery_with_an_invalid_citation_contract_is_a_generation_failure(recovered) -> None:
    llm = FakeLLMProvider(answer=(Truncated(PARTIAL), Stopped(recovered)))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [*LOCAL_SUFFICIENT, *RERANKED, S.GENERATION_STARTED, RECOVERY, S.GRAPH_FAILED]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 0)
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.ANSWER_VALIDATION_FAILURE
    assert result["grounding_passed"] is None
    assert "rivate partial" not in serialized_state(result)
    assert "rivate recovery" not in serialized_state(result)


@pytest.mark.anyio
async def test_truncated_text_with_a_broken_citation_is_recovered_not_structurally_rejected() -> None:
    # A cut-off answer is never structurally validated: it is redone, not failed for its citations.
    llm = FakeLLMProvider(answer=(Truncated("Private partial text cut inside a label [S"), Stopped(RECOVERED_ANSWER)))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert (result["generated_answer"], result["grounding_passed"], result["error_category"]) == (
        RECOVERED_ANSWER,
        True,
        None,
    )
    assert len(provider.generation_calls) == 2


@pytest.mark.anyio
async def test_grounding_regeneration_is_unchanged_by_finish_reasons() -> None:
    llm = FakeLLMProvider(answer=(Stopped(ANSWER), Stopped(REGENERATED_ANSWER)), grounding=(LOW, HIGH))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [
        *LOCAL_SUFFICIENT,
        *GENERATED,
        *UNGROUNDED,
        S.GENERATION_STARTED,
        S.GENERATION_COMPLETED,
        *GROUNDED,
        S.GRAPH_COMPLETED,
    ]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (2, 2)
    assert max_tokens(provider) == [NORMAL_MAX_TOKENS, NORMAL_MAX_TOKENS]
    assert (result["generated_answer"], result["grounding_attempts"], result["grounding_passed"]) == (
        REGENERATED_ANSWER,
        2,
        True,
    )


@pytest.mark.anyio
async def test_truncated_grounding_regeneration_gets_its_own_single_length_recovery() -> None:
    llm = FakeLLMProvider(
        answer=(Stopped(ANSWER), Truncated(PARTIAL_REGENERATION), Stopped(RECOVERED_ANSWER)),
        grounding=(LOW, HIGH),
    )
    deps, provider, retrieval = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [
        *LOCAL_SUFFICIENT,
        *GENERATED,
        *UNGROUNDED,
        S.GENERATION_STARTED,
        RECOVERY,
        S.GENERATION_COMPLETED,
        *GROUNDED,
        S.GRAPH_COMPLETED,
    ]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (3, 2)
    assert max_tokens(provider) == [NORMAL_MAX_TOKENS, NORMAL_MAX_TOKENS, RECOVERY_MAX_TOKENS]
    # The recovery repeats the regeneration request itself; the partial regeneration is not reused.
    assert provider.generation_calls[1][0] == provider.generation_calls[2][0]
    assert PARTIAL_REGENERATION not in repr(provider.generation_calls[2][0])
    assert PARTIAL_REGENERATION not in repr(provider.grounding_calls)
    # Length recovery is not a grounding attempt.
    assert (result["grounding_attempts"], result["grounding_passed"]) == (2, True)
    assert result["generated_answer"] == RECOVERED_ANSWER
    assert retrieval.search.call_count == 1
    serialized = serialized_state(result)
    assert PARTIAL_REGENERATION not in serialized and ANSWER not in serialized


@pytest.mark.anyio
async def test_truncated_grounding_regeneration_truncated_again_is_a_generation_failure() -> None:
    llm = FakeLLMProvider(
        answer=(Stopped(ANSWER), Truncated(PARTIAL_REGENERATION), Truncated(PARTIAL_REGENERATION + " more")),
        grounding=(LOW, HIGH),
    )
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert statuses(result) == [
        *LOCAL_SUFFICIENT,
        *GENERATED,
        *UNGROUNDED,
        S.GENERATION_STARTED,
        RECOVERY,
        S.GRAPH_FAILED,
    ]
    assert (len(provider.generation_calls), len(provider.grounding_calls)) == (3, 1)
    assert result["grounding_attempts"] == 1
    assert result["generated_answer"] is None and result["generation_result"] is None
    assert result["terminal_reason"] == TerminalReason.GENERATION_FAILED
    assert result["error_category"] == AgentErrorCategory.GENERATION_INCOMPLETE
    serialized = serialized_state(result)
    assert "rivate partial" not in serialized and ANSWER not in serialized


@pytest.mark.anyio
async def test_structural_and_semantic_failures_stay_distinct_in_state() -> None:
    structural_deps, _, _ = dependencies(llm=FakeLLMProvider(answer=Stopped("Private answer without a citation.")))
    semantic_deps, _, _ = dependencies(
        llm=FakeLLMProvider(answer=(Stopped(ANSWER), Stopped(REGENERATED_ANSWER)), grounding=LOW)
    )

    structural, semantic = await run(structural_deps), await run(semantic_deps)

    assert (structural["terminal_reason"], structural["error_category"]) == (
        TerminalReason.GENERATION_FAILED,
        AgentErrorCategory.ANSWER_VALIDATION_FAILURE,
    )
    assert (semantic["terminal_reason"], semantic["error_category"]) == (TerminalReason.GROUNDING_FAILED, None)
    assert (structural["grounding_attempts"], semantic["grounding_attempts"]) == (0, 2)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("recovery", "answer", "category"),
    [
        (Stopped(RECOVERED_ANSWER), RECOVERED_ANSWER, None),
        (IncompleteGenerationError("private provider detail"), None, AgentErrorCategory.GENERATION_INCOMPLETE),
    ],
)
async def test_provider_reported_empty_truncation_gets_the_same_single_recovery(recovery, answer, category) -> None:
    # A reasoning model can hit the limit before writing any text; the adapter reports that as incomplete.
    llm = FakeLLMProvider(answer=(IncompleteGenerationError("private provider detail"), recovery))
    deps, provider, _ = dependencies(llm=llm)

    result = await run(deps)

    assert RECOVERY in statuses(result)
    assert max_tokens(provider) == [NORMAL_MAX_TOKENS, RECOVERY_MAX_TOKENS]
    assert (result["generated_answer"], result["error_category"]) == (answer, category)
    assert len(provider.grounding_calls) == (1 if answer else 0)
    assert "private provider detail" not in serialized_state(result)
