from collections.abc import AsyncIterator

import pytest
from src.gradio_app import DemoState, clear_demo, stream_demo
from src.gradio_client import RAGUIClientError, SSEEvent


class FakeClient:
    def __init__(self, events=(), error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.questions = []

    async def stream(self, question: str) -> AsyncIterator[SSEEvent]:
        self.questions.append(question)
        for event in self.events:
            yield event
        if self.error:
            raise self.error


@pytest.mark.anyio
async def test_empty_question_does_not_call_backend() -> None:
    client = FakeClient()

    updates = [update async for update in stream_demo("  ", client=client)]

    assert client.questions == []
    assert "Enter a research question" in updates[-1][2]


@pytest.mark.anyio
async def test_ui_updates_answer_then_sources_and_marks_done_only_at_terminal() -> None:
    source = {
        "citation": "[S1]",
        "title": "Grounded Paper",
        "arxiv_id": "2401.00001",
        "section": "Results",
        "content": "Authoritative evidence.",
    }
    client = FakeClient(
        (
            SSEEvent("metadata", {"retrieval_mode": "hybrid", "source_count": 1}),
            SSEEvent("delta", {"text": "Grounded "}),
            SSEEvent("delta", {"text": "answer [S1]."}),
            SSEEvent("sources", {"sources": [source]}),
            SSEEvent(
                "done",
                {"model": "test-model", "prompt_tokens": 10, "completion_tokens": 4},
            ),
        )
    )

    updates = [update async for update in stream_demo("question", client=client)]

    assert "Connecting" in updates[0][2]
    assert updates[2][0] == "Grounded "
    assert "No validated sources" in updates[2][1]
    assert updates[3][0] == "Grounded answer [S1]."
    assert "Grounded Paper" in updates[-2][1]
    assert "Validating" in updates[-2][2]
    assert "validation passed" in updates[-1][2]
    assert "Model: `test-model`" in updates[-1][3]
    assert client.questions == ["question"]


@pytest.mark.anyio
async def test_error_keeps_partial_answer_but_marks_it_unvalidated_and_hides_sources() -> None:
    client = FakeClient(
        (
            SSEEvent("metadata", {"retrieval_mode": "hybrid", "source_count": 1}),
            SSEEvent("delta", {"text": "Partial answer"}),
            SSEEvent("error", {"code": "grounding_validation_failed", "message": "safe"}),
        )
    )

    updates = [update async for update in stream_demo("question", client=client)]

    answer, sources, status, _ = updates[-1]
    assert answer == "Partial answer"
    assert "No validated sources" in sources
    assert "unvalidated" in status
    assert "passed" not in status


@pytest.mark.anyio
async def test_client_failure_keeps_ui_alive_and_does_not_expose_raw_details() -> None:
    client = FakeClient(error=RAGUIClientError("RAG API is unavailable."))

    updates = [update async for update in stream_demo("question", client=client)]

    assert "RAG API is unavailable" in updates[-1][2]
    assert "traceback" not in updates[-1][2]


def test_state_rejects_invalid_delta_and_sources_payloads() -> None:
    state = DemoState()
    with pytest.raises(RAGUIClientError, match="answer update"):
        state.apply(SSEEvent("delta", {"text": 123}))
    with pytest.raises(RAGUIClientError, match="source metadata"):
        state.apply(SSEEvent("sources", {"sources": "not-a-list"}))


def test_clear_resets_all_visible_fields() -> None:
    assert clear_demo() == ("", "", "", "**Status:** Ready", "")


@pytest.mark.anyio
async def test_request_details_show_cache_pipeline_and_deterministic_response_time() -> None:
    client = FakeClient(
        (
            SSEEvent(
                "metadata",
                {
                    "retrieval_mode": "hybrid",
                    "source_count": 2,
                    "cache_status": "hit",
                    "retrieval_size": 5,
                    "embedding_model": "safe-embedding-model",
                    "embedding_dimension": 384,
                    "rrf_k": 60,
                    "llm_temperature": 0.1,
                    "max_completion_tokens": 1024,
                    "observability_provider": "langfuse",
                    "observability_status": "configured",
                },
            ),
            SSEEvent("delta", {"text": "Cached answer [S1]."}),
            SSEEvent("sources", {"sources": []}),
            SSEEvent(
                "done",
                {"model": "test-model", "prompt_tokens": 10, "completion_tokens": 4},
            ),
        )
    )
    ticks = iter((10.0, 10.18))

    updates = [update async for update in stream_demo("question", client=client, clock=lambda: next(ticks))]

    details = updates[-1][3]
    assert "Cache: **HIT**" in details
    assert "Response time: `0.18 s`" in details
    assert "Original prompt tokens: `10`" in details
    assert "Original completion tokens: `4`" in details
    assert "Retrieval size: `5`" in details
    assert "Embedding model: `safe-embedding-model`" in details
    assert "Embedding dimension: `384`" in details
    assert "RRF k: `60`" in details
    assert "Temperature: `0.1`" in details
    assert "Maximum completion tokens: `1024`" in details
    assert "Observability: `langfuse (configured)`" in details


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("cache_status", "label"),
    [("miss", "MISS"), ("bypass", "BYPASS"), ("failure", "UNAVAILABLE")],
)
async def test_cache_outcomes_are_presented_without_changing_request_success(cache_status, label) -> None:
    client = FakeClient(
        (
            SSEEvent("metadata", {"cache_status": cache_status}),
            SSEEvent("delta", {"text": "Answer"}),
            SSEEvent("sources", {"sources": []}),
            SSEEvent("done", {}),
        )
    )
    ticks = iter((1.0, 1.5))

    updates = [update async for update in stream_demo("question", client=client, clock=lambda: next(ticks))]

    assert f"Cache: **{label}**" in updates[-1][3]
    assert "validation passed" in updates[-1][2]


def test_metadata_remains_backward_compatible_and_ignores_unknown_fields() -> None:
    state = DemoState()

    state.apply(SSEEvent("metadata", {"retrieval_mode": "hybrid", "unknown_future_field": "ignored"}))

    assert state.retrieval_mode == "hybrid"
    assert state.cache_status is None


LOCAL_SOURCE = {
    "citation": "[S1]",
    "source_type": "local",
    "title": "Local Paper",
    "arxiv_id": "2401.00001",
    "section": "Results",
    "content": "Local evidence.",
    "source_url": None,
}
LIVE_SOURCE = {
    "citation": "[S2]",
    "source_type": "live_arxiv",
    "title": "Live <b>Paper</b>",
    "arxiv_id": "2501.00002",
    "section": None,
    "content": "Live evidence.",
    "source_url": "https://arxiv.org/pdf/2501.00002v1",
}


def agent_events(*, cache_status: str = "miss", statuses=("Guardrail passed", "Local retrieval completed")):
    return (
        SSEEvent("metadata", {"cache_status": cache_status, "pipeline_version": "v1"}),
        *(SSEEvent("status", {"code": f"step_{index}", "message": message}) for index, message in enumerate(statuses)),
        SSEEvent("answer", {"text": "Validated answer [S1][S2]."}),
        SSEEvent("sources", {"sources": [LOCAL_SOURCE, LIVE_SOURCE]}),
        SSEEvent(
            "done",
            {
                "outcome": "answered",
                "grounding_passed": True,
                "model": "test-model",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
        ),
    )


@pytest.mark.anyio
async def test_agent_statuses_accumulate_in_order_and_answer_appears_only_after_the_answer_event() -> None:
    statuses = ("Guardrail passed", "Evidence insufficient", "Query rewritten", "Grounding validation passed")
    client = FakeClient(agent_events(statuses=statuses))
    ticks = iter([10.0, 13.8])

    updates = [update async for update in stream_demo("question", client=client, clock=lambda: next(ticks))]

    answer_update = next(index for index, update in enumerate(updates) if "Validated answer" in update[0])
    for update in updates[:answer_update]:
        assert "Validated answer" not in update[0]
        assert "No validated sources" in update[1]
    assert "The answer will appear here once it has been validated" in updates[answer_update - 1][0]
    assert "Grounding validation passed" in updates[answer_update - 1][3]
    final = updates[-1]
    execution = final[3]
    positions = [execution.index(f"✓ {status}") for status in statuses]
    assert positions == sorted(positions)
    assert "**Execution path**" in execution
    assert "Cache: **MISS**" in execution
    assert "Response time: `3.80 s`" in execution
    assert "Model: `test-model`" in execution
    assert "**Status:** Grounding validation passed" in final[2]
    assert "Guardrail passed…" in updates[2][2]


@pytest.mark.anyio
async def test_agent_sources_distinguish_local_from_live_arxiv_and_escape_content() -> None:
    updates = [update async for update in stream_demo("question", client=FakeClient(agent_events()))]

    sources = updates[-1][1]
    assert "### [S1] Local Paper" in sources
    assert "`LOCAL` · **arXiv:** 2401.00001 · Section: Results" in sources
    assert "`LIVE ARXIV` · **arXiv:** 2501.00002 · [PDF](https://arxiv.org/pdf/2501.00002v1)" in sources
    assert "Live &lt;b&gt;Paper&lt;/b&gt;" in sources and "<b>" not in sources
    assert sources.count("[PDF](") == 1


@pytest.mark.anyio
async def test_agent_cache_hit_shows_hit_and_a_single_truthful_step() -> None:
    client = FakeClient(agent_events(cache_status="hit", statuses=("Validated cached answer found",)))
    ticks = iter([1.0, 1.045])

    updates = [update async for update in stream_demo("question", client=client, clock=lambda: next(ticks))]

    execution = updates[-1][3]
    assert "Cache: **HIT**" in execution
    assert "Response time: `0.04 s`" in execution or "Response time: `0.05 s`" in execution
    assert execution.count("✓") == 1 and "✓ Validated cached answer found" in execution
    assert "Guardrail" not in execution
    assert "Original prompt tokens: `100`" in execution


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("outcome", "label"),
    [
        ("out_of_scope", "Out of scope"),
        ("insufficient_evidence", "Insufficient evidence"),
        ("grounding_failed", "Grounding validation failed"),
        ("generation_failed", "No validated answer could be produced"),
    ],
)
async def test_agent_semantic_outcomes_show_the_safe_message_without_sources(outcome: str, label: str) -> None:
    client = FakeClient(
        (
            SSEEvent("metadata", {"cache_status": "miss"}),
            SSEEvent("status", {"code": "guardrail_started", "message": "Checking question scope"}),
            SSEEvent("outcome", {"outcome": outcome, "message": "A fixed safe message."}),
            SSEEvent("done", {"outcome": outcome, "grounding_passed": False if outcome == "grounding_failed" else None}),
        )
    )

    updates = [update async for update in stream_demo("question", client=client)]

    final = updates[-1]
    assert final[0] == "A fixed safe message."
    assert "No validated sources" in final[1]
    assert f"**Status:** {label}" in final[2]
    assert "Grounding validation passed" not in final[2]
    assert "✓ Checking question scope" in final[3]


@pytest.mark.anyio
async def test_agent_error_shows_only_the_safe_message_and_no_answer() -> None:
    client = FakeClient(
        (
            SSEEvent("metadata", {"cache_status": "miss"}),
            SSEEvent("status", {"code": "graph_started", "message": "Request accepted"}),
            SSEEvent("error", {"code": "timeout", "message": "The research assistant took too long to answer this question."}),
        )
    )

    updates = [update async for update in stream_demo("question", client=client)]

    final = updates[-1]
    assert "The answer will appear here once it has been validated" in final[0]
    assert "No validated sources" in final[1]
    assert "took too long" in final[2]
    assert "Traceback" not in "".join(final) and "{" not in final[2]


@pytest.mark.parametrize(
    "event",
    [
        SSEEvent("status", {"code": "x"}),
        SSEEvent("status", {"code": "x", "message": 5}),
        SSEEvent("answer", {"text": ""}),
        SSEEvent("answer", {"text": None}),
        SSEEvent("outcome", {"outcome": "answered", "message": "m"}),
        SSEEvent("outcome", {"outcome": "out_of_scope"}),
    ],
)
def test_state_rejects_invalid_agent_events(event: SSEEvent) -> None:
    with pytest.raises(RAGUIClientError):
        DemoState().apply(event)


def test_execution_steps_are_bounded_and_untrusted_links_are_not_rendered() -> None:
    state = DemoState()
    for index in range(200):
        state.apply(SSEEvent("status", {"code": "x", "message": f"Step {index} <script>"}))
    state.apply(SSEEvent("answer", {"text": "Answer [S1]."}))
    state.apply(
        SSEEvent(
            "sources",
            {"sources": [{**LIVE_SOURCE, "source_url": "https://evil.example/x.pdf"}, {**LIVE_SOURCE, "source_url": 5}]},
        )
    )

    from src.gradio_app import MAX_EXECUTION_STEPS, _render

    answer, sources, _, execution = _render(state)

    assert len(state.steps) == MAX_EXECUTION_STEPS == 60
    assert "<script>" not in execution and "&lt;script&gt;" in execution
    assert "[PDF](" not in sources and "evil.example" not in sources
    assert answer == "Answer [S1]."


def test_default_client_targets_the_agent_endpoint() -> None:
    from src.gradio_app import DEFAULT_API_BASE_URL
    from src.gradio_client import AgentStreamClient

    assert DEFAULT_API_BASE_URL == "http://127.0.0.1:8000"
    assert AgentStreamClient(base_url=DEFAULT_API_BASE_URL).endpoint == "http://127.0.0.1:8000/api/v1/agent/ask/stream"


@pytest.mark.anyio
async def test_agent_length_recovery_shows_the_safe_step_and_only_the_final_validated_answer() -> None:
    recovery = "Initial answer was incomplete; regenerating with a larger output limit"
    client = FakeClient(agent_events(statuses=("Generating the answer", recovery, "Grounding validation passed")))

    updates = [update async for update in stream_demo("question", client=client)]

    answer_update = next(index for index, update in enumerate(updates) if "Validated answer" in update[0])
    recovery_update = next(index for index, update in enumerate(updates) if recovery in update[3])
    assert recovery_update < answer_update
    # While the answer is being redone, the answer area holds only the placeholder.
    assert "The answer will appear here once it has been validated" in updates[recovery_update][0]
    assert f"✓ {recovery}" in updates[-1][3]
    assert sum("Validated answer" in update[0] for update in updates[:answer_update]) == 0
