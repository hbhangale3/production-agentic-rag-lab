from collections.abc import AsyncIterator

import pytest
from src import gradio_app
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
    assert "<b>Model</b>test-model" in updates[-1][4]
    assert "<b>Sources</b>1" in updates[-1][4]
    assert "<b>Tokens</b>10 in / 4 out" in updates[-1][4]
    assert "GROUNDED" in updates[-1][4]
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

    answer, sources, status, _, metadata = updates[-1]
    assert answer == "Partial answer"
    assert "No validated sources" in sources
    assert "unvalidated" in status
    assert "passed" not in status
    assert "NOT VALIDATED" in metadata and "GROUNDED" not in metadata


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


def test_clear_resets_the_question_and_every_response_component() -> None:
    question, answer, sources, status, execution, metadata = clear_demo()

    assert question == ""
    assert answer == gradio_app.ANSWER_PLACEHOLDER
    assert "No validated sources" in sources and "rag-source" not in sources.replace("rag-sources", "")
    assert "Status:</strong> Ready" in status
    assert "<li>" not in execution
    assert metadata == ""


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

    details = updates[-1][4]
    assert ">CACHE HIT<" in details
    assert "<b>Latency</b>180 ms" in details
    assert "<b>Tokens (original run)</b>10 in / 4 out" in details
    assert "<b>Sources retrieved</b>2" in details
    assert "Retrieval size: 5" in details
    assert "Embedding model: safe-embedding-model" in details
    assert "Embedding dimension: 384" in details
    assert "RRF k: 60" in details
    assert "Temperature: 0.1" in details
    assert "Maximum completion tokens: 1024" in details
    assert "Observability: langfuse (configured)" in details


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("cache_status", "label"),
    [("miss", "CACHE MISS"), ("bypass", "CACHE BYPASS"), ("failure", "CACHE UNAVAILABLE")],
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

    assert f">{label}<" in updates[-1][4]
    assert "<b>Latency</b>500 ms" in updates[-1][4]
    assert "Tokens" not in updates[-1][4] and "Model" not in updates[-1][4]
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
    assert execution.startswith('<ol class="rag-steps">') and execution.count("<li>") == len(statuses)
    metadata = final[4]
    assert ">CACHE MISS<" in metadata and ">GROUNDED<" in metadata
    assert "<b>Latency</b>3.80 s" in metadata
    assert "<b>Model</b>test-model" in metadata
    assert "<b>Sources</b>2" in metadata
    assert "<b>Tokens</b>100 in / 20 out" in metadata
    assert updates[answer_update - 1][4].count("rag-badge") == 1  # cache only; no grounding verdict before the end
    assert "Status:</strong> Grounding validation passed" in final[2]
    assert "Guardrail passed…" in updates[2][2]


@pytest.mark.anyio
async def test_agent_sources_distinguish_local_from_live_arxiv_and_escape_content() -> None:
    updates = [update async for update in stream_demo("question", client=FakeClient(agent_events()))]

    sources = updates[-1][1]
    local, live = sources.split("</article>")[:2]
    assert '<span class="rag-cite">[S1]</span><span class="rag-badge rag-neutral">LOCAL</span>' in local
    assert "Local Paper" in local and "arXiv: 2401.00001 · Section: Results" in local
    assert "<a " not in local
    assert '<span class="rag-cite">[S2]</span><span class="rag-badge rag-info">LIVE ARXIV</span>' in live
    assert "arXiv: 2501.00002" in live and "Section" not in live
    assert 'href="https://arxiv.org/pdf/2501.00002v1"' in live and ">Open arXiv</a>" in live
    assert "Live &lt;b&gt;Paper&lt;/b&gt;" in sources and "<b>" not in sources
    assert sources.count("<a ") == 1
    # The excerpt is collapsed, not part of the card face.
    assert "<details><summary>View source details</summary><p>Local evidence.</p></details>" in local


def test_source_cards_show_the_publication_date_and_truncate_long_excerpts() -> None:
    from src.gradio_app import SOURCE_EXCERPT_CHARS, _render_sources

    rendered = _render_sources(
        [
            {**LOCAL_SOURCE, "published_date": "2026-10-02T00:00:00Z", "content": "x" * 5000},
            {**LOCAL_SOURCE, "published_date": "not-a-date", "content": ""},
        ]
    )

    first, second = rendered.split("</article>")[:2]
    assert "Published: 2026-10-02" in first and "T00:00" not in first
    assert first.count("x") == SOURCE_EXCERPT_CHARS and "…</p>" in first
    assert "Published" not in second and "<details>" not in second


@pytest.mark.anyio
async def test_agent_cache_hit_shows_hit_and_a_single_truthful_step() -> None:
    client = FakeClient(agent_events(cache_status="hit", statuses=("Validated cached answer found",)))
    ticks = iter([1.0, 1.045])

    updates = [update async for update in stream_demo("question", client=client, clock=lambda: next(ticks))]

    execution, metadata = updates[-1][3], updates[-1][4]
    assert ">CACHE HIT<" in metadata and ">GROUNDED<" in metadata
    assert "<b>Latency</b>45 ms" in metadata or "<b>Latency</b>44 ms" in metadata
    assert execution.count("✓") == 1 and "✓ Validated cached answer found" in execution
    assert "Guardrail" not in execution
    assert "<b>Tokens (original run)</b>100 in / 20 out" in metadata


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("outcome", "label", "badge"),
    [
        ("out_of_scope", "Out of scope", "OUT OF SCOPE"),
        ("insufficient_evidence", "Insufficient evidence", "INSUFFICIENT EVIDENCE"),
        ("grounding_failed", "Grounding validation failed", "NO VALIDATED ANSWER"),
        ("generation_failed", "No validated answer could be produced", "NO VALIDATED ANSWER"),
    ],
)
async def test_agent_semantic_outcomes_show_the_safe_message_without_sources(
    outcome: str, label: str, badge: str
) -> None:
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
    assert f"Status:</strong> {label}" in final[2]
    assert "Grounding validation passed" not in final[2]
    assert "✓ Checking question scope" in final[3]
    assert f">{badge}<" in final[4] and "GROUNDED" not in final[4]
    assert "Tokens" not in final[4] and "Model" not in final[4]


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
    assert ">NO VALIDATED ANSWER<" in final[4] and "GROUNDED" not in final[4]


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

    answer, sources, _, execution, _ = _render(state)

    assert len(state.steps) == MAX_EXECUTION_STEPS == 60
    assert "<script>" not in execution and "&lt;script&gt;" in execution
    assert "<a " not in sources and "evil.example" not in sources
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


def test_server_binding_defaults_to_loopback_and_reads_the_container_environment(monkeypatch) -> None:
    monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    assert gradio_app.server_binding() == ("127.0.0.1", 7860)

    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7861")
    assert gradio_app.server_binding() == ("0.0.0.0", 7861)

    monkeypatch.setenv("GRADIO_SERVER_NAME", "  ")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "")
    assert gradio_app.server_binding() == ("127.0.0.1", 7860)


@pytest.mark.parametrize("port", ["not-a-port", "0", "70000"])
def test_server_binding_rejects_an_invalid_port(monkeypatch, port: str) -> None:
    monkeypatch.setenv("GRADIO_SERVER_PORT", port)

    with pytest.raises(ValueError, match="GRADIO_SERVER_PORT"):
        gradio_app.server_binding()


def test_main_launches_on_the_configured_binding_without_a_public_share_link(monkeypatch) -> None:
    launched = {}

    class FakeDemo:
        def launch(self, **kwargs) -> None:
            launched.update(kwargs)

    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    monkeypatch.setattr(gradio_app, "build_demo", lambda: FakeDemo())

    gradio_app.main()

    assert launched == {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "theme": gradio_app.THEME,
        "css": gradio_app.CSS,
    }


def test_architecture_image_is_a_packaged_png_inside_the_source_tree() -> None:
    image = gradio_app.ARCHITECTURE_IMAGE

    assert image.is_file() and image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The Docker image copies only src/, so the asset has to live there to reach the container.
    assert image.is_relative_to(gradio_app.Path(gradio_app.__file__).parent)


def test_architecture_diagram_source_has_current_labels_and_no_stale_ones() -> None:
    diagram = gradio_app.ARCHITECTURE_IMAGE.with_suffix(".svg").read_text()

    for label in ("BAAI/bge-small-en-v1.5", "rag:agent-response:v1:", "/api/v1/agent/ask/stream", "LangGraph"):
        assert label in diagram
    for stale in ("Planned", "bge-large", "CrewAI", "rag:response:v1"):
        assert stale not in diagram


def test_demo_builds_with_a_collapsed_architecture_section_and_the_required_examples() -> None:
    config = gradio_app.build_demo().get_config_file()

    components = {component["type"]: component for component in config["components"]}
    assert components["accordion"]["props"]["label"] == "How this system works"
    assert components["accordion"]["props"]["open"] is False
    assert "production-agentic-rag-final" in str(components["image"]["props"]["value"])
    assert len(gradio_app.EXAMPLE_QUESTIONS) == 4
    assert gradio_app.EXAMPLE_QUESTIONS[0].startswith("What biases were found when auditing open-source")
    assert "CrewAI" not in gradio_app.HERO_HTML + gradio_app.HOW_IT_WORKS


def test_latency_is_shown_in_milliseconds_below_one_second() -> None:
    assert gradio_app._format_latency(0.0071) == "7 ms"
    assert gradio_app._format_latency(12.3456) == "12.35 s"


def test_metadata_is_empty_until_the_api_returns_something_and_never_shows_placeholder_zeros() -> None:
    state = DemoState()
    assert gradio_app._render_metadata(state) == ""

    state.apply(SSEEvent("metadata", {"cache_status": "miss"}))

    metadata = gradio_app._render_metadata(state)
    assert ">CACHE MISS<" in metadata
    for absent in ("Latency", "Model", "Sources", "Tokens", "GROUNDED", ">0<"):
        assert absent not in metadata
