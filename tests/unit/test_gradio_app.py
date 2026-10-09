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
