import html
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
from src.gradio_client import RAGStreamClient, RAGUIClientError, SSEEvent

DEFAULT_API_BASE_URL = "http://127.0.0.1:8001"
EXAMPLE_QUESTIONS = [
    "How can artificial intelligence improve healthcare access for underserved populations?",
    "What kinds of bias can affect large language models used for clinical triage?",
    "How can smartphone-based image analysis make medical diagnosis more accessible in remote areas?",
    "What are the potential healthcare applications of AI smart glasses?",
]


@dataclass
class DemoState:
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str | None = None
    source_count: int | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: str = "Ready"
    terminal: str | None = None

    def apply(self, event: SSEEvent) -> None:
        if event.event == "metadata":
            self.retrieval_mode = _optional_text(event.data.get("retrieval_mode"))
            self.source_count = _optional_nonnegative_int(event.data.get("source_count"))
            self.status = "Generating…"
        elif event.event == "delta":
            text = event.data.get("text")
            if not isinstance(text, str):
                raise RAGUIClientError("RAG API returned an invalid answer update.")
            self.answer += text
            self.status = "Generating… Answer is provisional until validation completes."
        elif event.event == "sources":
            sources = event.data.get("sources")
            if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
                raise RAGUIClientError("RAG API returned invalid source metadata.")
            self.sources = sources
            self.status = "Validating…"
        elif event.event == "done":
            self.model = _optional_text(event.data.get("model"))
            self.prompt_tokens = _optional_nonnegative_int(event.data.get("prompt_tokens"))
            self.completion_tokens = _optional_nonnegative_int(event.data.get("completion_tokens"))
            self.status = "Grounding validation passed"
            self.terminal = "done"
        elif event.event == "error":
            self.sources = []
            self.status = (
                "Validation or generation failed. Any partial answer is unvalidated and "
                "must not be treated as a grounded final answer."
            )
            self.terminal = "error"


async def stream_demo(question: str, *, client: RAGStreamClient | None = None) -> AsyncIterator[tuple[str, str, str, str]]:
    state = DemoState()
    if not isinstance(question, str) or not question.strip():
        state.status = "Enter a research question before submitting."
        yield _render(state)
        return

    active_client = client or RAGStreamClient(base_url=os.getenv("RAG_API_BASE_URL", DEFAULT_API_BASE_URL))
    state.status = "Connecting to RAG API…"
    yield _render(state)
    try:
        async for event in active_client.stream(question):
            state.apply(event)
            yield _render(state)
    except RAGUIClientError as exc:
        state.sources = []
        state.terminal = "error"
        state.status = str(exc)
        if state.answer:
            state.status += " The partial answer is unvalidated."
        yield _render(state)


def clear_demo() -> tuple[str, str, str, str, str]:
    return "", "", "", "**Status:** Ready", ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Production Agentic RAG") as demo:
        gr.Markdown("# Production Agentic RAG\nGrounded research-paper question answering over the locally indexed corpus.")
        question = gr.Textbox(
            label="Research question",
            lines=3,
            placeholder="Ask a question grounded in the indexed research papers…",
        )
        with gr.Row():
            ask = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")
        status = gr.Markdown("**Status:** Ready")
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Authoritative sources")
        execution = gr.Markdown(label="Execution")
        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question)

        outputs = [answer, sources, status, execution]
        ask.click(stream_demo, inputs=question, outputs=outputs)
        question.submit(stream_demo, inputs=question, outputs=outputs)
        clear.click(
            clear_demo,
            inputs=None,
            outputs=[question, answer, sources, status, execution],
            queue=False,
        )
    return demo


def _render(state: DemoState) -> tuple[str, str, str, str]:
    answer = state.answer or "_Answer text will appear here as it is generated._"
    return answer, _render_sources(state.sources), f"**Status:** {state.status}", _render_execution(state)


def _render_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "_No validated sources available._"
    rendered: list[str] = []
    for source in sources:
        citation = html.escape(str(source.get("citation") or "[S?]"))
        title = html.escape(str(source.get("title") or "Untitled source"))
        arxiv_id = html.escape(str(source.get("arxiv_id") or "Unknown"))
        section_value = source.get("section")
        section = f" · Section: {html.escape(str(section_value))}" if section_value else ""
        content = str(source.get("content") or "")
        preview = content if len(content) <= 1200 else f"{content[:1200].rstrip()}…"
        escaped_preview = html.escape(preview).replace("\n", " ")
        rendered.append(f"### {citation} {title}\n**arXiv:** {arxiv_id}{section}\n\n> {escaped_preview}")
    return "\n\n".join(rendered)


def _render_execution(state: DemoState) -> str:
    fields = []
    if state.retrieval_mode:
        fields.append(f"Retrieval: `{html.escape(state.retrieval_mode)}`")
    if state.source_count is not None:
        fields.append(f"Sources retrieved: `{state.source_count}`")
    if state.model:
        fields.append(f"Model: `{html.escape(state.model)}`")
    if state.prompt_tokens is not None:
        fields.append(f"Prompt tokens: `{state.prompt_tokens}`")
    if state.completion_tokens is not None:
        fields.append(f"Completion tokens: `{state.completion_tokens}`")
    return "  \n".join(fields) if fields else "_Execution metadata will appear here._"


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
