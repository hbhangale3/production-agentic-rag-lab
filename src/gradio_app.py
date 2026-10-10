import html
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import gradio as gr
from src.gradio_client import AgentStreamClient, RAGStreamClient, RAGUIClientError, SSEEvent

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7860
MAX_EXECUTION_STEPS = 60
SOURCE_EXCERPT_CHARS = 1200
SOURCE_TYPE_LABELS = {"local": "LOCAL", "live_arxiv": "LIVE ARXIV"}
OUTCOME_STATUS = {
    "out_of_scope": "Out of scope",
    "insufficient_evidence": "Insufficient evidence",
    "grounding_failed": "Grounding validation failed",
    "generation_failed": "No validated answer could be produced",
}
OUTCOME_BADGES = {
    "out_of_scope": "OUT OF SCOPE",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE",
    "grounding_failed": "NO VALIDATED ANSWER",
    "generation_failed": "NO VALIDATED ANSWER",
}
CACHE_BADGES = {"hit": "CACHE HIT", "miss": "CACHE MISS", "bypass": "CACHE BYPASS", "failure": "CACHE UNAVAILABLE"}
EXAMPLE_QUESTIONS = [
    "What biases were found when auditing open-source large language models for clinical triage?",
    "What are the current healthcare challenges in India and how can AI help solve them?",
    "How can smartphone-based image analysis improve medical diagnosis?",
    "What are recent applications of AI in healthcare access?",
]
TECHNOLOGIES = (
    "LangGraph",
    "FastAPI",
    "Gradio",
    "OpenSearch",
    "PostgreSQL",
    "Redis",
    "Apache Airflow",
    "Groq",
    "Langfuse",
    "Docling",
    "BGE",
)
ARCHITECTURE_IMAGE = Path(__file__).parent / "assets" / "architecture" / "production-agentic-rag-final.png"
ANSWER_PLACEHOLDER = "_The answer will appear here once it has been validated._"
HERO_HTML = (
    '<header class="rag-hero"><h1>Production Agentic RAG</h1>'
    "<p>Grounded research intelligence with local-first retrieval, live arXiv fallback, "
    "semantic grounding, caching, and observability.</p>"
    '<div class="rag-tech" role="list" aria-label="Technologies">'
    + "".join(f'<span role="listitem">{name}</span>' for name in TECHNOLOGIES)
    + "</div></header>"
)
HOW_IT_WORKS = (
    "This system searches the local research corpus first. If the evidence is not sufficient, it rewrites "
    "the query and can fall back to live arXiv research. Selected evidence is reranked on a common "
    "embedding scale, used to generate a cited answer, and then validated structurally and semantically "
    "before being returned or cached.\n\n"
    "1. **Search locally** with hybrid BM25 and vector retrieval.\n"
    "2. **Expand to live research** only when local evidence is insufficient.\n"
    "3. **Generate** a cited answer from the selected evidence.\n"
    "4. **Validate** citations and grounding before returning."
)
# Colours come from the active Gradio theme, so light and dark both stay readable.
CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.rag-hero h1 { margin: 0 0 4px; font-size: 1.75rem; font-weight: 650; letter-spacing: -0.01em; }
.rag-hero p { margin: 0 0 10px; max-width: 760px; color: var(--body-text-color); opacity: 0.8; }
.rag-tech, .rag-chips { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.rag-tech span { padding: 1px 8px; font-size: 0.72rem; color: var(--body-text-color); opacity: 0.75;
  border: 1px solid var(--border-color-primary); border-radius: 999px; }
.rag-heading { margin: 0 0 6px; font-size: 0.78rem; font-weight: 650; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--body-text-color); opacity: 0.7; }
.rag-status { padding: 8px 12px; font-size: 0.9rem; border: 1px solid var(--border-color-primary);
  border-left: 3px solid var(--color-accent); border-radius: 8px; background: var(--background-fill-secondary); }
.rag-card { padding: 16px 18px !important; gap: 10px !important; border: 1px solid var(--border-color-primary);
  border-radius: 10px; background: var(--background-fill-primary); }
.rag-card .block, .rag-card .html-container { padding: 0 !important; border: 0 !important; background: transparent !important;
  box-shadow: none !important; }
#rag-answer { font-size: 1.02rem; line-height: 1.65; }
.rag-chip { display: inline-flex; align-items: baseline; gap: 6px; padding: 3px 10px; font-size: 0.8rem;
  border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-secondary); }
.rag-chip b { font-weight: 600; font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
  opacity: 0.7; }
.rag-badge { display: inline-block; padding: 2px 8px; font-size: 0.7rem; font-weight: 650; letter-spacing: 0.05em;
  border: 1px solid currentColor; border-radius: 6px; white-space: nowrap; }
.rag-good { color: #15803d; } .rag-info { color: #1d4ed8; } .rag-warn { color: #b45309; }
.rag-neutral { color: var(--body-text-color); opacity: 0.8; }
.dark .rag-good { color: #4ade80; } .dark .rag-info { color: #93c5fd; } .dark .rag-warn { color: #fbbf24; }
.rag-details { margin-top: 8px; font-size: 0.78rem; color: var(--body-text-color); opacity: 0.75; }
.rag-empty { margin: 0; font-size: 0.9rem; color: var(--body-text-color); opacity: 0.7; }
.rag-sources { display: grid; gap: 10px; }
.rag-source { min-width: 0; padding: 12px 14px; border: 1px solid var(--border-color-primary); border-radius: 10px;
  background: var(--background-fill-primary); overflow-wrap: anywhere; }
.rag-source-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 4px; }
.rag-cite { font-weight: 650; font-variant-numeric: tabular-nums; }
.rag-source-title { font-weight: 600; line-height: 1.35; }
.rag-source-meta { margin-top: 2px; font-size: 0.82rem; color: var(--body-text-color); opacity: 0.75; }
.rag-source a { font-size: 0.85rem; }
.rag-source details { margin-top: 6px; font-size: 0.85rem; }
.rag-source summary { cursor: pointer; color: var(--body-text-color); opacity: 0.75; }
.rag-source details p { margin: 6px 0 0; line-height: 1.5; }
.rag-steps { margin: 0; padding: 0; list-style: none; }
.rag-steps li { position: relative; padding: 0 0 10px 14px; margin-left: 6px; font-size: 0.9rem; line-height: 1.4;
  border-left: 2px solid var(--border-color-primary); }
.rag-steps li:last-child { padding-bottom: 0; border-left-color: transparent; }
#rag-architecture img { width: 100%; height: auto; border-radius: 8px; }
"""
THEME = gr.themes.Default(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="md",
    font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class DemoState:
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str | None = None
    source_count: int | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_status: str | None = None
    response_time_seconds: float | None = None
    retrieval_size: int | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    rrf_k: int | None = None
    llm_temperature: float | None = None
    max_completion_tokens: int | None = None
    observability_provider: str | None = None
    observability_status: str | None = None
    status: str = "Ready"
    terminal: str | None = None
    steps: list[str] = field(default_factory=list)
    outcome: str | None = None
    grounding_passed: bool | None = None

    def apply(self, event: SSEEvent) -> None:
        if event.event == "metadata":
            self.retrieval_mode = _optional_text(event.data.get("retrieval_mode"))
            self.source_count = _optional_nonnegative_int(event.data.get("source_count"))
            self.cache_status = _cache_status(event.data.get("cache_status"))
            self.retrieval_size = _optional_nonnegative_int(event.data.get("retrieval_size"))
            self.embedding_model = _optional_text(event.data.get("embedding_model"))
            self.embedding_dimension = _optional_nonnegative_int(event.data.get("embedding_dimension"))
            self.rrf_k = _optional_nonnegative_int(event.data.get("rrf_k"))
            self.llm_temperature = _optional_nonnegative_float(event.data.get("llm_temperature"))
            self.max_completion_tokens = _optional_nonnegative_int(event.data.get("max_completion_tokens"))
            self.observability_provider = _optional_text(event.data.get("observability_provider"))
            self.observability_status = _optional_text(event.data.get("observability_status"))
            self.status = "Generating…"
        elif event.event == "status":
            message = event.data.get("message")
            if not isinstance(message, str) or not message:
                raise RAGUIClientError("RAG API returned an invalid status update.")
            if len(self.steps) < MAX_EXECUTION_STEPS:
                self.steps.append(message)
            self.status = f"{message}…"
        elif event.event == "answer":
            # The agent sends the answer once, only after it has passed grounding validation.
            text = event.data.get("text")
            if not isinstance(text, str) or not text:
                raise RAGUIClientError("RAG API returned an invalid answer.")
            self.answer = text
            self.outcome = "answered"
            self.status = "Answer validated"
        elif event.event == "outcome":
            outcome, message = event.data.get("outcome"), event.data.get("message")
            if outcome not in OUTCOME_STATUS or not isinstance(message, str) or not message:
                raise RAGUIClientError("RAG API returned an invalid outcome.")
            self.answer = message
            self.sources = []
            self.outcome = outcome
            self.status = OUTCOME_STATUS[outcome]
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
            grounding = event.data.get("grounding_passed")
            self.grounding_passed = grounding if isinstance(grounding, bool) else None
            self.status = OUTCOME_STATUS.get(self.outcome, "Grounding validation passed")
            self.terminal = "done"
        elif event.event == "error":
            self.sources = []
            message = event.data.get("message")
            if self.steps and isinstance(message, str) and message:
                # Agent errors carry a fixed safe message and never a partial answer.
                self.answer = ""
                self.status = message
            else:
                self.status = (
                    "Validation or generation failed. Any partial answer is unvalidated and "
                    "must not be treated as a grounded final answer."
                )
            self.terminal = "error"


async def stream_demo(
    question: str,
    *,
    client: RAGStreamClient | AgentStreamClient | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[tuple[str, str, str, str, str]]:
    state = DemoState()
    if not isinstance(question, str) or not question.strip():
        state.status = "Enter a research question before submitting."
        yield _render(state)
        return

    active_client = client or AgentStreamClient(base_url=os.getenv("RAG_API_BASE_URL", DEFAULT_API_BASE_URL))
    started_at = clock()
    state.status = "Connecting to RAG API…"
    yield _render(state)
    try:
        async for event in active_client.stream(question):
            state.apply(event)
            if state.terminal is not None:
                state.response_time_seconds = max(0.0, clock() - started_at)
            yield _render(state)
    except RAGUIClientError as exc:
        state.sources = []
        state.terminal = "error"
        state.status = str(exc)
        if state.answer:
            state.status += " The partial answer is unvalidated."
        state.response_time_seconds = max(0.0, clock() - started_at)
        yield _render(state)


def clear_demo() -> tuple[str, str, str, str, str, str]:
    """Reset the question and every response component to its initial state."""
    return "", *_render(DemoState())


def build_demo() -> gr.Blocks:
    initial = _render(DemoState())
    with gr.Blocks(title="Production Agentic RAG") as demo:
        gr.HTML(HERO_HTML)

        gr.HTML('<h2 class="rag-heading">Ask a research question</h2>')
        question = gr.Textbox(
            label="Research question",
            show_label=False,
            container=False,
            lines=3,
            placeholder="Ask a question grounded in AI, machine learning, and healthcare AI research papers…",
        )
        with gr.Row():
            ask = gr.Button("Ask", variant="primary", scale=4, min_width=160)
            clear = gr.Button("Clear", variant="secondary", scale=1, min_width=100)
        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question, label="Example questions")
        status = gr.HTML(initial[2])

        with gr.Row():
            with gr.Column(scale=7, min_width=320):
                with gr.Column(elem_classes="rag-card"):
                    gr.HTML('<h2 class="rag-heading">Answer</h2>')
                    answer = gr.Markdown(initial[0], elem_id="rag-answer", container=False)
                    metadata = gr.HTML(initial[4])
                gr.HTML('<h2 class="rag-heading">Sources</h2>')
                sources = gr.HTML(initial[1])
            with gr.Column(scale=4, min_width=260):
                with gr.Column(elem_classes="rag-card"):
                    gr.HTML('<h2 class="rag-heading">Execution path</h2>')
                    execution = gr.HTML(initial[3])

        with gr.Accordion("How this system works", open=False):
            gr.Markdown(HOW_IT_WORKS)
            gr.Image(
                value=str(ARCHITECTURE_IMAGE),
                label="Architecture: offline ingestion, agent query flow, validation, caching, and observability",
                show_label=False,
                interactive=False,
                buttons=["fullscreen"],
                container=False,
                elem_id="rag-architecture",
            )
            gr.Markdown("_Architecture of the deployed system. Use the fullscreen button to enlarge it._")

        outputs = [answer, sources, status, execution, metadata]
        ask.click(stream_demo, inputs=question, outputs=outputs)
        question.submit(stream_demo, inputs=question, outputs=outputs)
        clear.click(clear_demo, inputs=None, outputs=[question, *outputs], queue=False)
    return demo


def _render(state: DemoState) -> tuple[str, str, str, str, str]:
    return (
        state.answer or ANSWER_PLACEHOLDER,
        _render_sources(state.sources),
        f'<div class="rag-status" role="status"><strong>Status:</strong> {html.escape(state.status)}</div>',
        _render_execution(state),
        _render_metadata(state),
    )


def _render_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return '<p class="rag-empty">No validated sources available.</p>'
    cards: list[str] = []
    for source in sources:
        citation = html.escape(str(source.get("citation") or "[S?]"))
        title = html.escape(str(source.get("title") or "Untitled source"))
        source_type = SOURCE_TYPE_LABELS.get(source.get("source_type"))
        tone = "rag-info" if source_type == "LIVE ARXIV" else "rag-neutral"
        badge = f'<span class="rag-badge {tone}">{source_type}</span>' if source_type else ""
        facts = []
        if source.get("arxiv_id"):
            facts.append(f"arXiv: {html.escape(str(source['arxiv_id']))}")
        if source.get("section"):
            facts.append(f"Section: {html.escape(str(source['section']))}")
        published = _ISO_DATE.match(str(source.get("published_date") or ""))
        if published:
            facts.append(f"Published: {published.group()}")
        url = source.get("source_url")
        link = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">Open arXiv</a>'
            if isinstance(url, str) and url.startswith("https://arxiv.org/")
            else ""
        )
        content = str(source.get("content") or "")
        excerpt = content if len(content) <= SOURCE_EXCERPT_CHARS else f"{content[:SOURCE_EXCERPT_CHARS].rstrip()}…"
        details = (
            f"<details><summary>View source details</summary><p>{html.escape(excerpt)}</p></details>"
            if excerpt.strip()
            else ""
        )
        cards.append(
            '<article class="rag-source">'
            f'<div class="rag-source-head"><span class="rag-cite">{citation}</span>{badge}</div>'
            f'<div class="rag-source-title">{title}</div>'
            f'<div class="rag-source-meta">{" · ".join(facts)}</div>'
            f"{link}{details}</article>"
        )
    return f'<div class="rag-sources">{"".join(cards)}</div>'


def _render_execution(state: DemoState) -> str:
    """The ordered backend statuses, verbatim; no step is inferred or invented here."""
    if not state.steps:
        return '<p class="rag-empty">The execution path will appear here.</p>'
    steps = "".join(f"<li>✓ {html.escape(step)}</li>" for step in state.steps)
    return f'<ol class="rag-steps">{steps}</ol>'


def _grounding_badge(state: DemoState) -> tuple[str, str] | None:
    if state.terminal is None:
        return None
    if state.terminal == "error":
        return ("NOT VALIDATED" if state.answer else "NO VALIDATED ANSWER"), "rag-warn"
    if state.outcome in OUTCOME_BADGES:
        return OUTCOME_BADGES[state.outcome], "rag-warn"
    if state.grounding_passed is False:
        return "NOT VALIDATED", "rag-warn"
    return "GROUNDED", "rag-good"


def _format_latency(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def _render_metadata(state: DemoState) -> str:
    """Compact summary built only from fields the API returned; absent values are omitted, never zero-filled."""
    items: list[str] = []
    if state.cache_status:
        tone = "rag-good" if state.cache_status == "hit" else "rag-neutral"
        items.append(f'<span class="rag-badge {tone}" role="listitem">{CACHE_BADGES[state.cache_status]}</span>')
    grounding = _grounding_badge(state)
    if grounding:
        items.append(f'<span class="rag-badge {grounding[1]}" role="listitem">{grounding[0]}</span>')

    def chip(label: str, value: object) -> None:
        items.append(f'<span class="rag-chip" role="listitem"><b>{label}</b>{html.escape(str(value))}</span>')

    if state.response_time_seconds is not None:
        chip("Latency", _format_latency(state.response_time_seconds))
    if state.model:
        chip("Model", state.model)
    if state.sources:
        chip("Sources", len(state.sources))
    elif state.source_count is not None:
        chip("Sources retrieved", state.source_count)
    if state.prompt_tokens is not None or state.completion_tokens is not None:
        counts = [
            f"{count} {direction}"
            for count, direction in ((state.prompt_tokens, "in"), (state.completion_tokens, "out"))
            if count is not None
        ]
        chip("Tokens (original run)" if state.cache_status == "hit" else "Tokens", " / ".join(counts))
    if not items:
        return ""

    # Pipeline settings are only sent by the non-agent stream endpoint.
    details = [
        f"{label}: {html.escape(str(value))}"
        for label, value in (
            ("Retrieval", state.retrieval_mode),
            ("Retrieval size", state.retrieval_size),
            ("Embedding model", state.embedding_model),
            ("Embedding dimension", state.embedding_dimension),
            ("RRF k", state.rrf_k),
            ("Temperature", None if state.llm_temperature is None else f"{state.llm_temperature:g}"),
            ("Maximum completion tokens", state.max_completion_tokens),
            ("Observability", _observability(state)),
        )
        if value is not None
    ]
    extra = f'<div class="rag-details">{" · ".join(details)}</div>' if details else ""
    return f'<div class="rag-chips" role="list" aria-label="Response summary">{"".join(items)}</div>{extra}'


def _observability(state: DemoState) -> str | None:
    if not state.observability_provider:
        return None
    if state.observability_status:
        return f"{state.observability_provider} ({state.observability_status})"
    return state.observability_provider


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _optional_nonnegative_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _cache_status(value: object) -> str | None:
    return value if value in {"hit", "miss", "bypass", "failure"} else None


def server_binding() -> tuple[str, int]:
    """Loopback by default; a container sets GRADIO_SERVER_NAME=0.0.0.0 to be reachable on its network."""
    host = os.getenv("GRADIO_SERVER_NAME", "").strip() or DEFAULT_SERVER_NAME
    raw_port = os.getenv("GRADIO_SERVER_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_SERVER_PORT
    except ValueError as exc:
        raise ValueError("GRADIO_SERVER_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("GRADIO_SERVER_PORT must be between 1 and 65535")
    return host, port


def main() -> None:
    host, port = server_binding()
    build_demo().launch(server_name=host, server_port=port, share=False, theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
