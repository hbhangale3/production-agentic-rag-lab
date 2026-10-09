# Week 5 grounded RAG and Gradio demo

Week 5 implements standard grounded RAG. It is not yet the agentic retrieval and
fallback system planned for Week 7.

## Architecture

The authoritative path is:

`question -> FastAPI -> Week 4 hybrid retrieval -> EvidenceContextBuilder -> grounded prompt -> Groq -> GroundedAnswerValidator -> response`

Retrieval remains local BGE-small embeddings plus OpenSearch BM25/vector search
and reciprocal-rank fusion. Groq is the generation provider. The evidence builder
selects deterministic, citation-labeled excerpts within the configured budget.
The system prompt treats the question and retrieved evidence as untrusted data,
requires supplied citations for substantive claims, and permits explicit
evidence-insufficiency responses.

The accepted citation form is `[S<number>]`, beginning at S1. Full-width
`【S<number>】` is canonicalized to the accepted form. Unknown, malformed, and S0
citations are rejected. Deterministic grounding validation is structural rather
than semantic: a valid `[S1]` does not prove that S1 entails the preceding claim.

## API contracts

- `POST /api/v1/ask` returns the complete JSON answer and authoritative sources.
- `POST /api/v1/ask/stream` provides genuine incremental server-sent events.

A successful SSE sequence is:

`metadata -> delta(s) -> sources -> done`

A post-start failure is:

`metadata -> optional delta(s) -> error`

Only `done` establishes validated success. Text before `done` is provisional;
already-streamed text cannot be retracted. Sources come exclusively from the
application-controlled `sources` event, never from model prose.

## Gradio presentation layer

`src.gradio_app` is a local demonstration interface. Its only backend dependency
is HTTP/SSE through `{RAG_API_BASE_URL}/api/v1/ask/stream`. It does not instantiate
retrieval, embedding, evidence, RAG, OpenSearch, or Groq services, and it never
needs `GROQ_API_KEY`.

The UI incrementally displays canonical answer deltas and preserves `[S#]`
citations. It shows source cards only after the authoritative source event. A
request is marked successful only after `done`; an SSE error or premature close
marks any partial answer unvalidated. HTTP 422, 502, 503, timeout, connection, and
protocol failures are presented as concise safe messages without automatic
retry. Clear resets only local UI fields.

## Local startup

Install the locked dependencies with `uv sync`. With PostgreSQL and OpenSearch
available on their host-exposed ports, start the API:

```bash
OPENSEARCH_HOST=http://127.0.0.1:9200 \
  uv run uvicorn src.main:app --host 127.0.0.1 --port 8001
```

In another terminal, start Gradio:

```bash
RAG_API_BASE_URL=http://127.0.0.1:8001 \
  uv run python -m src.gradio_app
```

Gradio binds to `127.0.0.1:7860` with sharing disabled. The API base URL defaults
to `http://127.0.0.1:8001`; set `RAG_API_BASE_URL` when the API is elsewhere.
FastAPI alone owns the ignored runtime `GROQ_API_KEY`.

For an SSH-tunneled VPS demo, keep both processes loopback-only on the VPS and
forward the Gradio port from the local machine:

```bash
ssh -L 7860:127.0.0.1:7860 user@vps
```

Then open `http://127.0.0.1:7860` locally. No firewall or public Gradio share is
required. Port 8001 can also be forwarded for direct API inspection if desired.

## Current limitations

- The local corpus is small. Canonical/paper storage contains 17 papers while
  the chunk index currently represents 15 distinct papers.
- Hybrid retrieval returns the nearest available evidence when possible, even
  for unrelated questions.
- Structural citation validation does not prove semantic evidence support.
- Out-of-domain questions can still fail deterministic validation.
- There is no semantic evidence-sufficiency grader, live arXiv fallback, query
  rewriting, LangGraph flow, Redis cache, or Langfuse observability yet.
- Groq provider/free-tier limits can increase latency through bounded SDK retries.
- Lazy BGE startup is relatively expensive on CPU and adds substantial API-worker
  memory; Gradio does not load BGE or another model.

## Final local acceptance (2026-10-09)

FastAPI ran on `127.0.0.1:8001` with the process-local OpenSearch override, and
Gradio ran on `127.0.0.1:7860` with `share=False`. Ping, health, hybrid search,
non-streaming ask, and streaming ask all returned 200. Direct API answers used
five authoritative sources and only allowed citation labels.

The supported healthcare question was then submitted through the running Gradio
server. The Gradio client observed 535 incremental UI updates. The first answer
update arrived after 20.581 seconds and the completed UI state after 22.185
seconds; one natural 429 caused a 19-second SDK-level retry. The final answer was
2,752 characters, ended naturally, preserved `[S1]`, `[S2]`, `[S5]`, and `[S3]`,
and displayed five source cards `[S1]` through `[S5]`. Execution metadata showed
hybrid retrieval, five sources, `openai/gpt-oss-120b`, 4,440 prompt tokens, and
804 completion tokens. The terminal status changed to grounding validation
passed only after the API's `done` event.

One out-of-domain sourdough question was submitted through Gradio. A natural 429
caused one bounded SDK retry. The UI retained the 190-character partial answer,
showed no source cards, and clearly marked it unvalidated after the API's
grounding error. No automatic retry or replacement answer was attempted.

The local component configuration confirmed the title, research-question input,
Ask/Clear controls, answer panel, authoritative-source panel, status, and
execution panel. Five sources remained usable in the generated Markdown layout;
a graphical browser screenshot was not available in this execution environment.
After BGE initialization the API worker used approximately 1,002 MiB RSS. The
Gradio process used approximately 165 MiB RSS and did not load BGE or another
model. Host memory remained healthy with about 8.4 GiB available and no swap.
