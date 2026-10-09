# Week 7 M02: domain guardrail and local retrieval

W7-M02 replaces the foundation pass-through graph with a production-shaped
domain guardrail followed by one existing local hybrid retrieval attempt. It
does not expose an API or generate an answer.

## Scope and topology

```text
START
  -> graph_start
  -> guardrail
       rejected -> out_of_scope -> END
       failed   -----------------> END
       passed   -> local_retrieval -> END
```

`AgentGraphDependencies` receives an existing provider-neutral `LLMProvider`
and `HybridSearchService`. Nodes never create Groq, BGE, OpenSearch, Redis,
PostgreSQL, or Langfuse clients. The graph is still constructed explicitly and
is not integrated into the application lifecycle or Week 6 endpoints.

## Guardrail domain and scoring

The guardrail answers only whether a question belongs to the project's
scientific-paper domain: AI, machine learning, NLP, healthcare AI, health
equity technology, and technical healthcare applications of AI. General
knowledge, weather, sports, creative-writing, and unrelated questions are out
of scope. This is domain routing, not moderation, evidence grading, answer
grounding, or factuality grading.

`GuardrailPromptBuilder` keeps the local replaceable prompt separate from graph
orchestration and evaluation. It supplies the question as JSON-encoded
untrusted data and directs the provider not to answer, retrieve, explain, or
follow instructions embedded in that data. No prompt or question enters an
execution event.

The existing asynchronous `LLMProvider.complete` boundary is called with a
zero temperature and a small completion allowance. The only accepted response
is exactly:

```json
{"score": 82}
```

Surrounding whitespace or a single Markdown code fence around that object is
tolerated because chat models commonly add one; the object inside is validated
identically. The score must be an integer from 0 through 100; booleans, floats, missing or
additional fields, prose, malformed JSON, empty content, and out-of-range
values fail safely. Scores below the configured threshold are rejected, while
scores equal to or above it pass. The default threshold remains 60.

Provider and parser failures produce `internal_error`, `guardrail_failure`, and
`graph_failed`. They are never represented as out-of-scope and no raw
exception or completion enters state events.

## Routing and local retrieval

An out-of-scope result records the score and false decision, emits
`graph_completed`, sets `terminal_reason=out_of_scope`, and ends before
retrieval. This is a normal expected domain outcome.

An accepted question calls the existing `HybridSearchService.search` with
`current_query` and `AgentGraphConfig.retrieval_size`. The synchronous search
is offloaded with `asyncio.to_thread` so it does not directly block the async
graph path. The attempt count changes from zero to one. The existing `hybrid`,
`bm25_fallback`, and `vector_fallback` results are all successful retrievals;
the graph does not duplicate BM25, vector, or RRF decisions.

Retrieval failure records one attempted retrieval, leaves the result unset,
sets `terminal_reason=retrieval_failed` and `error_category=retrieval_failure`,
and emits `graph_failed` without leaking an exception message.

Successful retrieval leaves `terminal_reason` unset. `graph_completed` means
the current M02 topology ended normally, not that a final answer exists.

## Safe event sequences

Pass and retrieval success:

```text
graph_started
guardrail_started
guardrail_passed
local_retrieval_started
local_retrieval_completed
graph_completed
```

Domain rejection:

```text
graph_started
guardrail_started
guardrail_rejected
graph_completed
```

Guardrail failure:

```text
graph_started
guardrail_started
graph_failed
```

Retrieval failure:

```text
graph_started
guardrail_started
guardrail_passed
local_retrieval_started
graph_failed
```

No path emits both `graph_completed` and `graph_failed`. Metadata remains
allowlisted and content-free; no chain-of-thought or model rationale is stored.

## W7-M03 boundary

**Retrieved does not mean sufficient.** W7-M02 only stores local candidates in
`local_retrieval_result`. It leaves `evidence_sufficient` and `evidence_grade`
unset and creates no generated answer. W7-M03 can insert an evidence-grading
node directly after `local_retrieval` without replacing retrieval behavior.

Query rewriting, a second retrieval, live arXiv search, PDF/Docling processing,
transient evidence, generation, semantic answer grounding, prompt management,
cost tracking, cache integration, observability changes, agentic HTTP/SSE, and
Gradio integration remain deferred.

## Verification

- Agent foundation and M02 tests: 105 passed.
- Focused LLM/retrieval/RAG/cache/observability/API/lifecycle regression: 372 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 766 passed
  (W7-M01 baseline: 685), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fake providers and retrievers; no external service is called.
