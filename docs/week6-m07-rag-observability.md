# Week 6 M07: end-to-end RAG observability

Both RAG transports now create one provider-neutral `rag.request` root
observation. Explicit application-level children use stable names:

- `rag.cache.lookup`
- `rag.retrieval`
- `rag.evidence`
- `rag.generation`
- `rag.grounding`
- `rag.cache.write` when an eligible write is attempted

No Langfuse SDK type or import appears outside the observability package.
`SafeObservation` guards every start, update, and completion call so telemetry
failure cannot replace a RAG result.

## Trace shapes

A normal miss produces cache lookup, retrieval, evidence, one generation span,
grounding, and optional cache-write observations. `/ask/stream` keeps the root
and its single generation span open across actual provider iteration; it never
creates a span per delta. The root is completed only after validated response
construction and best-effort cache storage, immediately before the terminal
successful response or `done` event.

A hit produces only the root and `rag.cache.lookup[outcome=hit]`. Retrieval,
evidence, generation, and grounding are neither executed nor represented by
fake spans. Stored model and token values use `artifact_*` names because they
describe the cached answer, not work performed by the current request.

Disabled or ineligible caching records `bypass`; a clean eligible absence is a
`miss`; a structured backend read error is `failure`. Cache read/write failure
does not mark an otherwise answered request as failed. Eligible writes record
`stored` or `failed`; no key, payload, or serialized answer is attached.

## Safe metadata

Retrieval records requested size, result count, and actual retrieval mode.
Evidence records candidate and selected counts, context token estimate/budget,
and truncation count. Generation records streaming mode, configured generation
limits, actual returned model, and provider-reported token counts. Grounding
records pass/fail, source/citation counts, and citation-free insufficiency.
Roots record transport, cache outcome, final retrieval mode, source count,
model/tokens, grounding state, and a bounded success/failure classification.

Failure classifications are limited to `retrieval_failure`,
`evidence_failure`, `generation_failure`, `grounding_failure`, `cancelled`, and
`internal_failure`. Child errors use exception class names or safe cache
classifications, never `str(exception)`.

Valid empty-evidence and citation-free insufficiency responses remain success
traces. Only operations actually executed are present.

## Streaming and cancellation

Provider failure after deltas marks the single generation span and root as
failed, preserves the SSE `error` contract, and performs no cache write.
Grounding failure completes generation, marks grounding/root as failed, and
also emits no `sources` or `done`. `CancelledError` and async-generator close
are classified as `cancelled`, active generation/root observations close
best-effort, cancellation propagates, and no partial response is cached.

## Privacy and lifecycle

`LANGFUSE_CAPTURE_CONTENT=false` exports metadata only. Questions, prompt
messages, evidence, answers, source chunks, cache keys/payloads, streaming
deltas, and arbitrary exception messages are absent. M07 intentionally remains
metadata-only even when `LANGFUSE_CAPTURE_CONTENT=true`; bounded explicit
content export is deferred rather than weakening the false-mode guarantee.

There is no automatic HTTP/FastAPI/Groq/Redis/OpenSearch instrumentation, no
per-request flush, and no remote Langfuse health check. Application shutdown
retains SDK batching and flush ownership from M06. CacheStats, cache identity,
public request/response schemas, SSE events, and Gradio remain unchanged. M08
may decide how to present observability without requiring trace data for RAG
correctness.
