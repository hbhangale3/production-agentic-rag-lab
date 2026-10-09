# Week 6 final: cached and observable RAG

Week 6 adds optional exact-response caching, operational cache visibility,
privacy-first Langfuse tracing, and a Gradio request-details panel without
changing answer correctness or making Redis/Langfuse hard dependencies.

## Final architecture

```text
Browser
  -> Gradio
  -> POST /api/v1/ask/stream
  -> rag.request
  -> exact cache lookup
       HIT
         -> validated AskResponse envelope
         -> one-delta cached SSE replay
       MISS / BYPASS / FAILURE
         -> BM25 + BGE vector retrieval + RRF
         -> evidence context construction
         -> Groq streaming generation
         -> structural grounding validation
         -> validated AskResponse
         -> best-effort Redis write
         -> SSE sources + done

Langfuse observes safe execution metadata and never controls correctness.
```

## Exact cache

Both `/ask` and `/ask/stream` use the same canonical M02 identity, private
SHA-256 key, `rag:response:v1` namespace, and validated `AskResponse` envelope.
Identity includes the exact question and answer-affecting model, retrieval,
evidence, generation, grounding, index, and corpus state. It excludes TTL,
Redis location, counters, timing, and traces.

`RAG_CORPUS_GENERATION` is the manual invalidation boundary until ingestion
owns automatic generation updates. A valid hit bypasses embedding, OpenSearch,
evidence building, Groq, and grounding. Streaming hits replay the complete
answer in one deterministic Unicode-safe delta rather than simulating model
token boundaries. Cached model/token fields describe the original artifact.

Redis remains optional and fail-open. Health distinguishes disabled, healthy,
and unavailable states. Process-local counters cover hits, misses, bypasses,
writes, read/write failures, invalid entries, and hit rate; they reset on
restart and are not aggregated between workers.

## Observability and privacy

The provider-neutral observability layer uses one `rag.request` root with
actual executed children for cache lookup/write, retrieval, evidence,
generation, and grounding. Cache-hit traces contain only cache lookup. Export
and SDK failures are best-effort and never replace the RAG result.

Tracing remains metadata-only even if content capture is enabled. Questions,
prompts, evidence, answers, source chunks, cache keys/payloads, streaming
deltas, credentials, and arbitrary exception messages are not exported. There
is no automatic HTTP/provider instrumentation, per-request flush, or remote
Langfuse health probe.

## Gradio visibility

The existing SSE `metadata` payload now includes optional backward-compatible
fields for the authoritative cache outcome (`hit`, `miss`, `bypass`, or
`failure`) and safe worker configuration. No event type was added. Gradio shows:

- cache outcome (`failure` is presented as unavailable caching, not request failure)
- total client-observed response time using a monotonic clock
- retrieval mode and selected source count
- model and token usage, with cache-hit tokens labeled as original usage
- temperature, completion allowance, retrieval size, embedding configuration,
  and RRF k
- configured observability provider/status

Gradio remains only an HTTP/SSE client and creates no embedding, retrieval,
Redis, Groq, Langfuse, or RAG services. Provider-neutral trace IDs and trace
URLs were deferred: the current abstraction has no stable identifier, and no
Langfuse URL format is guessed.

## Acceptance and latency

Automated offline tests prove miss/write/hit behavior, cross-endpoint reuse,
expensive-work bypass, resilience, trace shapes, privacy, metadata compatibility,
and deterministic UI timing.

Live Redis acceptance on 2026-10-09 used one unique streaming question followed
by four exact repeats against the loopback API and Compose Redis. The observed
outcomes were `miss, hit, hit, hit, hit`; all four cached responses had the same
answer, sources, and completion metadata as the original response. Cache health
remained `healthy`, with one miss, one write, four hits, and an 80% hit rate.
The miss took 22,671.61 ms and the cached repeats took 52.12, 51.57, 49.24, and
53.49 ms (51.60 ms mean, approximately 439x faster than the miss). These are
local measurements, not portable performance claims.

Live Langfuse export acceptance was skipped because Langfuse credentials were
not configured in the ignored local environment. Langfuse remains covered by
the provider-contract, resilience, privacy, and trace-shape test suites.

## Known limitations and deferred work

- Exact cache only; paraphrases miss and there is no semantic/fuzzy cache.
- Corpus invalidation requires a manual `RAG_CORPUS_GENERATION` bump.
- Concurrent identical misses are not coalesced; there is no distributed lock.
- CacheStats are process-local with no cross-worker aggregation.
- One API worker remains preferred because each worker owns a BGE model.
- Langfuse export is best-effort and remote health is not actively probed.
- Tracing is metadata-only with no automated RAG quality evaluation.
- No semantic evidence grading, query rewriting, live arXiv fallback, or
  LangGraph orchestration has been started.
- The previously observed 17-paper PostgreSQL versus 15-paper chunk-index
  discrepancy remains documented technical debt and was not repaired.
