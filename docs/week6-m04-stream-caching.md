# Week 6 M04: exact caching for streaming RAG

`POST /api/v1/ask` and `POST /api/v1/ask/stream` now use the same
application-scoped `RAGResponseCacheCoordinator`. Transport is not part of the
answer identity, so both endpoints derive the same M02
`rag:response:v1:<sha256>` key and store the same validated `AskResponse`
envelope. No SSE events or model-token boundaries are stored.

## Cache-hit replay

The streaming endpoint checks the cache before retrieval. A valid hit bypasses
BGE/OpenSearch retrieval, evidence and prompt construction, Groq streaming, and
grounding validation. It is emitted as a cached replay using the existing
successful SSE sequence:

1. `metadata` with the original retrieval mode and source count
2. one `delta` containing the complete cached answer
3. `sources` containing the cached sources in authoritative order
4. `done` containing the original model and token counts

One Python-string delta is deterministic, preserves Unicode and citation text
exactly, and does not pretend to reconstruct Groq token boundaries. The
existing metadata payload is unchanged; no cache indicator or new event type
was added. Ordered unique `done.citations` are recovered from the canonical
cached answer.

## Cache misses and writes

A miss retains genuine provider streaming. The router reuses the canonical
deltas already emitted by `RAGGenerationService` to assemble the final answer;
it does not maintain a second generation path. Provider-reported model and
token usage are copied into the cache response without estimation.

Storage occurs only after the stream reaches `RAGStreamComplete`, which means
generation and grounding validation succeeded, and after a complete
`AskResponse` has been constructed. The best-effort store occurs before
`sources` and `done`; serialization or Redis write failure cannot prevent those
success events. Provider failures, grounding failures, incomplete streams, and
cancelled generators before normal completion do not reach the store path.

The deterministic empty-evidence success and citation-free insufficiency
answers accepted by the existing validator follow the same M03 policy and may
be cached under the current corpus fingerprint.

## Failure and lifecycle policy

- Disabled Redis or an unavailable corpus fingerprint bypasses reads and
  writes while preserving genuine streaming.
- Read failures and corrupt entries are misses; a later successful stream may
  replace the entry.
- Cache hits do not refresh `RAG_CACHE_TTL_SECONDS`.
- The worker-scoped cache and coordinator are reused; no per-stream Redis
  client is created or closed.
- If a client disconnects, Starlette cancellation remains authoritative. A
  response is stored only if the async generator advances through its normal
  validated completion path; no partial-response recovery is attempted.

Cross-endpoint tests cover both `/ask` to `/ask/stream` and the reverse. The
existing Gradio parser remains compatible because event names and payload
requirements are unchanged. This milestone adds no UI indicator, semantic
cache, Langfuse integration, metrics, artificial replay delay, TTL refresh, or
stampede protection. Concurrent identical misses may still generate in
parallel.
