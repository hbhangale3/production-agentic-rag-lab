# W5-M06 streaming grounded RAG

The existing non-streaming flow remains `retrieval -> evidence -> prompt -> Groq
complete -> JSON answer + authoritative sources`. The streaming flow is `retrieval
-> evidence -> prompt -> Groq stream -> SSE deltas -> authoritative sources ->
done`.

`POST /api/v1/ask/stream` uses server-sent events. Every `data` value is JSON and
successful streams have this order:

1. `metadata`: retrieval mode and source count.
2. One or more `delta` events containing incremental canonical answer text.
3. `sources`: every application-selected `AskSource` supplied to the model.
4. `done`: model, provider usage when available, and cited labels.

No usable evidence follows the same event sequence with the fixed insufficiency
answer, no sources, nullable model/usage, and no provider call. A failure after
headers are committed produces a terminal `error` event and no `sources` or
`done`; already-emitted text cannot be retracted. Retrieval, prompt-budget, and
configuration failures detected before `StreamingResponse` retain normal HTTP
error statuses.

Retrieval happens exactly once before streaming. It includes synchronous BGE and
OpenSearch work, so the route keeps it in Starlette's thread pool rather than
blocking the event loop. Evidence construction, prompt creation, the approximate
application token-budget check, and empty-evidence detection also happen before
the response begins. Groq iteration is asynchronous and direct: there is no
producer queue, so network consumption naturally provides backpressure.

Source metadata is application-controlled and is never accepted from model
output. The returned selected content is the exact excerpt—possibly truncated—
that the evidence builder placed in the prompt. At completion, citation labels
are deduplicated in first-occurrence order and checked against those sources.

Full-width citations require cross-delta handling because `【S1】` may arrive in
several chunks. The normalizer retains only a trailing possible `【S<digits>`
prefix, emits all other text promptly, and canonicalizes only completed
`【S<number>】` forms to `[S<number>]`. The canonical answer is assembled for final
allow-list validation. An unknown label terminates with `error` and does not
trigger a repair call.

The provider is worker-scoped. Each Groq `AsyncStream` is closed after completion,
failure, or cancellation, while the shared `AsyncGroq` client remains open until
application shutdown. Cancellation is allowed to propagate and stops direct
iteration without being converted to a generation error.

The configured completion ceiling increased from 512 to 1024 tokens because live
M05 output reached 512 and ended mid-sentence. Prompt/input tokens (system prompt,
question, source metadata, and excerpts) remain distinct from completion/output
tokens. The approximate character-based application budget reserves 1024 output
tokens plus the safety margin inside the 8192-token context window before calling
Groq; provider-returned usage, when present, remains authoritative.

## Local acceptance (2026-10-09)

Standalone Uvicorn used the process-local
`OPENSEARCH_HOST=http://127.0.0.1:9200` override. Ping and health returned 200,
and the hybrid-search precheck returned five results in hybrid mode. Corpus state
was 17 PostgreSQL papers, 17 paper-index documents, 513 chunk documents, and 15
distinct chunk paper IDs; the known 17-versus-15 discrepancy was not modified.

The healthcare-access stream returned 302 deltas and 1,595 answer characters.
Time to metadata was 0.394 seconds, time to the first generated delta was 1.434
seconds, and total duration was 2.093 seconds. Provider usage was 4,417 prompt
tokens and 583 completion tokens. The answer ended syntactically and did not
appear hard-truncated at the old 512-token ceiling.

The health-inequity stream returned 486 deltas and 2,467 answer characters. Time
to metadata was 0.191 seconds, time to first generated delta was 9.335 seconds,
and total duration was 10.347 seconds. Provider usage was 3,966 prompt tokens and
783 completion tokens. A natural 429 caused one bounded SDK retry and explains
most of its first-delta delay.

The independent non-streaming healthcare call took 39.779 seconds and returned
1,960 answer characters with 4,417 prompt and 565 completion tokens. It naturally
encountered two 429 responses and recovered through the configured SDK retries.
All three answers used `openai/gpt-oss-120b`, hybrid retrieval, five authoritative
sources, and only allowed citation labels. These stochastic calls are not expected
to produce identical prose; streaming improved perceived latency in this sample.
