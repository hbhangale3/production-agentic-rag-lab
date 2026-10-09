# Week 6 M03: exact caching for `POST /api/v1/ask`

M03 wraps only the non-streaming RAG endpoint with the deterministic M02 cache
contract. `RAGResponseCacheCoordinator` is application-scoped, depends on the
provider-neutral `AsyncCache`, and keeps Redis policy out of retrieval and RAG
generation services.

## Request flow

After FastAPI validates `AskRequest`, the coordinator obtains the configured
corpus fingerprint, builds the M02 identity from the exact question and current
answer-affecting settings, derives the private SHA-256 key, and performs the
cache read before retrieval.

A cache value is accepted only when the M02 deserializer reconstructs a valid,
version-compatible `AskResponse`. A valid hit is returned unchanged, including
its original retrieval mode, model, source order, and token counts. Therefore a
hit performs no query embedding, OpenSearch/BGE retrieval, evidence building,
Groq generation, or grounding validation.

On a miss, the existing Week 5 pipeline runs unchanged. Only after it produces
a complete successful `AskResponse` is the M02 envelope serialized and stored.
The write uses `RAG_CACHE_TTL_SECONDS` (86,400 seconds by default). Reads do not
refresh the TTL.

## Safety and failure policy

- `REDIS_ENABLED=false` bypasses cache reads and writes; Redis is not required.
- An unavailable corpus fingerprint bypasses both cache operations. No
  placeholder fingerprint or key is created.
- A missing value, failed read, or corrupt/incompatible value is a cache miss.
  Successful generation replaces a corrupt value at the same key.
- A failed write or serialization failure does not change a successful HTTP
  response.
- Retrieval, prompt-budget, provider, generation, and grounding failures are
  never cached.
- Successful insufficient-evidence responses are cached. This includes the
  deterministic empty-evidence response and citation-free insufficiency output
  accepted by the existing validator. They are HTTP-200 artifacts bound to the
  same corpus fingerprint as other successful answers.

The public `AskRequest` and `AskResponse` schemas are unchanged. Keys, hit state,
and cache metadata are not exposed to clients, and diagnostic logging contains
no question, answer, evidence, payload, credentials, or full cache key.

## Deferred work

`POST /api/v1/ask/stream` and its SSE contract are unchanged; streaming cache
replay belongs to M04. This milestone adds no Gradio behavior, semantic cache,
Langfuse, metrics, or distributed locking. Concurrent misses may independently
run RAG and overwrite the same deterministic key; request coalescing is a future
production optimization.
