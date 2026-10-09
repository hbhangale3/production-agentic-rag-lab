# W6-M01 asynchronous Redis cache foundation

This milestone adds optional Redis infrastructure for future exact grounded-RAG
response caching. It does not cache `/api/v1/ask` or `/api/v1/ask/stream` yet,
and it makes no cache-performance claims.

## Architecture and behavior

`AsyncCache` is a provider-neutral asynchronous string-cache protocol with
`get`, `set` with TTL, `delete`, `ping`, `health`, and `close`. Redis SDK types
remain inside `RedisCache`. The adapter uses `redis.asyncio.Redis`, its connection
pool, UTF-8 string responses, and application-owned worker lifecycle.

All storage operations are fail-open:

- a failed GET is a cache miss;
- a failed SET or DELETE returns false;
- failed PING reports unavailable;
- SDK exceptions and secret-bearing connection details do not escape;
- close is idempotent and shutdown-safe.

Factory-created clients are owned and closed once by the application. Injected
clients are caller-owned and are not closed. Disabled configuration returns a
deterministic no-op cache and constructs no Redis client. Client construction
failure returns a no-op cache whose internal health state is `unavailable`.
Normal Redis connectivity is lazy, so temporary unavailability cannot prevent
FastAPI startup.

The application stores one cache object in `app.state.cache`, but no endpoint or
RAG service consumes it in M01. Public request/response schemas, retrieval,
generation, SSE, and Gradio behavior are unchanged.

## Configuration

```dotenv
REDIS_ENABLED=false
REDIS_URL=redis://127.0.0.1:6379/0
RAG_CACHE_TTL_SECONDS=86400
```

Redis is disabled by default. TTL must be positive. An enabled cache requires a
nonblank URL. The URL is represented as a secret and is never logged.

Compose provides `redis:7-alpine`, persistent `redis_data`, a PING healthcheck,
256 MiB `maxmemory`, and `allkeys-lru` eviction. Its host port binds only to
`127.0.0.1:6379`; Redis is not publicly exposed. The API container receives the
internal `redis://redis:6379/0` address when caching is enabled later.

## Future exact-response cache invariants

The reserved namespace is `rag:response:v1:<hash>`. W6-M02 will define the exact
key inputs and invalidation policy; M01 intentionally does not derive keys.

Only an answer that has completed retrieval, generation,
`GroundedAnswerValidator`, and `AskResponse` construction may eventually be
stored. Partial streams and retrieval, prompt-budget, provider, grounding, and
HTTP-validation failures must never be cached. Final endpoint integration is
deferred to W6-M03.

Langfuse and all observability integration are also deferred. M01 adds no tracing,
semantic cache, query normalization, cache status fields, or endpoint behavior.

## Local acceptance (2026-10-09)

The Compose Redis service became healthy and the async adapter successfully
performed PING, SET with a 30-second TTL, GET, and DELETE using only the dedicated
`rag:test:w6m01` key. The key was absent after cleanup and the test container was
stopped afterward. Runtime configuration reported 256 MiB maxmemory,
`allkeys-lru`, and append-only persistence. Redis used approximately 3.4 MiB at
idle; the host retained approximately 9.0 GiB available memory with no swap. No
Groq, retrieval, or RAG endpoint request was made for this acceptance.
