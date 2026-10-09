# Week 6 M05: cache health and operational visibility

Redis remains an optional performance dependency. `GET /api/v1/health` now
contains a bounded `cache` snapshot while preserving the existing top-level
core status and database service check. Cache degradation alone does not change
the endpoint from HTTP 200 or mark core RAG unavailable.

## Health states

- `disabled`: `REDIS_ENABLED=false`; no backend health call is made.
- `healthy`: caching is enabled and the provider-neutral cache health check
  succeeds.
- `unavailable`: caching is enabled but construction, connectivity, health
  checking, or its timeout fails.

Health calls reuse the worker-scoped `AsyncCache`; they never construct a Redis
client. `REDIS_HEALTH_TIMEOUT_SECONDS` bounds the check and defaults to one
second. Startup performs no mandatory ping, so unavailable Redis cannot prevent
application startup.

The safe health snapshot exposes only enabled state, generic backend name,
status, response TTL, cache schema version, counters, and hit rate. It never
exposes the Redis URL, credentials, cache keys, questions, answers, evidence, or
payloads.

## Process-local counters

One `CacheStats` instance belongs to the shared `RAGResponseCacheCoordinator`,
so `/ask` and `/ask/stream` contribute to the same totals:

- `hits`: an eligible lookup produced a successfully deserialized response.
- `misses`: an eligible backend lookup had no value, or its value was invalid.
- `bypasses`: no lookup was attempted because caching was disabled, corpus
  fingerprint state was unavailable, or identity construction failed.
- `writes`: a validated response was successfully stored.
- `read_failures`: a backend read failed or the cache adapter raised. These are
  intentionally distinct from misses and excluded from hit rate.
- `write_failures`: serialization failed, a backend write returned false, or a
  write raised.
- `invalid_entries`: a retrieved value failed the M02 envelope validation. It
  is also counted as a miss.

Hit rate is `hits / (hits + misses)`; bypasses and backend failures are excluded.
It is `0.0` before any hit/miss outcome.

The provider-neutral `AsyncCache` gained `get_result`, which distinguishes hit,
miss, and backend failure without leaking Redis exceptions or types. Existing
`get` behavior remains fail-open and compatible. The coordinator retains a
legacy fallback for test or future adapters that only implement `get`.

Counters are lightweight, best-effort, process-local values. They reset on
restart and are not aggregated across multiple workers. They are not stored in
Redis, PostgreSQL, files, or an external metrics service. Instrumentation
failure cannot make a RAG request fail.

No answer identity fields or namespace versions changed. `AskRequest`,
`AskResponse`, SSE events, cached replay, and Gradio are unchanged. This
milestone adds no persistent telemetry, Prometheus, OpenTelemetry, Langfuse,
cache stampede protection, or distributed locking.
