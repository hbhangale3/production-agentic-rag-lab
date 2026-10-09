# W6-M02 deterministic RAG cache contract

This milestone defines exact cache identity, versioned keys, corpus invalidation,
and validated response serialization. It performs no Redis I/O and does not wire
caching into `/api/v1/ask` or `/api/v1/ask/stream`.

## Exact identity and key

The key format is:

`rag:response:v1:<64-character lowercase SHA-256>`

The digest input is UTF-8 deterministic JSON using sorted object keys, compact
separators, and unescaped Unicode. Raw questions, prompts, source content, and
model output never appear in the Redis key.

`RAGCacheIdentity` contains the answer-affecting state currently used by the
application:

- exact validated question;
- LLM provider and configured model;
- prompt contract version;
- retrieval contract version and requested result size;
- embedding model and dimension;
- RRF k, candidate multiplier, and maximum candidate results;
- chunk index name;
- evidence contract version and evidence token budget;
- generation temperature and maximum completion tokens;
- grounding contract version;
- corpus fingerprint; and
- cache schema version.

The existing `AskRequest` validates nonblank input but preserves the submitted
string. Cache identity therefore also preserves surrounding whitespace, case,
punctuation, and Unicode. No lowercasing, punctuation stripping, paraphrase
matching, embedding, or fuzzy comparison occurs.

TTL, Redis enabled state/URL/maxmemory, API and Gradio hosts, log level, runtime
latency, health/fallback state, context-window ceiling, and safety margin are not
included. The last two guard whether a request may execute but do not change the
prompt/evidence/model request after it passes; material changes to evidence
availability instead require the explicit evidence version or budget to change.

## Version boundaries

Current versions are:

- cache schema: `v1`;
- prompt: `grounded-rag-v1`;
- retrieval: `hybrid-bm25-bge-rrf-v1`;
- evidence: `evidence-context-v1`;
- grounding: `structural-grounding-v1`.

Bump the cache schema version for incompatible envelope or key-contract changes.
Bump prompt version for material grounding-instruction changes. Bump retrieval
version for BM25 query/boost, candidate-generation, embedding, vector, or RRF
policy changes not already represented by an explicit identity value. Bump
evidence version for material selection, rendering, or truncation changes. Bump
grounding version when answer-acceptance semantics change. Behavior-preserving
comments and refactors need no bump.

## Corpus fingerprint

The current repository has no cheap authoritative ingestion generation counter.
M02 therefore adds `RAG_CORPUS_GENERATION=corpus-v1`. A configured fingerprint
provider hashes this explicit generation token together with the chunk-index
name. This costs only one small in-process SHA-256 operation and performs no
PostgreSQL/OpenSearch enumeration or I/O.

This mechanism is deliberately manual: operators or future ingestion code must
increment the generation token after answer-relevant chunk-index changes. M02
does not claim automatic invalidation. Missing/blank generation or index state
returns `None`, which is an explicit future signal to bypass caching rather than
sharing an unsafe `unknown` fingerprint.

## Cached response envelope

The JSON envelope is:

```json
{"response": {"...": "complete AskResponse"}, "schema_version": "v1"}
```

Serialization uses deterministic JSON only—never pickle, eval, repr, SDK objects,
prompts, vectors, or hidden reasoning. It preserves the exact answer, source
order and all public source fields, retrieval mode, original model, and original
prompt/completion token counts.

Deserialization treats Redis data as untrusted. It requires an object root, the
supported envelope version, a complete Pydantic-validated `AskResponse`, valid
source structures/citation syntax, nonnegative usage, and a nonblank answer.
Malformed JSON, incompatible versions, missing fields, wrong types, and invalid
responses become cache misses; no partially trusted response is returned.
Harmless additional response fields follow the existing public Pydantic schema's
ignore policy, while unknown envelope-level fields are rejected.

## Deferred endpoint policy

M03 will integrate non-streaming lookup/store and M04 will define streaming
replay. Until then, no endpoint builds keys or reads/writes Redis.

Only responses that successfully complete retrieval, generation, structural
grounding validation, and `AskResponse` construction may eventually be stored.
Request, retrieval, prompt-budget, provider, grounding, malformed-answer, partial
stream, and SSE-error results must never be cached. The policy for successful
explicit insufficiency responses remains an M03 decision. Langfuse, semantic
caching, cache indicators, and performance claims are also deferred.
