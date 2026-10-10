# Week 7 M10: agent API, cache integration, Gradio, and live acceptance

W7-M10 puts the LangGraph agent behind production endpoints, adds the
agent-aware Redis cache, streams safe execution statuses, updates the Gradio
demo, and records what happened when the whole path was run against the real
services.

## Request path

```text
request
  -> validate
  -> resolve the prompt bundle once (concurrent, one Langfuse timeout window)
  -> build the agent cache key from that bundle's identities
  -> Redis lookup
       HIT  -> return the validated grounded answer; the graph is not built or run
       MISS -> run LangGraph with the same bundle, under one request deadline
               -> map the terminal state to a public outcome
               -> cache only a grounded, cited answer
               -> return answer, sources, and execution metadata
```

## Dependency wiring

`make_agent_service` (`src/services/agent/wiring.py`) is called once per
worker in the FastAPI lifespan, only when the LLM provider is configured.

| Reused from the worker | New for the agent |
|---|---|
| LLM provider | a dedicated `ArxivClient` with short online bounds |
| hybrid search service | `ArxivLiveSearchService` |
| the one lazy BGE embedding provider | `LiveDocumentProcessingService` + `LivePaperAcquisitionService` |
| RAG generation service and its evidence builder | `LazyPDFParser` (Docling created on first use) |
| Redis cache client | `AgentResponseCache` |
| observability provider | prompt resolver from `make_prompt_resolver` |

No second embedding model is created. Startup loads neither BGE nor Docling:
BGE loads on the first retrieval, and Docling is imported and constructed the
first time a live paper is parsed. `LazyPDFParser` holds a lock, so one worker
parses one PDF at a time across all requests. Shutdown closes the agent's
arXiv client.

`AgentGraphConfig` takes `retrieval_size` from `RAG_RETRIEVAL_SIZE`; its other
fields use their defaults.

## Prompt resolution

`AgentService.prepare` resolves the seven-prompt bundle once per request with
`LANGFUSE_PROMPT_LABEL` and `LANGFUSE_TIMEOUT_SECONDS`, then uses that same
bundle object for the cache identity and for the graph. Compiled graphs are
kept in a four-entry LRU keyed by the bundle's identities, so a graph is
compiled once per distinct prompt set, not per request.

## Agent cache

| | |
|---|---|
| Namespace | `rag:agent-response:v1:<sha256>` (the W7-M09 identity contract) |
| Value | `{schema_version, pipeline_version, response}` |
| `response` | answer, sources, `grounding_passed: true`, model, prompt/completion tokens, execution summary |
| TTL | `RAG_CACHE_TTL_SECONDS` |
| Corpus | `RAG_CORPUS_GENERATION` through the existing fingerprint provider |
| Identity additions in M10 | `length_recovery_max_tokens`, `length_recovery_multiplier` |

The value contains no question, prompt, candidate evidence, embeddings,
earlier rejected answer, or reasoning. The Week 6 namespace `rag:response:v1`
is untouched.

**Write only when all hold:** `terminal_reason` is `None`, there is no error
category, `grounding_passed` is `True`, a generation result exists, its finish
reason is not `length`, the answer cites at least one source, and at least one
source is present. A recovered answer is cacheable on the same terms; a
partial answer never is. Out-of-scope,
insufficient-evidence, grounding-failed, failed, and timed-out requests are
never cached. A citation-free "evidence is insufficient" answer is returned
but not cached, so a transient gap cannot become sticky.

**Read back as untrusted:** the stored JSON is validated against a strict
schema (unknown fields rejected, non-blank answer, at least one source,
`grounding_passed` literally true, matching schema and pipeline versions).
Anything else counts as a miss and an invalid entry, and the graph runs.

**Fail-open:** a read error or unavailable Redis runs the graph and reports
`cache_status=failure`; a write error still returns the grounded answer; a
disabled cache reports `bypass`.

**On a hit** nothing else runs: no graph, guardrail, retrieval, embedding,
LLM call, arXiv request, Docling, or grounding. Only prompt resolution and
the lookup happen.

## Outcomes

A model-written answer is public only when the run ended with no terminal
reason, a generation result exists, and semantic grounding passed. An answer
left in state after grounding was exhausted is never returned.

| Agent result | HTTP | `outcome` | `answer` |
|---|---|---|---|
| grounded answer | 200 | `answered` | the grounded answer |
| out of scope | 200 | `out_of_scope` | fixed safe message |
| insufficient evidence | 200 | `insufficient_evidence` | fixed safe message |
| grounding exhausted | 200 | `grounding_failed` | fixed safe message; no model text |
| answer structurally invalid, or incomplete after recovery | 200 | `generation_failed` | fixed safe message; no model text |
| retrieval failed | 503 | - | `detail`: fixed safe message |
| generation failed (provider error) | 502 | - | `detail`: fixed safe message |
| request deadline | 504 | - | `detail`: fixed safe message |
| internal error | 500 | - | `detail`: fixed safe message |
| agent not configured | 503 | - | `detail`: fixed safe message |

Structural and semantic failures are different classes and stay different in
agent state:

| What happened | `terminal_reason` | `error_category` | Public |
|---|---|---|---|
| answer has no citation, an unknown label, or a malformed label | `generation_failed` | `answer_validation_failure` | 200, `generation_failed`, fixed message |
| answer cut off at the token limit, and the one recovery cut off too | `generation_failed` | `generation_incomplete` | 200, `generation_failed`, fixed message |
| provider or internal error during generation | `generation_failed` | `generation_failure` | 502 |
| complete, cited answer scored below the grounding threshold on every allowed attempt | `grounding_failed` | none | 200, `grounding_failed`, fixed message |

In the first two rows the model did respond, so the public result is a
controlled outcome with the fixed message "I could not produce a validated
answer from the available evidence." and no model text, not a gateway error.
No semantic grading call is made for them.

The safe messages are constants in `src/services/agent/status.py` and
`src/routers/agent.py`. No exception text, prompt, or state is exposed.

## Endpoints

The Week 5/6 endpoints `POST /api/v1/ask` and `POST /api/v1/ask/stream` are
unchanged and are not routed through the agent.

### `POST /api/v1/agent/ask`

Request: `{"question": "..."}` (blank is rejected with 422; there are no
configuration overrides).

Response fields: `answer`, `outcome`, `sources`, `cache_status`
(`hit`/`miss`/`bypass`/`failure`), `terminal_reason`, `grounding_passed`,
`model`, `prompt_tokens`, `completion_tokens`, `execution`, `statuses`,
`pipeline_version`, `latency_ms`.

- `model` and token counts describe the final successful answer generation
  only. Grader, rewrite, and selector usage is in Langfuse, not summed here.
- `execution` is counts only: retrieval attempts, whether the query was
  rewritten, live fallback used, live papers selected, generation attempts,
  grounding attempts, source count.
- There is no `trace_id`: the observability abstraction does not expose one.

### Sources

Sources are exactly `generation_result.sources`: the labeled, budgeted
sources the final answer was generated from and grounded against, not all
retrieval hits or all final evidence. Each has `citation`, `source_type`
(`local` or `live_arxiv`), `arxiv_id`, `chunk_id`, `chunk_index`, `title`,
`section`, `content`, `truncated`, `authors`, `categories`, `published_date`,
and `source_url` (the PDF URL for live sources, `null` for local).

### `POST /api/v1/agent/ask/stream`

Server-sent events with JSON data. Prompt resolution and the cache lookup
happen before the stream opens, so the first event already carries the cache
status.

| Event | Data |
|---|---|
| `metadata` | `cache_status`, `pipeline_version` |
| `status` | `code`, `message` |
| `answer` | `text` (one event, the whole validated answer) |
| `sources` | `sources` |
| `outcome` | `outcome`, `message` (semantic terminal only) |
| `done` | outcome, terminal reason, grounding passed, cache status, model, tokens, citations, execution, latency |
| `error` | `code`, `message` |

Sequences:

```text
grounded MISS:      metadata, status*, answer, sources, done
HIT:                metadata, status (cache_hit), answer, sources, done
semantic terminal:  metadata, status*, outcome, done
execution failure:  metadata, status*, error
```

`done` and `error` are terminal; `error` is not followed by `done`, as in the
existing stream. While the graph is working, a comment frame (`: keepalive`)
is sent every 15 seconds so a long silent step does not look like a dead
connection.

**No answer before grounding.** Generation tokens are never streamed. The
single `answer` event is sent only after the graph has finished and the
answer has passed semantic grounding. If grounding is exhausted, no model
text is sent at all. After a regeneration, only the second, grounded answer
is sent.

**Cache hit.** One `status` event, "Validated cached answer found". No
guardrail, retrieval, or grounding statuses are replayed, because none of
that work happened.

### Execution statuses

Statuses come from the graph's safe execution events as each node finishes
and are converted to fixed text in one place (`status.py`). They are
operational states only. Because a node reports its start and end together,
a "started" status and its result arrive at the same moment.

| Event | Text |
|---|---|
| `guardrail_passed` / `guardrail_rejected` | Guardrail passed / Question is outside the supported research scope |
| `local_retrieval_completed` | Local retrieval completed / Local retry completed |
| `evidence_sufficient` / `evidence_insufficient` | Evidence sufficient / Evidence insufficient |
| `query_rewritten` | Query rewritten |
| `live_fallback_started` | Live arXiv fallback used |
| `live_fallback_completed` | Live arXiv search found N candidate papers |
| `live_selection_completed` | N live papers selected |
| `live_document_processing_completed` | Live document processing completed (N papers usable) |
| `evidence_rerank_completed` | Evidence reranked (N sources selected) |
| `generation_length_recovery_started` | Initial answer was incomplete; regenerating with a larger output limit |
| `generation_completed` | Answer generated / Answer regenerated |
| `grounding_passed` / `grounding_failed` | Grounding validation passed / Grounding validation failed |
| `graph_completed` / `graph_failed` | Completed / The request could not be completed |

No status contains a question, query, evidence, title, URL, prompt, score,
grader output, exception text, or reasoning. Tests serialise every status on
representative paths and assert this.

## Observability

Each request has one root observation, `agent.request`, with child spans
`agent.prompt_resolution`, `agent.cache.lookup`, and `agent.cache.write`, and
the graph's generation observations from W7-M09 (`agent.guardrail`,
`agent.evidence_grader`, `agent.query_rewrite`, `agent.live_selector`,
`rag.generation`, `rag.regeneration`, `agent.answer_grounding`) plus the
existing `rag.evidence` and `rag.grounding` spans.

A cache hit creates no generation observation; values copied from the cached
answer are recorded with an `artifact_` prefix. Root metadata is content-free:
cache outcome, outcome, terminal reason, grounding passed, attempt counts,
source count, model, token counts, pipeline version, latency. With
`LANGFUSE_CAPTURE_CONTENT=false` no question, evidence, prompt, or answer is
sent.

When a node fails, the API log records the stage and exception type names
only, for example `Agent answer generation failed (GroundingValidationError,
cause=None)`.

## Timeouts and bounds

| Bound | Default | Setting |
|---|---|---|
| Whole request (prompt resolution + graph) | 540 s | `AGENT_REQUEST_TIMEOUT_SECONDS` |
| One live paper (download + parse) | 240 s | `AGENT_LIVE_PAPER_TIMEOUT_SECONDS` |
| Live arXiv request | 15 s, 1 retry | `AGENT_LIVE_ARXIV_TIMEOUT_SECONDS`, `AGENT_LIVE_ARXIV_MAX_RETRIES` |
| Prompt bundle resolution | 5 s | `LANGFUSE_TIMEOUT_SECONDS` |
| Local attempts / live papers / chunks per paper / final sources | 2 / 2 / 8 / 5 | `AgentGraphConfig` |
| Completion tokens for each classifier call | 1024 | fixed |
| Answer completion tokens | 1024 | `LLM_MAX_COMPLETION_TOKENS` |
| Length-recovery completion tokens | min(2 x normal, 4096, context room) = 2048 | `LLM_LENGTH_RECOVERY_MAX_TOKENS` (the 4096 cap) |
| Length recoveries per generation | 1 | fixed |
| Concurrent Docling conversions per worker | 1 | fixed |

On the deadline the request ends as a controlled timeout (504, or an `error`
event) with no partial answer. A cache hit returns in milliseconds.

The request and per-paper defaults were raised from the first values (180 s
and 120 s) after measurement: one Docling conversion took 161 s on this
six-core, CPU-only host during the first live run, so a live fallback could
never have finished. Later conversions of other papers took 32-77 s.

A Docling conversion that is already running cannot be cancelled; after a
timeout it finishes in the background and its result is discarded.

## Completion finish reason and length recovery

The provider's finish reason is carried through the provider-neutral stack:
Groq `finish_reason` -> `LLMCompletion.finish_reason` ->
`RAGGenerationResult.finish_reason` -> the agent's generation node. The Groq
adapter lower-cases it and passes unknown values through.

| Finish reason | Meaning | Handling |
|---|---|---|
| `stop` | complete | structural validation, then semantic grounding |
| `length` | cut off at the completion-token limit | incomplete: one length recovery |
| missing, or any other value | not reported as cut off | treated as complete, as before |
| empty content with `length` | the model used the whole allowance before writing | incomplete: one length recovery |

Punctuation is never used to guess. Missing metadata is not a failure, so
providers and fakes that report nothing behave as they did.

**Incomplete output is discarded inside the generation call.** It is not
structurally validated, not graded, not stored in agent state, not returned,
not streamed, and not cached.

**Length recovery** repeats that one generation from the beginning with the
same question, evidence context, source labels, order, and content, and the
same prompt. Nothing is retrieved, rewritten, searched, or reranked again,
and the partial text is neither appended to nor sent back to the model.

**Budget.** The normal allowance stays `LLM_MAX_COMPLETION_TOKENS` (1024); it
was not raised, because the 8192-token window must also hold up to 6000
tokens of evidence. The recovery allowance is

```text
min(2 x LLM_MAX_COMPLETION_TOKENS, LLM_LENGTH_RECOVERY_MAX_TOKENS, context window - safety margin - prompt)
```

which is 2048 with the defaults (cap 4096). If that is not larger than the
normal allowance, no recovery call is made.

**Cache identity.** Recovery configuration decides whether a cut-off answer
can be redone, so it is answer-affecting and part of the agent cache
identity: `length_recovery_max_tokens` (the same
`LLM_LENGTH_RECOVERY_MAX_TOKENS` setting the generation service reads) and
`length_recovery_multiplier` (the fixed 2x rule, taken from the constant the
generation service uses). Changing either changes every agent cache key. The
namespace stays `rag:agent-response:v1`. A change to recovery behaviour that
these two values do not express needs a new identity field or an
`AGENT_PIPELINE_VERSION` bump.

**Bounds.** One recovery per generation. If the recovery is cut off too, the
run ends as `generation_failed` (`generation_incomplete`); there is no third
call. Grounding regeneration is separate and unchanged: a complete, cited
answer that fails semantic grounding is regenerated once. The regeneration
is itself a generation, so it may use one length recovery of its own. A
recovery is not a grounding attempt and does not change
`grounding_attempts`. The worst case is four answer-generation calls and two
grounding calls.

**Telemetry.** Every call is its own Generation observation. A recovery has
`generation_reason="length_recovery"`, the same `generation_attempt`, its
actual `max_completion_tokens`, and the same prompt fingerprint. The finish
reason is recorded when reported. No answer text is recorded. Public token
counts are those of the final successful generation only.

The older `/ask` and `/ask/stream` endpoints do not opt in and behave as
before.

## Gradio

```text
Browser -> Gradio (src.gradio_app) -> FastAPI /api/v1/agent/ask/stream -> agent
```

Gradio holds no agent dependencies. `AgentStreamClient` posts to the agent
stream endpoint and parses the events above; the older `RAGStreamClient` and
the `delta` protocol still work.

The UI shows the validated answer, the sources with a `LOCAL` or
`LIVE ARXIV` badge, arXiv ID, section, and a PDF link for live sources, an
execution path of the status messages in order, `Cache: HIT/MISS`, the
response time measured in the client, and the model and token counts. The
answer area stays empty until the `answer` event arrives. Semantic outcomes
show their safe message with no sources; errors show a fixed safe message and
never a stack trace or raw payload. Status text and source fields are
HTML-escaped, and only `https://arxiv.org/` links are rendered.

The default API base URL is now `http://127.0.0.1:8000`, the Docker API.

## Live acceptance

Run on 2026-10-10 against the real stack: Docker API (rebuilt from this
worktree), OpenSearch (17 papers), Redis, Groq `openai/gpt-oss-120b`, the
local BGE model, Langfuse, arXiv, and Docling. Prompt management was off, so
the local prompt bundle was used. Requests went through the SSE endpoint.

| # | Question | Cache | Guardrail | Local attempts | Rewrite | Live fallback | PDFs | Final sources | Generation attempts | Grounding attempts | Terminal | Latency | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | What is a dog? | miss | rejected | 0 | no | no | 0 | 0 | 0 | 0 | `out_of_scope` | 0.69 s | controlled out-of-scope message; not cached |
| B | What biases were found when auditing open-source large language models for clinical triage? | miss | passed | 1 | no | no | 0 | 5 | 1 | 1 | none | 79.1 s | grounded answer, 5 local sources, cached |
| B (repeat) | same | **hit** | - | - | - | - | - | 5 | - | - | none | 3.2 ms | identical cached answer |
| C | Tell me about ML stuff | miss | passed | 2 | yes (once) | no | 0 | 5 | 1 | 1 | none | 79.9 s | grounded answer after the rewrite, 5 local sources, cached |
| D | What are the current Healthcare issues in India and how can AI help us solve them | miss | passed | 2 | yes (once) | yes, 5 candidates | 1 selected, 1 usable | 5 (4 local, 1 live) | 1 | 1 | none | 198.4 s | grounded answer from merged local and live evidence, cached |
| D (repeat) | same | **hit** | - | - | - | - | - | 5 | - | - | none | 4.9 ms | identical cached answer |

Notes:

- **A** was repeated after the other runs and was a miss again, so it is not
  cached. An earlier repeat, sent seconds after B, ended as a controlled
  `internal_error`: the guardrail call exhausted its Groq rate-limit retries.
  No detail reached the client.
- **B** cites `[S1]`, `[S2]`, `[S4]`; all five sources are `local`, from paper
  2610.01963. The miss includes the first BGE load in the new container (about
  22 s) and a 42 s rate-limit wait on the grounding call.
- **C** stayed within two local attempts and one rewrite. Evidence was
  insufficient, the query was rewritten, the retry was sufficient, and no live
  fallback was needed. The answer cites `[S1]`-`[S5]`, all `local`. Its
  generation used exactly the 1024-token completion limit and the text stops
  mid-sentence. This run predates the finish-reason handling below, which now
  redoes such an answer instead of accepting it.
- **D** ran the whole live path: two local attempts, one rewrite, live arXiv
  search (5 candidates), 1 paper selected (2310.18361), downloaded and parsed
  by Docling in 153 s, then 5 sources reranked from local and live chunks
  together. The answer cites `[S1]`, `[S3]`, `[S5]` (local). The live chunk is
  `[S4]`, with `source_type=live_arxiv` and its arXiv PDF URL; it was supplied
  to the model and grounded against, but not cited.
- In every answered run each cited label exists in the returned sources and
  `grounding_passed` is true.
- The agent namespace held 4 keys afterwards (B, C, D, and one answer from the
  2026-10-09 run). The Week 6 namespace was not written by these requests.

On 2026-10-09, D twice ended as a 502 because the model returned an answer
with no citation. That case is still `generation_failed` internally, but is
now a 200 controlled outcome publicly. It did not recur in this run, so that
mapping is covered by tests, not by a live run.

The finish-reason handling was added after these runs and the scenarios were
not repeated. Two small direct Groq calls confirmed the provider's values:
partial text came back with `finish_reason="length"`, and a very small limit
came back with empty content at the limit, reported as incomplete.

### Cache latency

| Question | MISS | HIT | Ratio |
|---|---|---|---|
| B | 79.1 s API (79.1 s client) | 3.2 ms API (53 ms client) | about 25,000x by API latency; about 1,500x by client wall time |
| D | 198.4 s API (198.4 s client) | 4.9 ms API (54 ms client) | about 40,385x by API latency; about 3,700x by client wall time |

Miss latency is dominated by Groq rate-limit waits and, for D, one Docling
conversion, so these ratios describe this host and plan, not the design.

### Langfuse

With Langfuse enabled and content capture off, the observations were read
back through the public API (`/api/public/v2/observations`; the older traces
endpoint is not available to this organization):

- one root `agent.request` per request, with safe metadata (cache outcome,
  outcome, or the failure classification);
- `GENERATION` observations with model, input and output tokens, latency, and
  prompt name, version, and fingerprint: 1 for A (`agent.guardrail`), 4 for B,
  6 for C (two evidence gradings and a rewrite), 7 for D (adding
  `agent.live_selector`);
- child spans `agent.prompt_resolution`, `agent.cache.lookup`,
  `agent.cache.write`, `rag.evidence`, `rag.grounding`;
- both cache hits had 3 observations and no generation;
- `input` and `output` were empty on every observation;
- cost was empty, as no token prices are configured.

`rag.regeneration` was not observed live because no run needed a
regeneration.

### Gradio

Gradio was started on the host against the Docker API
(`RAG_API_BASE_URL=http://127.0.0.1:8000`, listening on `127.0.0.1:7860`).
The page loads, and the submit handler was called through the running Gradio
server:

- D (cached): `Cache: HIT`, one "Validated cached answer found" step, the
  answer, four `LOCAL` sources and one `LIVE ARXIV` source with its section
  and PDF link, response time 0.09 s, and the original model and token counts.
- A: the safe out-of-scope message, no sources, `Cache: MISS`, and its
  four-step path.
- The answer area held only the placeholder until the `answer` event.

This exercised the real server and handler, not a browser. A visual check in
a browser has not been done.

## Known risks and follow-ups

- **A truncated answer cached before this fix is still in Redis.** Scenario C's
  entry was written before finish reasons were checked and expires with its
  24-hour TTL. Delete it, or wait, before a demo.
- **A recovery request is larger.** The recovery call asks for up to twice the
  normal allowance, which counts against Groq's tokens-per-minute limit and
  can add a rate-limit wait.
- **Rate-limited calls surface as `internal_error`.** A guardrail or grader
  call that exhausts its Groq retries ends the request with a generic 500 or
  `error` event; a 503 with a retry hint would describe it better.
- **Reasoning-model token caps.** The configured model spends completion
  tokens on reasoning before output. The classifier calls originally capped
  completions at 32 tokens and returned empty content; the cap is now 1024.
- **Groq rate limits.** Free-tier limits caused retry waits of up to 29 s on
  single calls, which dominates local-path latency.
- **Live fallback is slow.** Two Docling conversions take roughly two to five
  minutes on this host. Fewer papers, a page limit, or a faster parser would
  shorten it.
- **Docling threads cannot be cancelled** and keep using CPU after a timeout.
- **Prompt SDK threads** can outlive a bundle timeout by up to the SDK fetch
  timeout.
- **VPS resources.** The API container used up to about 3.3 GB with BGE and
  Docling loaded; the host has 12 GB and no swap.
- **Cold start.** The first request after a container start loads BGE, and
  the first live paper downloads Docling's layout model.
- **Statuses arrive per node**, so "started" and "completed" appear together.
- **Agent cache statistics are not in `/health`.** The health endpoint reports
  the Week 6 coordinator only; the agent cache keeps its own counters.
- **Tests must run with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`.**
  Without them the suite uses the services in `.env`: it wrote fake-model
  entries to `rag:response:v1` and traces to Langfuse.
- **One worker.** The compiled-graph cache, the Docling lock, and cache
  statistics are per process.

## Next

- M11: Dockerize Gradio and add a production Compose file.
- M12: Nginx, DNS/domain, and HTTPS.
- M13: systemd autostart, resource hardening, and a reboot test.
- M14: public recruiter acceptance and final architecture/README polish.

## Verification

- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 2090 passed
  (W7-M09 baseline: 1802), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Unit and API tests use fakes; the live acceptance above used the real services.
