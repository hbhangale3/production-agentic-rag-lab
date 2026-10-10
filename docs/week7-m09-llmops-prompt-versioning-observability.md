# Week 7 M09: prompt versioning, generation telemetry, and agent cache identity

W7-M09 gives every LLM-driven stage a stable prompt identity, lets those
prompts optionally be managed in Langfuse with the local text as the
fallback, records each model call as a true generation observation with
provider-reported usage, measured latency, and known cost, and defines the
cache key contract for agent responses.

It does not change agent routing, add a grader, read or write a cache, or
expose an API. W7-M10 does the runtime integration.

**Prompt identity fingerprints template content, not runtime user data.**

## Prompt identity

A prompt identity is four values:

| Field | Meaning |
|---|---|
| `name` | stable catalog name |
| `version` | `local-v1` for a local template, `langfuse-v<N>` for a managed one |
| `label` | deployment label used to resolve it; `production` by default |
| `fingerprint` | SHA-256 of the canonical template text |

Canonicalization converts CRLF and CR to LF and strips outer whitespace;
nothing else is altered. The fingerprint is computed here, from the effective
content, for both local and managed prompts. Local versions are
human-controlled constants, never a git SHA.

The question, evidence, sources, candidate abstracts, and a previous answer
are runtime data. They are placed in the user message and never enter a
prompt identity. A test renders every prompt with private sentinels and
asserts each identity is unchanged.

### Catalog

| Stage | Name | Local version | Template source |
|---|---|---|---|
| Guardrail | `agent-guardrail` | `local-v1` | `GuardrailPromptBuilder.SYSTEM_PROMPT` |
| Evidence grader | `agent-evidence-grader` | `local-v1` | `EvidenceGraderPromptBuilder.SYSTEM_PROMPT` |
| Query rewrite | `agent-query-rewrite` | `local-v1` | `QueryRewritePromptBuilder.SYSTEM_PROMPT` |
| Live paper selector | `agent-live-paper-selector` | `local-v1` | `LivePaperSelectionPromptBuilder.SYSTEM_PROMPT` |
| Answer generation | `rag-answer-generation` | `local-v1` | `RAG_SYSTEM_PROMPT` |
| Answer regeneration | `rag-answer-regeneration` | `local-v1` | `RAG_REGENERATION_INSTRUCTION` |
| Answer grounding | `agent-answer-grounding` | `local-v1` | `AnswerGroundingPromptBuilder.SYSTEM_PROMPT` |

A regeneration renders the generation system prompt plus the regeneration
instruction. The instruction is its own catalog entry, so both identities are
tracked: the regeneration observation carries the instruction's identity and
the generation prompt's as `base_prompt_*`, and the cache key includes both.

The five classifier prompts each declare a required output-contract marker
(`{"score"`, `{"query"`, or `"selected_arxiv_ids"`). A managed prompt that
drops its marker is rejected, so a prompt edit cannot silently break the
strict parsers.

## Prompt resolution

```text
PromptResolver.resolve(definition, *, label) -> ResolvedPrompt
```

`ResolvedPrompt` holds `name`, `content`, `version`, `label`, `source`
(`local` or `langfuse`), and derives `fingerprint` and `identity`. No SDK
object is stored.

| Resolver | Behaviour |
|---|---|
| `LocalPromptResolver` | always the local template |
| `LangfusePromptResolver` | managed text prompt by name and label, local template on any problem |

The Langfuse resolver lives in the observability adapter package and is the
only place that calls the SDK: `client.get_prompt(name, label=..., type="text",
cache_ttl_seconds=..., max_retries=0, fetch_timeout_seconds=...)`. Prompt
builders never import Langfuse.

Local prompts stay authoritative. The local template is used when:

- prompt management is disabled (the default) or Langfuse is disabled;
- the fetch raises or exceeds the timeout;
- the result is blank, not a text prompt, an SDK fallback, or has no integer
  version;
- the managed text is missing a required contract marker.

None of these fails a request. A warning with the exception type, and no SDK
detail, is logged.

Latency is bounded: one fetch attempt, no retries, the existing
`LANGFUSE_TIMEOUT_SECONDS` as both the SDK fetch timeout and an outer
`asyncio` deadline, and the fetch runs off the event loop. The SDK caches
fetched prompts in process for `LANGFUSE_PROMPT_CACHE_TTL_SECONDS` (default
60), so most resolutions make no request and a managed edit takes effect
within that window.

Settings (all optional, added to `.env.example`):

| Setting | Default |
|---|---|
| `LANGFUSE_PROMPT_MANAGEMENT_ENABLED` | `false` |
| `LANGFUSE_PROMPT_LABEL` | `production` |
| `LANGFUSE_PROMPT_CACHE_TTL_SECONDS` | `60` |

`make_prompt_resolver(settings, observability_provider)` returns the Langfuse
resolver only when Langfuse is enabled, prompt management is enabled, and the
provider is configured; it reuses the provider's client.

## Prompt bundle

`AgentPromptBundle` holds the seven resolved prompts; its `identities`
property is an `AgentPromptIdentityBundle` of seven `PromptIdentity` values.

```python
bundle = await resolve_agent_prompt_bundle(resolver, label=settings.langfuse_prompt_label)
graph = build_agent_graph(dependencies=deps, config=config, prompts=bundle)
```

`build_agent_graph` constructs every prompt builder from the bundle,
including a per-bundle copy of the generation service, so the prompts that
run are exactly the bundle's. With no bundle, the local bundle is used, which
needs no I/O. The start node writes `prompt_identities` (identities only, no
template text) into `AgentState`.

Resolution involves no retrieval and no LLM call, so it can run before a
cache lookup.

### Concurrent, bounded resolution

`resolve_agent_prompt_bundle(resolver, *, label, timeout_seconds=5.0)`
resolves the seven prompts concurrently, as separate tasks, under one
deadline for the whole bundle. Production callers should pass
`LANGFUSE_TIMEOUT_SECONDS` as `timeout_seconds`.

- Each prompt still fails open on its own: a resolver error, a wrong or
  malformed result, or a missing contract marker gives that prompt its local
  fallback and leaves the others untouched.
- When the deadline elapses, prompts that have not resolved are cancelled and
  use their local fallback. Prompts that already resolved keep their result.
  The bundle is therefore a deterministic function of which prompts finished
  in time, and its identities always describe its contents.
- No exception from a prompt problem escapes. If the caller itself is
  cancelled, outstanding resolutions are cancelled too.

The worst case with an unresponsive Langfuse is one timeout window (about 5
seconds by default), not seven. A warning records how many prompts were
unresolved at the deadline, and nothing else.

## Generation observations

The provider-neutral `Observation` protocol gained:

- `start_generation(name, model=None, metadata=None)` - a child observation
  representing one model call;
- `record_generation(model, input_tokens, output_tokens, cost)`.

The Langfuse adapter maps `start_generation` to
`start_observation(as_type="generation", ...)` and `record_generation` to
`update(model=..., usage_details={"input", "output", "total"},
cost_details={"input", "output", "total"})`. The no-op and fail-open
wrappers implement both. An observation object that has no generation
support is recorded as a span.

`observed_completion` wraps one `LLMProvider.complete` call with a generation
observation. No Langfuse import exists in RAG or agent code.

### What is instrumented

| Observation name | Type | Call |
|---|---|---|
| `agent.guardrail` | generation | guardrail |
| `agent.evidence_grader` | generation | evidence grading, each attempt |
| `agent.query_rewrite` | generation | query rewrite |
| `agent.live_selector` | generation | live paper selection |
| `rag.generation` | generation | answer generation, including Week 6 `/ask` and `/ask/stream` |
| `rag.regeneration` | generation | regeneration, a separate observation |
| `agent.answer_grounding` | generation | answer grounding, each attempt |

`rag.generation` keeps its Week 6 name; only its type changed from span to
generation, which is what lets Langfuse model and latency views populate. The
other Week 6 observations (`rag.request`, `rag.cache.lookup`,
`rag.retrieval`, `rag.evidence`, `rag.grounding`, `rag.cache.write`) are
unchanged, and a cache hit still creates no generation.

The agent receives its request observation at invocation:

```python
await graph.ainvoke(state, config={"configurable": {AGENT_OBSERVATION_KEY: observation}})
```

Without one, no telemetry work is done.

### Recorded for each generation

| Data | Source |
|---|---|
| model | the provider's completion |
| provider | configured `llm_provider` |
| input / output / total tokens | provider-reported counts; never estimated |
| `latency_ms` | monotonic clock around the provider call |
| cost | only when known (below) |
| prompt identity | `prompt_name`, `prompt_version`, `prompt_label`, `prompt_fingerprint` |
| attempt | `generation_attempt`, `grounding_attempt`, or `retrieval_attempt` |
| failure | exception type name only |

For streaming, usage is whatever the stream reports.

### Cost

Cost is recorded only from explicitly configured prices:
`LLM_INPUT_COST_PER_MILLION_TOKENS` and `LLM_OUTPUT_COST_PER_MILLION_TOKENS`.
Both must be set and the provider must report both token counts; otherwise no
cost is sent and `cost_usd` is absent. No price is built in. Independently,
Langfuse may compute cost itself for models it has pricing for, from the
model and usage this milestone now reports.

### Privacy

Generation observations carry metadata only. No `input` or `output` is set,
so no question, evidence, rendered prompt, previous answer, or answer is sent,
whatever `LANGFUSE_CAPTURE_CONTENT` is. Prompt identity is template metadata
and is always safe to send. This milestone adds no content capture.

### Fail-open

Every observation call is wrapped. If creating, updating, recording, or
ending an observation fails, or the clock fails, the model call and its
result are unaffected. Tests run the agent with broken observations and
compare results with an unobserved run.

## Agent pipeline version

`AGENT_PIPELINE_VERSION = "v1"`. Bump it when agent routing or semantics
change materially. It is part of the cache identity.

## Agent cache identity

A new contract, separate from Week 6:

```text
rag:agent-response:v1:<sha256 of canonical identity JSON>
```

The Week 6 namespace `rag:response:v1` is untouched and is never used for
agent answers.

### Included (answer-affecting)

- question, whitespace-trimmed and collapsed, case preserved;
- corpus fingerprint;
- retrieval: version, size, RRF k, candidate multiplier, max results, chunk
  index name;
- embedding model and dimension;
- evidence: version, context token budget, chunk target/overlap/minimum
  words (these shape live chunks);
- LLM provider, model, temperature, max completion tokens, structural
  grounding version;
- every `AgentGraphConfig` field: guardrail threshold, evidence threshold,
  max local attempts, live fallback enabled, live max results, PDF max
  papers, chunks per paper, final evidence max sources, grounding threshold,
  max grounding attempts;
- agent pipeline version and cache schema version;
- all seven prompt identities: name, version, label, fingerprint.

Both version and fingerprint are included, so a deliberate version bump
invalidates the cache even if the text is identical, and a text change
invalidates it even if the version was not bumped.

### Excluded (operational only)

Langfuse host, keys, timeout, capture-content, prompt-management toggle and
cache TTL; token prices; Redis settings and cache TTL; LLM and arXiv
timeouts and retry counts; embedding device and batch size; environment;
timestamps, request IDs, trace IDs, and latency. Changing any of these does
not change the key. The prompt-management toggle is excluded on purpose: it
only matters through the prompt identities it produces.

### Determinism and privacy

The key is SHA-256 over JSON with sorted keys and compact separators. A
known-vector test pins the key for a fixed identity. The raw question never
appears in the key; the canonical payload exists only in memory.

### Not wired

Nothing reads or writes Redis for the agent, and the graph has no cache node.

## W7-M10 boundary

```text
request arrives
  -> bundle = resolve_agent_prompt_bundle(resolver, label)
  -> key = AgentCacheKeyBuilder.build(factory.create(question, corpus_fingerprint, prompts=bundle.identities))
  -> cache lookup
       hit  -> return the cached response; the graph does not run
       miss -> build_agent_graph(..., prompts=bundle) and run it with the request observation
               -> cache only a successful, grounded result under the same key
```

Because the key and the graph are built from the same bundle object, a
managed prompt changing between two requests cannot produce a key for one
prompt and an answer from another. `state["prompt_identities"]` lets M10
verify this after a run.

## Deferred

Agent cache read/write, agentic HTTP and SSE, Gradio integration, production
wiring of graph dependencies and the prompt resolver, an overall request
deadline, and deployment work.

## Verification

- Agent tests: 993 passed (858 after W7-M08).
- Prompt, observability, cache, and RAG tests: 299 passed.
- Focused prompts/embeddings/arXiv/PDF/chunking/LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle/config regression: 662 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 1802 passed
  (W7-M08 baseline: 1566), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fake LLM, embedding, observation, and Langfuse clients; no network request is made.
