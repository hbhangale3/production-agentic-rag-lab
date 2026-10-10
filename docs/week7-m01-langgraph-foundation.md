# Week 7 M01: LangGraph foundation

W7-M01 adds the typed orchestration contracts needed by later agentic RAG
milestones without changing the production Week 6 request path. LangGraph is
installed and a real graph compiles and runs, but the graph deliberately makes
no retrieval, model, cache, database, observability, or network calls.

## Architecture boundary

The new `src/services/agent` package separates four concerns:

- `state.py` defines provider-neutral data passed between graph nodes.
- `config.py` validates bounded graph behavior.
- `events.py` defines content-free execution transitions and allowlisted metadata.
- `graph.py` owns topology, compilation, and the future dependency-injection boundary.

Terminal outcomes and bounded failure categories live in `types.py`. Service
clients are dependencies and never belong in `AgentState`. The empty,
immutable `AgentGraphDependencies` boundary is intentional in this foundation;
future milestones can add narrowly typed worker-scoped services without nodes
constructing infrastructure.

## Agent state

`AgentState` is a LangGraph-compatible `TypedDict`. It reserves explicit fields
for request identity, guardrails, local retrieval, evidence grading, rewriting,
live fallback, selected generation evidence, generation, grounding, terminal
outcomes, and safe execution history. It reuses `HybridSearchResult` and
`EvidenceSource`, which are existing application-owned domain types.

`None` means a future stage has not executed. This remains distinct from a
completed false result, an empty evidence tuple, or counters initialized to
zero. Live fallback starts as `False`; future evidence values remain `None`.
`create_initial_agent_state` rejects blank questions, preserves the caller's
question exactly, initializes `current_query` consistently, and creates a new
event list for every state.

Most node updates replace prior values. `execution_events` alone uses an
explicit additive reducer, so events accumulate deterministically in graph
order rather than being overwritten.

## Graph configuration

`AgentGraphConfig` is immutable, rejects unknown fields, and defaults to:

- guardrail threshold: 60 on a future 0–100 scale
- local retrieval attempts: 2 total
- retrieval size: 5
- grounding attempts: 2 total
- live fallback enabled: true

Two local retrieval attempts means **one initial attempt plus at most one
rewritten-query retry**, not two retries. Thresholds outside 0–100 and attempt
or size values below one are rejected. No loop is implemented in M01.

## Safe execution contract

`AgentExecutionStatus` constrains factual transitions from `graph_started`
through `graph_completed` or `graph_failed`, including the statuses future
guardrail, retrieval, grading, rewrite, fallback, generation, and grounding
nodes will need. Events contain only a sequence and allowlisted operational
metadata such as attempt/source counts and pass/fail values.

The contract has no timestamps, prompts, questions, evidence, answers,
scratchpads, arbitrary metadata, or model rationale. It is an execution-status
contract, not chain-of-thought, and is suitable for later API, SSE, Gradio, and
Langfuse metadata use.

`TerminalReason` distinguishes successful completion, out-of-scope input,
insufficient evidence, grounding failure, retrieval failure, generation
failure, and internal error. Expected domain outcomes such as insufficient
evidence are not conflated with provider or infrastructure failures.

## Minimal topology

The compiled M01 graph is deliberately small:

```text
START -> initialize_foundation -> END
```

The foundation node preserves input state, appends ordered `graph_started` and
`graph_completed` events, and sets the terminal reason to `completed`. It does
not instantiate or call any external service.

## Deferred work

M01 does not implement guardrail grading, retrieval nodes, semantic evidence
grading, query rewriting, retrieval retry routing, live arXiv search, PDF
acquisition or parsing, evidence reranking, answer generation, semantic
grounding, prompt management, cost accounting, prompt-aware cache identity,
an agentic API/SSE contract, or Gradio integration. Existing `/ask`,
`/ask/stream`, caching, observability, and application lifecycle are unchanged.
Telegram remains out of scope.

## Verification

- Agent foundation tests: 24 passed.
- Focused agent/API/cache/observability/lifecycle regression: 188 passed.
- Full isolated suite: 685 passed with 10 existing warnings.
- Tests ran with Redis and Langfuse disabled and the invalid host-level `DEBUG`
  variable removed from the test process; the user's `.env` was not modified.
