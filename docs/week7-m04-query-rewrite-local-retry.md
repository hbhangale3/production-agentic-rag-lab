# Week 7 M04: query rewrite and one bounded local retry

When the first local retrieval is graded insufficient, W7-M04 rewrites the
retrieval query once, retrieves locally a second time, and grades that second
result against the original question. Then the graph ends. There is no live
arXiv fallback, no PDF processing, and no answer generation.

## Topology

```text
START
  -> graph_start
  -> guardrail
       rejected -> out_of_scope -> END
       failed   -----------------> END
       passed   -> local_retrieval <--------------------+
                     failed    --> END                  |
                     retrieved -> evidence_grading      |
                                    sufficient -> graph_complete -> END
                                    rewrite    -> query_rewrite
                                    |               rewritten --+
                                    |               failed   --> END
                                    exhausted  -> graph_complete -> END
                                    failed     ----------------> END
```

The same `local_retrieval` and `evidence_grading` nodes serve every attempt.
After an insufficient grade the router chooses:

```text
retrieval_attempts < max_local_retrieval_attempts  -> rewrite
otherwise                                          -> exhausted
```

## Attempt semantics

`retrieval_attempts` is the number of local `HybridSearchService.search` calls
attempted. It is 0 initially, 1 when the first search is made, 2 when the
second is made. A search that raises still counts.

`AgentGraphConfig.max_local_retrieval_attempts` is the **total** number of
local attempts. The default is 2:

```text
maximum local retrieval attempts = 2
  = 1 initial retrieval + 1 rewritten retry
  NOT 1 initial retrieval + 2 retries
```

With `max_local_retrieval_attempts=1` an insufficient first grade completes
the graph normally with no rewrite and no retry. No retry-count setting was
added.

The bound is enforced twice. The router never selects `rewrite` once the total
is reached, and the retrieval node itself refuses to search when
`retrieval_attempts >= max_local_retrieval_attempts`: it does not call the
search service, leaves the count unchanged, and ends with `graph_failed`,
`terminal_reason=internal_error`, `error_category=internal_failure`. Normal
routing cannot reach that branch.

## Query rewriter

| Component | Responsibility |
|---|---|
| `QueryRewritePromptBuilder` | Local replaceable prompt; isolates untrusted inputs |
| `QueryRewriter` | Calls `LLMProvider.complete`, parses, normalizes, validates |
| `QueryRewriteResult` | One field, `query` |
| `QueryRewriteError` | Any provider, parsing, or validation failure |

The rewriter reuses the injected `LLMProvider` (temperature 0.0,
`max_tokens=128`). It receives only `original_question` and `current_query`;
no retrieved evidence and no grade are sent. The prompt asks for one improved
scientific-paper retrieval query that preserves the topic, every part of a
multi-part question, and named constraints such as a country, population,
condition, or technology, and that does not answer, explain, or cite.

Both inputs travel in one JSON-encoded user message under an
`UNTRUSTED_REWRITE_INPUT_JSON` header and never appear in the system message,
which says to ignore instructions inside the payload. Tests verify this
construction; they do not prove real-model injection resistance, and nothing
verifies locally that a rewrite preserved the user's intent.

## original_question / current_query / rewritten_query

| Field | Meaning | After a successful rewrite |
|---|---|---|
| `original_question` | Immutable user information need | unchanged |
| `current_query` | Query the next retrieval uses | the rewrite |
| `rewritten_query` | The validated rewrite | the rewrite |

Evidence grading always evaluates `original_question`, on every attempt.

## Rewrite contract

The model must return exactly `{"query": "<string>"}`, optionally inside one
Markdown code fence: the same policy as the score parser, sharing its helper
in `structured_output.py`. Missing or extra fields, non-string values, lists,
and prose around the JSON are rejected.

The string is then normalized (trimmed, internal whitespace collapsed) and
must be non-empty and at most 300 characters. `HybridSearchService` only
requires a non-empty query, so the limit is the rewriter's own.

A rewrite that equals `current_query` after normalization, ignoring case, is
unusable: retrying the same query would waste the second attempt. It is a
`QueryRewriteError`. The graph does not ask again and does not invent a query.

## Rewrite failure

Provider failure, malformed output, an empty, overlong, or identical rewrite
all end the graph with `graph_failed`, `terminal_reason=internal_error`,
`error_category=query_rewrite_failure`. `retrieval_attempts` stays 1,
`current_query` and `rewritten_query` are unchanged, and the first attempt's
result and insufficient grade remain in state because they are still true.
This is not insufficient evidence, a retrieval failure, or out-of-scope.

## Stale-grade clearing

A successful rewrite clears `local_retrieval_result`, `evidence_grade`, and
`evidence_sufficient` in the same update that sets the new `current_query`,
because they describe the previous query. The retrieval node also resets the
grade fields whenever it stores a new result or fails. So between the second
retrieval and the second grade the state shows the new result with no grade,
and a second-attempt retrieval or grading failure never leaves the first
attempt's evidence or grade looking current. The first grade survives only in
the execution events.

## Second attempt outcomes

| Outcome | State |
|---|---|
| Sufficient | `retrieval_attempts=2`, `evidence_sufficient=True`, attempt-2 grade and result, `terminal_reason=None` |
| Insufficient | same with `evidence_sufficient=False`; `graph_completed`; `terminal_reason=None` |
| Retrieval raises | `retrieval_attempts=2`, `retrieval_failed` / `retrieval_failure`, no result, no grade |
| Context build or grader fails | `retrieval_attempts=2`, `internal_error` / `evidence_grading_failure`, no grade |

Second-attempt insufficiency is deliberately not
`terminal_reason=insufficient_evidence`: in the final design it leads to live
fallback.

## Events

Retry then sufficient (or insufficient):

```text
graph_started
guardrail_started
guardrail_passed
local_retrieval_started        attempt 1
local_retrieval_completed      attempt 1
evidence_grading_started       attempt 1
evidence_insufficient          attempt 1
query_rewrite_started          attempt 1 -> next 2
query_rewritten                attempt 1 -> next 2
local_retrieval_started        attempt 2
local_retrieval_completed      attempt 2
evidence_grading_started       attempt 2
evidence_sufficient | evidence_insufficient
graph_completed
```

Failure tails:

```text
rewrite failure:           ... evidence_insufficient, query_rewrite_started, graph_failed
second retrieval failure:  ... query_rewritten, local_retrieval_started, graph_failed
second grading failure:    ... local_retrieval_completed, evidence_grading_started, graph_failed
```

The first-attempt-sufficient, out-of-scope, guardrail-failure, and
first-retrieval-failure sequences are unchanged from W7-M03.

`query_rewritten` is the completion status reserved in W7-M01. The reserved
`local_retry_started` status is not emitted: the reused retrieval node emits
`local_retrieval_started` with `retrieval_attempt=2`.

One metadata field was added, `next_retrieval_attempt`. Rewrite events carry
only `retrieval_attempt` and `next_retrieval_attempt`. No question, query,
rewritten query, evidence, prompt, completion, exception text, or reasoning is
placed in events. The model is never asked to explain a rewrite, and only the
query itself is stored in state.

## W7-M05 insertion point

"Local retrieval exhausted and still insufficient" is the `exhausted` route:

```text
evidence_sufficient is False
and retrieval_attempts >= max_local_retrieval_attempts
and error_category is None
```

W7-M05 changes one routing entry, `"exhausted": "graph_complete"`, to point at
the live-search node.

## Deferred

Live arXiv, PDF/Docling, transient evidence, merge/rerank, generation, answer
grounding, prompt management, cost tracking, cache integration, observability
changes, agentic HTTP/SSE, and Gradio integration.

## Verification

- Agent tests: 329 passed (220 after W7-M03).
- Focused LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle regression: 390 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 990 passed
  (W7-M03 baseline: 881), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fake providers and retrievers; no external service is called.
