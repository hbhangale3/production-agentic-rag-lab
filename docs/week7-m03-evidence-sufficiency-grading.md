# Week 7 M03: semantic evidence-sufficiency grading

W7-M03 adds a semantic grader between local retrieval and the end of the graph.
It decides whether the locally retrieved candidates are enough to support a
grounded answer to the user's actual question. It does not rewrite queries,
retry retrieval, search live sources, or generate an answer.

## Retrieved != relevant enough != sufficient

A retrieval system always returns its best available candidates. That says
nothing about whether those candidates answer the question. In Week 6 the
question "What are the current healthcare issues in India and how can AI help
us solve them?" retrieved five sources, none India-specific enough, and
generation correctly declined. Five sources were retrieved; the evidence was
not sufficient. The grader makes that judgement explicit before generation,
so later milestones can react to it. Nothing in the grader is specific to
that question.

## Topology

```text
START
  -> graph_start
  -> guardrail
       rejected -> out_of_scope -> END
       failed   -----------------> END
       passed   -> local_retrieval
                     failed    --> END
                     retrieved -> evidence_grading
                                    sufficient   -> graph_complete -> END
                                    insufficient -> graph_complete -> END
                                    failed       ----------------> END
```

`local_retrieval` no longer emits `graph_completed`; the dedicated
`graph_complete` node does. Both semantic outcomes are separate conditional
routes that currently share that node.

## Grader architecture

| Component | Responsibility |
|---|---|
| `EvidenceContextBuilder` (existing, injected) | `HybridSearchResult` -> ordered, deduplicated, token-bounded `EvidenceContext` |
| `EvidenceGraderPromptBuilder` | Local replaceable prompt; isolates untrusted inputs |
| `EvidenceSufficiencyGrader` | Calls `LLMProvider.complete`, parses the score, derives the boolean |
| `EvidenceGrade` (existing, extended) | `score`, `sufficient`, `source_count`, `grader_version` |
| `structured_output.parse_score_object` | Strict `{"score": <0-100>}` parser shared with the guardrail |

`AgentGraphDependencies` gains `evidence_context_builder`. The grader reuses
the same injected `LLMProvider` as the guardrail; no model name or Groq type
appears in agent code. The call uses temperature 0.0 and `max_tokens=32`.

`EvidenceGrade` gained one required field, `sufficient`, so the stored grade
carries the decision made with the threshold in force at grading time.
`AgentErrorCategory.EVIDENCE_GRADING_FAILURE` and the
`evidence_score` event-metadata field are the other additive contract changes.

## Sufficiency semantics and score

The prompt asks whether an answer could be written from the supplied evidence
alone, considering:

- **relevance** - the evidence addresses the question rather than sharing keywords;
- **coverage** - every important part of a multi-part question is supported;
- **specificity** - a named country, population, technology, medical context,
  condition, comparison, or intervention is supported at that specificity;
- **groundability** - no major gaps would need outside model knowledge.

| Score | Meaning |
|---|---|
| 0-39 | clearly insufficient |
| 40-59 | partial or weak evidence with material gaps |
| 60-79 | generally sufficient for a grounded answer |
| 80-100 | strong coverage |

The model returns only `{"score": N}`. The application derives
`sufficient = score >= AgentGraphConfig.evidence_sufficiency_threshold`
(default 60, validated 0-100), so 59 is insufficient and 60 is sufficient, and
the model cannot return a contradictory score/boolean pair.

The grader does not verify individual answer sentences; that is the
post-generation answer-grounding grader in W7-M08.

## original_question vs current_query

Retrieval uses `current_query`. The grader evaluates against
`original_question`. W7-M04 will rewrite the retrieval query, but the user's
information need must not drift with it.

## Evidence representation and bounding

The node builds an `EvidenceContext` with the injected builder and the grader
sends its selected sources as a JSON list of `label`, `paper_title`,
`section_title`, and `content`, in retrieval order. Deduplication, truncation,
and the token budget are the builder's existing behaviour, so prompt size is
bounded by the builder's `max_context_tokens` regardless of `retrieval_size`,
and generation can later consume the same representation. The generation
evidence contract is unchanged.

The question and evidence travel in one JSON-encoded user message under an
`UNTRUSTED_GRADING_INPUT_JSON` header. Neither is interpolated into the system
message, which states that the payload is data and that instructions inside it
must be ignored. Tests verify this construction; they do not prove real-model
injection resistance.

## Empty evidence

`HybridSearchResult` may validly contain zero hits, and the builder may select
zero sources (for example, blank chunks). In either case the grader returns
score 0, `sufficient=False`, without calling the LLM. The graph emits
`evidence_insufficient` and completes normally; this is not a retrieval
failure.

## Failure behaviour

Output is parsed with the guardrail's policy: exactly `{"score": <int 0-100>}`,
optionally inside one Markdown code fence. Anything else, an empty completion,
or a provider exception raises `EvidenceGradingError`. An unexpected exception
from `EvidenceContextBuilder.build` is handled by the same node: retrieval has
already succeeded, so it is an evidence-grading-stage failure, the grader LLM
is not called, and `evidence_grading_started` carries no `source_count`. In
every one of these cases the graph records
`terminal_reason=internal_error`,
`error_category=evidence_grading_failure`, emits `graph_failed`, and leaves
`evidence_sufficient` and `evidence_grade` unset. "Could not grade" is never
reported as "insufficient".

## Events

Sufficient / insufficient:

```text
graph_started
guardrail_started
guardrail_passed
local_retrieval_started
local_retrieval_completed
evidence_grading_started
evidence_sufficient | evidence_insufficient
graph_completed
```

Grader failure:

```text
graph_started
guardrail_started
guardrail_passed
local_retrieval_started
local_retrieval_completed
evidence_grading_started
graph_failed
```

Out-of-scope, guardrail-failure, and retrieval-failure sequences are unchanged
from W7-M02. Evidence events carry only `retrieval_attempt`, `source_count`,
`evidence_score`, and `evidence_sufficient`. No question, evidence text, title,
prompt, completion, exception text, or model reasoning is stored in events or
in `EvidenceGrade`; the model is never asked for reasoning.

## Temporary terminal semantics

Both graded outcomes emit `graph_completed` because the current topology ended
normally, and leave `terminal_reason` unset: no final answer exists, so
`completed` would mislead, and local insufficiency is not terminal in the
final design (it leads to rewrite, then live fallback), so
`insufficient_evidence` is not used either.

## W7-M04 insertion point

Change one routing entry: `"insufficient": "graph_complete"` becomes
`"insufficient": "query_rewrite"`. The grader takes a question, a context, and
a threshold; it does not know which attempt it is grading. In W7-M03
`retrieval_attempts` stays 1, `rewritten_query` stays unset, and
`current_query` is never modified.

## Deferred

Query rewriting, a second retrieval, live arXiv, PDF/Docling, transient
evidence, generation, answer grounding, prompt management, cost tracking,
cache integration, observability changes, agentic HTTP/SSE, and Gradio
integration.

## Verification

- Agent tests: 220 passed (105 after W7-M02).
- Focused LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle regression: 390 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 881 passed
  (W7-M02 baseline: 766), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fake providers and retrievers; no external service is called.
