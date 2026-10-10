# Week 7 M08: semantic answer grounding and bounded regeneration

W7-M08 checks, after a structurally valid answer has been generated, whether
the answer's claims are actually supported by the sources it was generated
from. If not, the graph regenerates once from the same evidence and checks
again. A second failure ends the run as `grounding_failed`.

**M08 grades the answer against the exact sources sent to generation.**

Nothing is retrieved, rewritten, searched, or reranked again. Prompt
versioning, Langfuse changes, cost tracking, caching, and API/UI exposure are
not part of this milestone.

## M03 versus M08

| | M03 evidence sufficiency | M08 answer grounding |
|---|---|---|
| When | before generation | after generation |
| Question asked | Can these sources answer the question? | Is this answer supported by these sources? |
| Input | question + retrieved evidence | question + generated answer + sources sent to generation |
| Component | `EvidenceSufficiencyGrader` | `AnswerGroundingGrader` |

They are separate components with separate prompts. The evidence grader is
not reused.

## Topology

```text
answer_generation
  failed       -> END
  insufficient -> insufficient_evidence -> END
  generated    -> answer_grounding
                    failed     -> END
                    passed     -> graph_complete -> END
                    regenerate -> answer_generation      (same evidence)
                    exhausted  -> grounding_failed -> END
```

After a failed grade the router chooses:

```text
grounding_attempts < max_grounding_attempts  -> regenerate
otherwise                                    -> exhausted
```

Everything before `answer_generation` is unchanged from W7-M07.

## Order of validation

```text
generate -> structural validation -> semantic grounding
```

The existing `GroundedAnswerValidator` still runs inside
`RAGGenerationService` and owns citation syntax: empty answers, malformed
labels, unknown labels, `[S0]`, and uncited substantive claims. An answer that
fails it is a generation failure and never reaches the semantic grader. The
semantic grader owns meaning: whether cited sources support the claims.

## Grounding input

The grader receives:

- `original_question`;
- `generation_result.answer`;
- `generation_result.sources`.

`generation_result.sources` is authoritative because it is the labeled,
token-budgeted set the model actually saw. `final_evidence` can be larger:
the context builder omits sources that do not fit the budget, truncates the
last one that partly fits, and drops text identical to an earlier source.
Grading against content the model never saw would credit or penalise the
answer for evidence it could not have used, so `final_evidence` is not used.
Truncated sources are graded in their truncated form.

Per source the prompt carries `label`, `paper_title`, `section_title`, and
`content`. Authors, URLs, IDs, categories, source type, retrieval scores, and
embeddings are not sent. The payload is bounded by the evidence context
budget plus one answer.

## Grader

| Component | Responsibility |
|---|---|
| `AnswerGroundingPromptBuilder` | Local replaceable prompt; isolates untrusted inputs |
| `AnswerGroundingGrader` | Calls `LLMProvider.complete`, parses the score, derives pass/fail |
| `AnswerGroundingResult` | `score`, `passed`, `grader_version` |
| `AnswerGroundingError` | Provider or parsing failure |

The injected `LLMProvider` is reused (temperature 0.0, `max_tokens=32`). The
question, answer, and sources travel in one JSON-encoded user message under
`UNTRUSTED_GROUNDING_INPUT_JSON`; none is interpolated into the system
message, which says the payload is data and instructions inside it must be
ignored. Tests verify this construction, not real-model injection resistance.

The prompt forbids outside knowledge and asks the model to check, against the
labeled sources each claim cites, that:

- factual claims are supported by the cited source;
- important claims carry a citation, and that citation supports the claim;
- no material fact is added that the sources do not contain;
- evidence is not overstated and uncertainty is not turned into certainty;
- the answer stays within the scope of the sources.

| Score | Meaning |
|---|---|
| 0-39 | major unsupported or contradicted claims |
| 40-59 | partially supported; material claims weakly grounded or unsupported |
| 60-79 | generally grounded; important claims supported |
| 80-100 | strongly grounded; claims closely match cited evidence |

The model returns exactly `{"score": N}`, optionally inside one Markdown code
fence: the shared strict parser. The application derives
`passed = score >= AgentGraphConfig.answer_grounding_threshold` (default 60,
validated 0-100), so a contradictory score/boolean pair cannot be returned.
The model is not asked for reasoning or a claim list, and none is stored.
`grader_version` is the constant `"local-v1"`; real prompt versioning is
W7-M09.

## Attempts and regeneration

`grounding_attempts` counts grading calls attempted. A grader call that fails
still counts.

`AgentGraphConfig.max_grounding_attempts` is the **total** number of grading
attempts, reusing the field reserved in W7-M01. The default is 2:

```text
attempt 1: grade the generated answer
attempt 2: grade the one regenerated answer
```

With `max_grounding_attempts=1` a failed first grade ends as
`grounding_failed` with no regeneration. The bound is enforced by the router
and again inside the grounding node, which refuses to grade when
`grounding_attempts >= max_grounding_attempts` (`graph_failed`,
`internal_error` / `internal_failure`).

### How regeneration works

The same `answer_generation` node runs again. It rebuilds the context from
the unchanged `final_evidence` with the same builder, which is deterministic,
so the sources, labels, order, and content are identical. No retrieval,
rewrite, live search, document processing, or rerank occurs.

`RAGGenerationService.generate_from_context` gained an optional
`previous_answer`. When given, `RAGPromptBuilder.build_regeneration` produces
the same system prompt and the same user prompt, followed by:

```text
BEGIN PREVIOUS ANSWER (untrusted data)
<the rejected answer, at most 4000 characters>
END PREVIOUS ANSWER

<a fixed instruction: answer again from the evidence only, make no unsupported
claim, cite every substantive claim, say when evidence is insufficient>
```

The rejected answer is included so the model can correct it; it is marked as
untrusted data. No grader score or output is included, and there is no grader
reasoning to include. Temperature, completion limit, context-window check,
and structural validation are those of every other generation. Without
`previous_answer` the service behaves exactly as before.

### State across a regeneration

On a successful regeneration `generated_answer` and `generation_result` are
replaced by the new answer, and `grounding_passed` and `grounding_result` are
cleared until the new answer is graded, so the first decision never appears
to describe the second answer. The first answer is not kept anywhere in
state; only its events remain.

If regeneration fails (provider error or structural validation),
`generated_answer` and `generation_result` are cleared, so the rejected first
answer does not remain looking current.

## Outcomes

| Situation | Events tail | `terminal_reason` | `error_category` |
|---|---|---|---|
| First grade passes | `grounding_started, grounding_passed, graph_completed` | `None` | `None` |
| Fail, regenerate, pass | `..., grounding_failed, generation_started, generation_completed, grounding_started, grounding_passed, graph_completed` | `None` | `None` |
| Fail on every allowed attempt | `..., grounding_started, grounding_failed, graph_completed` | `grounding_failed` | `None` |
| Grader provider or parser failure | `grounding_started, graph_failed` | `internal_error` | `answer_grounding_failure` |
| Regeneration provider or structural failure | `..., grounding_failed, generation_started, graph_failed` | `generation_failed` | `generation_failure` |

A low score is a semantic outcome: the graph completes normally with
`terminal_reason=grounding_failed`, like `insufficient_evidence`. A grader
that could not run is an execution failure. They are never collapsed.

When grounding is exhausted, the last answer remains in `generated_answer`
with `grounding_passed=False`. Callers must check `terminal_reason` and
`grounding_passed` before presenting it.

`AgentErrorCategory.ANSWER_GROUNDING_FAILURE` was added for grader execution
failures. The older reserved `GROUNDING_FAILURE` value is not emitted.

## Citation-free insufficiency answers

The structural validator allows a citation-free answer only when it
explicitly states that the evidence is insufficient. M08 does not fail such
an answer for lacking citations. It is sent to the grader like any other
answer, and the prompt says to judge it by whether the statement is
consistent with the sources. A pass completes normally; a low score follows
the ordinary regeneration path.

## Events

First pass:

```text
generation_started
generation_completed
grounding_started
grounding_passed
graph_completed
```

Regenerate then pass (or fail on the last line but one):

```text
generation_started        generation_attempt 1
generation_completed      generation_attempt 1
grounding_started         grounding_attempt 1
grounding_failed          grounding_attempt 1
generation_started        generation_attempt 2
generation_completed      generation_attempt 2
grounding_started         grounding_attempt 2
grounding_passed | grounding_failed
graph_completed
```

The three `grounding_*` statuses were reserved in W7-M01. Three metadata
fields were added: `grounding_attempt`, `grounding_score`, and
`generation_attempt`; `grounding_passed` already existed. No question,
answer, previous answer, source text, title, prompt, grader output,
exception text, or reasoning is placed in events.

## W7-M09 insertion point

M09 can attach to what already exists without changing routing:

- prompts to version: guardrail, evidence grader, query rewriter, paper
  selector, RAG generation and its regeneration instruction, answer grounding.
  Each has a prompt-builder class or constant as its single source;
- `AnswerGroundingResult.grader_version` and `EvidenceGrade.grader_version`
  for prompt identity;
- `generation_result` for model and token counts; `generation_attempt` and
  `grounding_attempt` in events for per-attempt telemetry;
- `RAGGenerationService` already accepts an `observation` on
  `generate_from_context`; the agent does not pass one yet.

## Deferred

Langfuse prompt management, prompt identity and fingerprints, generation
observations, latency/token/cost accounting, agent-aware cache identity, an
overall request deadline, agentic HTTP/SSE, and Gradio integration.

## Verification

- Agent tests: 858 passed (727 after W7-M07).
- RAG generation/validation and evidence-context tests: 121 passed.
- Focused embeddings/arXiv/PDF/chunking/LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle regression: 516 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 1566 passed
  (W7-M07 baseline: 1424), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fake LLM and embedding providers; no real model is called.
