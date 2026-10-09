# W5-M07 structural grounding validation

Both grounded endpoints now use one deterministic `GroundedAnswerValidator`
after generation. It receives only the completed canonical answer and the
application-controlled evidence sources. It does not retrieve data, invoke an
embedding model, call an LLM, or attempt answer repair.

## Citation contract

The only accepted citation form is `[S<number>]`, where numbering begins at 1
and the label must exactly match a source supplied to the model. Full-width
`【S<number>】` is canonicalized to `[S<number>]` before validation and before the
answer is returned. `[S0]` and `【S0】` are always invalid, even if a malformed
source list were to contain S0.

Unknown labels are rejected. Obvious attempts to use the source namespace with
malformed syntax are also rejected, including `[S]`, `[Sabc]`, `[S-1]`, `[S 1]`,
and their full-width equivalents. Detection is deliberately scoped: ordinary
bracketed words such as `[Summary]` and uppercase acronyms such as `[SQL]` are
not treated as citation attempts.

Repeated citations remain in the answer but terminal metadata deduplicates them
in first-occurrence order. Citations may occur in any order, and not every
supplied source needs to be cited. Successful responses continue returning all
authoritative sources supplied to the model.

## Citation-free refusals

A substantive answer must cite at least one supplied source. A citation-free
answer is accepted only when the entire normalized answer matches a narrow
refusal family:

- available/retrieved/supplied/provided evidence or sources are insufficient;
- those sources do not provide enough information;
- there is not enough evidence or information in the supplied material; or
- the answer cannot be determined from the supplied evidence or sources.

These patterns are anchored to the complete answer. Merely including words such
as “insufficient evidence” does not exempt additional uncited claims. Ambiguous
or mixed refusal-and-claim output fails closed.

The deterministic no-evidence fallback is unchanged and bypasses model-answer
validation because no model is called.

## Scope and limitations

This is structural validation, not semantic entailment or factual verification.
For example, a false claim followed by an existing `[S1]` can pass structurally;
M07 does not prove that the source supports the claim. Semantic grading belongs
to later evaluation work and is not implemented here.

The system prompt reinforces that every substantive claim requires a supplied
citation and that citation-free output is reserved for explicit evidence
insufficiency. Questions and evidence remain delimited as untrusted data, so
instructions embedded in either are not promoted into the system message.

## Streaming semantics

Streaming remains genuinely incremental and does not buffer the answer for
pre-emission validation. A successful grounded stream is:

`metadata -> delta(s) -> sources -> done`

`done` is the terminal proof that structural validation succeeded. A stream
ending in `error`, or a connection ending without `done`, is not a successful
grounded answer even if text was already emitted. Previously emitted text cannot
be retracted.

Grounding failures use the stable SSE code `grounding_validation_failed` and
emit neither `sources` nor `done`. Provider and transport failures retain the
existing `generation_failed` code. Non-streaming failures retain the safe HTTP
502 response. No raw answer, prompt, evidence, provider body, or exception is
included in public errors.

Validation failure does not trigger another retrieval, an LLM retry, or an
answer-repair call. Transient request retries remain exclusively at the existing
provider/SDK boundary.

## Local acceptance (2026-10-09)

Standalone Uvicorn used the process-local
`OPENSEARCH_HOST=http://127.0.0.1:9200` override. Ping, health, and hybrid-search
prechecks returned 200. The corpus remained at 17 PostgreSQL papers, 17 paper
documents, 513 chunks, and 15 distinct chunk paper IDs; no synchronization work
was performed.

The supported healthcare-access stream returned 200 with
`metadata -> delta(s) -> sources -> done`. It used hybrid retrieval, five sources,
384 deltas, and allowed citations `[S1]`, `[S2]`, `[S5]`, and `[S3]`. The model was
`openai/gpt-oss-120b`; provider usage was 4,440 prompt and 672 completion tokens.
Time to metadata was 2.128 seconds, time to first generated delta was 3.477
seconds, and total duration was 4.280 seconds. The answer ended naturally.

One out-of-domain sourdough question was tested without prompt tuning or retry at
the RAG layer. Retrieval returned nonempty evidence, Groq completed after one
natural SDK-level 429 retry, and deterministic validation failed closed as a safe
HTTP 502. The public response intentionally did not expose the rejected model
text, so it cannot be classified as an accepted citation-free insufficiency
answer. This is safe behavior, but it also demonstrates the deliberate narrowness
of the deterministic refusal classifier and the unresolved semantic-sufficiency
limitation.
