# Week 7 M07: evidence merge, common rerank, and grounded generation

W7-M07 takes whatever usable evidence the earlier nodes produced, puts local
and live evidence into one representation, rescores all of it on one scale,
keeps a bounded final set, and generates one grounded answer with the
existing Week 5 prompt and structural citation validator.

**Local OpenSearch scores and live evidence are not directly compared. Both
are rescored through one common relevance model.**

Semantic answer grounding, regeneration, prompt versioning, caching, and
API/UI exposure are not part of this milestone.

## Topology

```text
evidence_grading
  sufficient -> evidence_rerank                      (first attempt or after retry)
live_document_processing
  evidence   -> evidence_rerank                      (local exhausted, live chunks parsed)

evidence_rerank
  failed   -> END
  empty    -> insufficient_evidence -> END
  selected -> answer_generation
                failed       -> END
                insufficient -> insufficient_evidence -> END
                generated    -> graph_complete -> END
```

Every path that previously ended at `graph_complete` with evidence now goes
through `evidence_rerank` and `answer_generation` first. The out-of-scope,
insufficient-evidence, and failure paths are unchanged.

## Two evidence sources

| | Local | Live |
|---|---|---|
| State field | `local_retrieval_result` (latest attempt) | `transient_live_evidence` |
| Origin | OpenSearch hybrid search | arXiv PDF parsed on demand |
| Native ordering | BM25/vector ranks fused by RRF | lexical cap within each paper |
| Comparable to the other? | no | no |

An RRF score and a lexical overlap count are different quantities from
different systems. Adding them, or comparing their magnitudes, would invent a
scale that does not exist, so neither is used to order the final evidence.

Locally insufficient evidence is still merged in: "insufficient alone" does
not mean "irrelevant". Local chunks may cover one part of a question and live
chunks another. Only the latest local attempt is used; the first attempt's
result was discarded by the rewrite in W7-M04 and is not restored.

## Common evidence model

`AgentEvidenceCandidate` (frozen dataclass):

| Field | Local | Live |
|---|---|---|
| `source_type` | `"local"` | `"live_arxiv"` |
| `arxiv_id`, `chunk_id`, `chunk_index` | from the hit | from the chunk (`live::` chunk ID) |
| `paper_title`, `section_title`, `text` | whitespace-normalized | whitespace-normalized |
| `authors`, `categories`, `published_date` | from the hit | from the paper |
| `source_url` | `None` | validated https PDF URL |
| `original_rank` | position in the local result | position in `transient_live_evidence` |
| `relevance_score` | `None` until reranked | `None` until reranked |

The model has no BM25, vector, or RRF field: local retrieval scores are
dropped at normalization. `original_rank` is provenance only. No OpenSearch,
Docling, HTTP, or SDK object is retained. Local hits with blank text or
missing IDs are skipped.

## Merge and deduplication

`merge_evidence(local, live)` concatenates local then live and drops, keeping
the first occurrence:

1. a repeated `chunk_id`;
2. the same version-independent arXiv ID with equivalent text (whitespace
   collapsed, case-insensitive).

Because local comes first, content present in both sources keeps its local,
already-indexed representation. Different chunks of one paper are kept, and
equal text from different papers is kept at this stage. The source-specific
state fields are not modified.

## Common reranking

`FinalEvidenceSelector` uses the injected `EmbeddingProvider`, which in the
application is the existing local BGE provider. No second model, remote API,
or model name appears in the agent.

- **Target:** `original_question`. `current_query` may be a retrieval-oriented
  rewrite and is not the user's information need.
- **Candidate text:** `paper_title`, `section_title`, `text` joined by
  newlines, identical for both sources. Authors, URLs, categories, and IDs are
  not embedded.
- **Calls:** one `embed_query` and one `embed_passages` batch.
- **Offload:** the provider is synchronous and CPU-bound, so both calls run
  inside a single `asyncio.to_thread`.
- **Score:** true cosine similarity, computed with explicit norms. The
  existing provider does normalize its vectors, but the `EmbeddingProvider`
  protocol does not promise it.
- **Order:** score descending; ties keep merged order (local before live,
  then each source's own rank). There is no randomness.
- **Bound on embedding work:** at most 64 candidates are embedded. With the
  default configuration the merged set is at most 5 local + 16 live. Beyond
  the cap, candidates are kept alternately from each source in rank order so
  neither is starved.

Mismatched vector counts or dimensions, zero vectors, non-finite scores, and
provider exceptions raise `EvidenceRerankError`. There is no fallback to an
arbitrary order.

No diversity rule is applied: after deduplication the final ranking is purely
by relevance, with no per-paper cap and no local/live quota.

## Final evidence selection

`AgentGraphConfig.final_evidence_max_sources` (default 5, validated 1-20)
bounds the final set independently of retrieval size, paper count, and
chunks per paper. The result is stored in `final_evidence` as ranked
candidates with their `relevance_score`.

If nothing survives normalization and merge, `final_evidence` is `()`, no
embedding or generation call is made, and the graph completes with
`terminal_reason=insufficient_evidence`.

## EvidenceContext construction

`EvidenceContextBuilder` gained `build_from_sources(inputs, *, query,
retrieval_mode, degradation_reason=None)`, which takes already-ranked
`EvidenceInput` items of any provenance. The existing `build(result)` now
converts hits to `EvidenceInput` and calls it, so Week 5 output is identical
and labeling, block rendering, token budgeting, truncation, and
deduplication have one implementation. No `HybridSearchResult` is faked.

`EvidenceSource` gained `source_type` (default `"local"`) and `source_url`
(default `None`), and `rrf_score` may now be `None` for evidence that has no
retrieval score. The public `/ask` source schema exposes none of these.

Final candidates become `[S1]`, `[S2]`, ... in final ranked order, for local
and live alike; provenance is metadata, not citation syntax. The context
`retrieval_mode` and `degradation_reason` are those of the latest local
retrieval.

The builder's own rules still apply: a source that does not fit the token
budget is truncated or omitted, and text identical to an earlier source is
dropped even when it comes from a different paper. So the sources actually
sent to the model, recorded in `generation_result.sources`, can be fewer than
`final_evidence`.

## Grounded generation

`RAGGenerationService` gained `generate_from_context(question, evidence)` and
`prepare_from_context(question, evidence)`. The existing
`generate(question, retrieval_result)` and `prepare(...)` share the same
internals: one prompt builder, one context-window check, one provider call,
one structural validator. `/ask` and `/ask/stream` behave as before.

The agent calls `generate_from_context` exactly once per run with
`original_question`. The existing `RAGPromptBuilder` and
`GroundedAnswerValidator` are used unchanged, so an answer is rejected when
it is empty, uses malformed citation syntax, cites an unknown label or
`[S0]`, or makes a substantive claim without a citation. A citation-free
answer is valid only when it explicitly states the evidence is insufficient.

On success the state holds:

| Field | Content |
|---|---|
| `final_evidence` | ranked `AgentEvidenceCandidate` tuple |
| `generation_result` (new) | existing `RAGGenerationResult`: answer, labeled sources, cited labels, model, token counts |
| `generated_answer` | the validated answer text |
| `grounding_passed`, `grounding_attempts` | unchanged (`None`, 0) |
| `terminal_reason`, `error_category` | `None` |

The answer is stored only after structural validation passes.

## Failure and insufficiency

| Situation | Events tail | `terminal_reason` | `error_category` |
|---|---|---|---|
| Embedding or scoring fails | `evidence_rerank_started, graph_failed` | `internal_error` | `evidence_rerank_failure` |
| No candidates after merge | `evidence_rerank_started, evidence_rerank_completed, graph_completed` | `insufficient_evidence` | `None` |
| Nothing fits the context budget | `..., generation_started, graph_completed` | `insufficient_evidence` | `None` |
| Provider error, prompt budget error, or structural validation failure | `..., generation_started, graph_failed` | `generation_failed` | `generation_failure` |

A failed generation leaves `generated_answer` and `generation_result` unset.
There is no retry.

## Events

Local-sufficient tail:

```text
evidence_grading_started
evidence_sufficient
evidence_rerank_started
evidence_rerank_completed
generation_started
generation_completed
graph_completed
```

Live tail:

```text
live_document_processing_started
live_document_processing_completed
evidence_rerank_started
evidence_rerank_completed
generation_started
generation_completed
graph_completed
```

Two statuses (`evidence_rerank_started`, `evidence_rerank_completed`) and six
metadata fields were added: `local_candidate_count`, `live_candidate_count`,
`merged_candidate_count`, `final_source_count`, `prompt_tokens`,
`completion_tokens`. `generation_started` and `generation_completed` were
already reserved.

Events carry counts and token counts only. No question, query, title,
abstract, chunk text, URL, embedding, relevance score, prompt, answer, model
name, completion, exception text, or reasoning is placed in events.

## No side effects

M07 performs no PostgreSQL write, no OpenSearch write, no Airflow trigger, and
no live-paper promotion. Embeddings are computed in memory for ranking and
discarded.

## W7-M08 insertion point

Change one routing entry after `answer_generation`:
`"generated": "graph_complete"`. M08 consumes `generation_result` (answer,
labeled sources as sent to the model, cited labels, model, token counts),
`final_evidence`, and `original_question`. The reserved `grounding_*`
statuses, state fields, terminal reason, and error category are untouched.

## Deferred

Semantic answer grounding, regeneration, prompt management, Langfuse
generation observations, cost tracking, agent cache identity, an overall
request deadline, agentic HTTP/SSE, and Gradio integration.

## Verification

- Agent tests: 727 passed (622 after W7-M06).
- Evidence-context and RAG generation/validation tests: 110 passed.
- Focused embeddings/arXiv/PDF/chunking/LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle regression: 505 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 1424 passed
  (W7-M06 baseline: 1295), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fake LLM and embedding providers; the real BGE model is not loaded.
