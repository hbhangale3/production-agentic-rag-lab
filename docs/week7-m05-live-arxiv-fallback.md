# Week 7 M05: bounded live arXiv fallback

When local retrieval is exhausted and still insufficient, W7-M05 searches live
arXiv once and stores a small set of typed metadata-plus-abstract candidates.

**M05 discovers live metadata and abstracts. M05 does not fetch or parse
PDFs.** It also does not chunk, embed, index, persist, merge, rerank, grade
the abstracts, or generate an answer.

## Topology

```text
START -> graph_start -> guardrail
           rejected -> out_of_scope -> END
           failed   -> END
           passed   -> local_retrieval <------------------+
                         failed    -> END                 |
                         retrieved -> evidence_grading    |
                           sufficient    -> graph_complete -> END
                           rewrite       -> query_rewrite
                                              rewritten --+
                                              failed   -> END
                           live_fallback -> live_arxiv_search
                                              candidates -> graph_complete -> END
                                              empty      -> insufficient_evidence -> END
                                              failed     -> END
                           exhausted     -> insufficient_evidence -> END
                           failed        -> END
```

## Trigger

After an insufficient grade with no local attempt left
(`retrieval_attempts >= max_local_retrieval_attempts`, no error):

| `live_fallback_enabled` | Route |
|---|---|
| `True` (default) | `live_fallback` -> `live_arxiv_search` |
| `False` | `exhausted` -> `insufficient_evidence` |

Live search is never reached when local evidence is sufficient, before local
attempts are exhausted, or from any failure path.

With fallback disabled, arXiv is not called, `live_fallback_used` stays
`False`, and the graph completes normally with
`terminal_reason=insufficient_evidence` and no error category: every evidence
source the configuration allows has been used. `build_agent_graph` requires a
`live_search_service` only when fallback is enabled, and omits the
`live_arxiv_search` node otherwise.

## Live search abstraction

The graph depends on the `LiveResearchSearchService` protocol:

```python
async def search(query, *, max_results, exclude_arxiv_ids=()) -> LiveArxivSearchResult
```

`ArxivLiveSearchService` implements it over the existing Week 2 `ArxivClient`.
The graph contains no HTTP, Atom, or XML code.

### Reuse of Week 2 infrastructure

| Reused | From |
|---|---|
| HTTP request, timeout, retry cap, rate limiting, User-Agent | `ArxivClient.fetch_papers` / `_request` |
| Atom parsing, title/abstract whitespace cleaning, PDF link, dates | `ArxivClient.parse_response` |
| Version-independent ID rule | `ARXIV_VERSION_SUFFIX` |
| Metadata shape | `ArxivPaper` |

No second arXiv protocol implementation exists and `ArxivClient` is unchanged.
The adapter calls only `fetch_papers`, which performs one metadata request and
has no persistence, download, or indexing side effects. The ingestion-side
`MetadataFetcher`, `download_pdf`, Docling, and indexing services are not
used.

### Query

Live search uses `current_query`: after local exhaustion that is the rewritten,
retrieval-optimised query (or the original question when
`max_local_retrieval_attempts=1`). `original_question` is unchanged. No LLM is
called and no second rewrite happens.

`ArxivClient` accepts raw arXiv query syntax, so the adapter never passes user
text through. It extracts alphanumeric terms, drops a small stop-word list
(including the `AND`/`OR`/`ANDNOT` operators), deduplicates case-insensitively,
keeps at most 16, and builds `all:term OR all:term ...`, sorted by relevance.
Quotes, parentheses, colons, and field prefixes in the text cannot change the
expression. A query with no usable terms returns an empty result without a
network request.

### Async boundary and network behaviour

`ArxivClient.fetch_papers` is already asynchronous, so the node awaits it; no
thread offload is needed. One fallback makes one logical search. Timeout,
retry count, and the inter-request delay are those of the injected client
(`arxiv_timeout`, `arxiv_max_retries`, `arxiv_rate_limit_delay` when built by
`make_arxiv_client`), so retries are finite. No overall deadline is added on
top; see follow-ups.

## Candidate model

`LiveArxivPaper` (frozen dataclass): `arxiv_id`, `title`, `abstract`,
`authors` (tuple), `categories` (tuple), `published_date`, `updated_date`,
`pdf_url` (string).

`LiveArxivSearchResult` (frozen dataclass): `query`, `candidates` (tuple),
and a derived `count`.

There is no `entry_url`: the existing parser does not retain the entry link,
and one is not invented. No feed, response, client, or exception object enters
state.

### Normalization, filtering, bounding

- Title, abstract, authors, and categories: whitespace collapsed; blank authors
  and categories dropped. Abstract text is not otherwise altered or summarised.
- `arxiv_id`: version suffix removed. `pdf_url` is kept as arXiv supplied it.
- Order: arXiv's relevance order is preserved.
- Duplicates: by normalized arXiv ID, first occurrence kept. Titles are not
  compared.
- Entries with a blank title or abstract are dropped individually. An entry
  missing a required Atom field still fails the whole feed, which is the
  existing parser's behaviour and surfaces as a live-search failure.
- At most `live_arxiv_max_results` candidates are returned (default 5,
  validated 1-10). The adapter truncates after filtering, and the graph node
  truncates again if a service returns more.

### Already-local papers

The node passes the arXiv IDs of the hits in the **latest**
`local_retrieval_result` as `exclude_arxiv_ids`; candidates with those
normalized IDs are dropped. To avoid starving the bounded result, the adapter
requests `max_results + excluded`, capped at `2 * max_results`.

This is not corpus-wide: a paper that is indexed locally but was not among the
latest local hits can still appear as a live candidate. No PostgreSQL or
OpenSearch lookup is made for this. Stronger cross-source normalization is
deferred to W7-M07.

## State

| Field | After live search |
|---|---|
| `live_fallback_used` | `True` once a live search is attempted (candidates, zero, or failure); `False` otherwise |
| `live_search_result` (new) | the typed result; `None` if not attempted or failed |
| `local_retrieval_result` | latest local result, retained |
| `evidence_sufficient`, `evidence_grade` | latest local grade, still insufficient |
| `retrieval_attempts` | unchanged; live search is not a local attempt |
| `live_evidence`, `final_evidence` | still unset; reserved for processed evidence |

## Outcomes and terminal semantics

| Outcome | Events tail | `terminal_reason` | `error_category` |
|---|---|---|---|
| Candidates found | `live_fallback_started, live_fallback_completed, graph_completed` | `None` (M06 continues) | `None` |
| Zero candidates | same | `insufficient_evidence` | `None` |
| Search failed | `live_fallback_started, graph_failed` | `internal_error` | `live_search_failure` |
| Fallback disabled | `graph_completed` | `insufficient_evidence` | `None` |

Zero candidates means the search ran and found nothing usable; a failure means
it could not run. They are never collapsed. Candidates do not make
`evidence_sufficient` true.

Full live-success sequence:

```text
graph_started, guardrail_started, guardrail_passed,
local_retrieval_started, local_retrieval_completed,
evidence_grading_started, evidence_insufficient,
query_rewrite_started, query_rewritten,
local_retrieval_started, local_retrieval_completed,
evidence_grading_started, evidence_insufficient,
live_fallback_started, live_fallback_completed,
graph_completed
```

## Safe events and privacy

Two metadata fields were added: `max_results` and `candidate_count`. Live
events carry only `live_fallback_used`, `max_results`, and `candidate_count`.
No query, title, abstract, author, category, URL, feed content, or exception
text is placed in events, and the failure path stores nothing from the
exception in state.

## W7-M06 insertion point

Change one routing entry after `live_arxiv_search`:
`"candidates": "graph_complete"` becomes the paper-selection node. M06 reads
`state["live_search_result"].candidates`, each with `arxiv_id`, `title`,
`abstract`, `pdf_url`, and provenance metadata.

## Deferred

PDF acquisition, Docling, transient chunks and embeddings, persistence,
merge/rerank, grading of live evidence, generation, answer grounding, prompt
management, cost tracking, cache integration, observability changes, agentic
HTTP/SSE, and Gradio integration.

## Verification

- Agent tests: 413 passed (329 after W7-M04).
- Existing arXiv client, discovery-profile, and metadata-fetcher tests: 39 passed.
- Focused arXiv/LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle regression: 429 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 1074 passed
  (W7-M04 baseline: 990), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fakes or an in-process HTTP mock transport; no network request is made.
