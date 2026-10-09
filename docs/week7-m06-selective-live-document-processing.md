# Week 7 M06: selective live document processing

W7-M06 turns live arXiv candidates into in-memory full-text evidence for the
current request.

**Live arXiv candidates are not automatically downloaded. Only a bounded,
selected subset is processed.** The default is two papers out of at most five
candidates.

M06 does not merge or rerank local and live evidence, grade the live evidence,
generate an answer, or persist anything.

## Topology (live tail)

```text
evidence_grading
  live_fallback -> live_arxiv_search
                     failed     -> END
                     empty      -> insufficient_evidence -> END
                     candidates -> live_paper_selection
                                     failed   -> END
                                     none     -> insufficient_evidence -> END
                                     selected -> live_document_processing
                                                   failed   -> END
                                                   unusable -> insufficient_evidence -> END
                                                   evidence -> graph_complete -> END
```

Everything before `live_arxiv_search` is unchanged from W7-M05. When live
fallback is disabled none of the three live nodes is added.

## Paper selection

| Component | Responsibility |
|---|---|
| `LivePaperSelectionPromptBuilder` | Local replaceable prompt; isolates untrusted inputs |
| `LivePaperSelector` | One `LLMProvider.complete` call for the whole candidate set; validates IDs |
| `LivePaperSelection` | The chosen `LiveArxivPaper` objects, nothing else |
| `LiveSelectionError` | Provider or parsing failure |

The selector receives `original_question` and, per candidate, only `arxiv_id`,
`title`, and `abstract` (abstract capped at 2000 characters in the prompt).
They travel as one JSON-encoded user message under
`UNTRUSTED_SELECTION_INPUT_JSON`; nothing is interpolated into the system
message, which says to ignore instructions inside the payload. Temperature is
0.0 and `max_tokens` is 128.

The model must return exactly `{"selected_arxiv_ids": ["...", ...]}`,
optionally in one Markdown code fence: the shared strict parser policy. No
scores or rationale are requested or stored.

Local validation, in model order:

- an ID not among the candidates is dropped and never acted on;
- duplicates, including versioned forms of the same ID, are dropped;
- anything beyond the limit is dropped.

The limit is `AgentGraphConfig.live_pdf_max_papers` (default 2, at least 1,
and validated not to exceed `live_arxiv_max_results`). The graph node
truncates again.

| Selection outcome | Result |
|---|---|
| One or more valid IDs | `live_selected_papers` set; processing runs |
| Zero valid IDs (empty list, or only unknown IDs) | no download; `terminal_reason=insufficient_evidence`; `graph_completed` |
| Provider or parsing failure | `graph_failed`; `internal_error` / `live_selection_failure` |

## PDF acquisition

`LivePaperAcquisitionService` wraps the existing Week 2
`ArxivClient.download_pdf`, which already provides streaming, PDF-signature
validation, atomic write, retry cap, rate limiting, and the filesystem-safe
filename derived from the arXiv ID. The graph contains no HTTP code.

Two optional keyword arguments were added to `ArxivClient.download_pdf`;
ingestion calls are unaffected because both default to the previous behaviour:

- `cache_dir`: write to this directory instead of the configured cache;
- `max_bytes`: abort when the declared `Content-Length` or the streamed size
  exceeds the limit. A size violation is not retried.

### URL validation

Only the `pdf_url` of a selected `LiveArxivPaper` is ever used.
`validate_arxiv_pdf_url` requires:

- host exactly `arxiv.org`, `www.arxiv.org`, or `export.arxiv.org`;
- no user-info, and no non-default port;
- a path under `/pdf/`;
- scheme `https`, or `http` on one of those hosts, which is rewritten to
  `https` because arXiv feeds may still advertise `http` links.

Query string and fragment are dropped. Anything else fails that paper before
any request. There is no generic downloader.

### Bounds

| Bound | Value | Owner |
|---|---|---|
| Papers processed | `live_pdf_max_papers`, default 2 | graph config |
| PDF size | 20 MB default | `LivePaperAcquisitionService(max_bytes=...)` |
| Per-request timeout, retries, rate limit | the injected `ArxivClient`'s | client |
| Pages and file size at parse time | the injected parser's (`pdf_parser_max_pages`, `pdf_parser_max_file_size_mb`) | parser |
| Per-paper deadline (download + parse) | 120 s default | `LiveDocumentProcessingService(paper_timeout_seconds=...)` |
| Chunks kept per paper | `live_max_chunks_per_paper`, default 8 | graph config |

Papers are processed one at a time: Docling is CPU-heavy and arXiv asks for
unhurried requests. The per-paper deadline bounds how long the graph waits; a
Docling conversion already running in a worker thread cannot be interrupted
and finishes in the background.

### Temporary files

Each paper is downloaded into its own `tempfile.TemporaryDirectory`
(`live-arxiv-*`), parsed, and the directory is removed, whether parsing
succeeds, fails, or times out. The canonical ingestion cache
(`arxiv_pdf_cache_dir`) is never written, so live PDFs cannot leak into the
ingestion corpus and do not accumulate.

## Docling parsing

The existing `DoclingPDFParser.parse_pdf` is used as is. It validates the
file, enforces its page and size limits, and already offloads conversion with
`asyncio.to_thread`, so the graph's event loop is not blocked and no second
offload is added. The service depends on a small `PDFParser` protocol so that
importing the agent package does not load Docling.

## Chunking

The existing Week 4 `PaperChunkingService.chunk_paper` chunks the parsed
paper: section-aware when sections exist, raw-text fallback otherwise, with
the injected service's target, overlap, and minimum sizes. No second chunking
algorithm exists.

When a paper yields more chunks than `live_max_chunks_per_paper`, the chunks
sharing the most distinct content terms with `original_question` are kept
(ties broken by position) and returned in document order. This is a
deterministic lexical cap to bound memory. **It is not a final rerank**; that
is W7-M07. The default of 8 is deliberately below a whole paper: with
600-word chunks, two papers at 8 chunks is already more text than the
6000-token evidence context.

No embeddings are computed and nothing is indexed.

## Transient evidence model

`TransientLiveEvidenceChunk` (frozen dataclass):

| Field | Source |
|---|---|
| `source_type` | always `"live_arxiv"` |
| `arxiv_id`, `paper_title`, `authors`, `categories`, `published_date` | the selected `LiveArxivPaper` |
| `pdf_url` | the validated https URL |
| `chunk_id` | `live::<arxiv_id>::chunk::<NNN>` |
| `chunk_index` | position in the paper's full chunking, stable across the cap |
| `section_title`, `text`, `word_count` | the Week 4 chunk |

The `live::` prefix keeps these IDs distinct from local index chunk IDs. No
parser, chunker, or HTTP object is retained.

`LiveDocumentProcessingResult` carries the chunks plus `selected_count`,
`processed_count`, `unusable_count`, and `failed_count`, which must add up.

## Per-paper outcomes

Papers are independent, and each ends in exactly one of three outcomes:

| Outcome | When | Counted as |
|---|---|---|
| Usable | Acquired, parsed, and chunking yields at least one chunk | processed |
| Unusable (no extractable evidence) | Acquired, and either the parser reports that the PDF has no extractable text, or the parsed text produces no chunk | unusable |
| Execution failure | URL rejected, download error, size limit, deadline, invalid PDF, any other parser error, I/O error, unexpected chunking error | failed |

"No extractable text" has a typed signal. `DoclingPDFParser` now raises
`PDFNoTextError`, a subclass of the existing `PDFParserError`, where it
previously raised `PDFParserError` with the same message; existing callers
that catch `PDFParserError` are unaffected. The processing service treats
only that exception type as unusable. Every other parser error, including a
plain `PDFParserError`, remains an execution failure; nothing is classified
by message text.

| Overall | Result |
|---|---|
| At least one usable paper | `transient_live_evidence` set; `graph_completed`; `terminal_reason=None` |
| No usable paper, and at least one was unusable | `transient_live_evidence=()`; `insufficient_evidence`; `error_category=None`; `graph_completed` |
| No usable paper, and every selected paper failed (or the processor itself raised) | `transient_live_evidence=()`; `graph_failed`; `internal_error` / `live_document_processing_failure` |

Examples with two selected papers:

| A | B | Result |
|---|---|---|
| unusable | unusable | insufficient evidence |
| unusable | failed | insufficient evidence |
| failed | failed | processing failure |
| usable | failed | success with A's evidence |
| usable | unusable | success with A's evidence |

One unusable or failing paper never discards another's chunks. A failure is
visible only as `failed_count`; unusable papers are the remainder of
`selected_count`. The exception's message, which contains a file path, is not
stored in the result, state, or events.

## State

| Field | Meaning |
|---|---|
| `local_retrieval_result` | latest local result, retained |
| `evidence_grade`, `evidence_sufficient` | latest local grade; still `False` after M06 |
| `live_search_result` | M05 metadata candidates, retained |
| `live_selected_papers` (new) | papers chosen for processing; `()` for zero selection |
| `transient_live_evidence` (new) | parsed live chunks; `None` if processing never ran |
| `live_evidence`, `final_evidence` | still unset |

Live chunks do not make `evidence_sufficient` true.

## Events

Live-success tail:

```text
live_fallback_started
live_fallback_completed
live_selection_started
live_selection_completed
live_document_processing_started
live_document_processing_completed
graph_completed
```

Failure tails: `live_selection_started, graph_failed` and
`live_document_processing_started, graph_failed`.

Four statuses and four metadata fields were added. Events carry counts only:
`candidate_count`, `selected_count`, `processed_count`, `failed_count`,
`transient_chunk_count`. No arXiv ID, title, abstract, URL, document or chunk
text, prompt, completion, exception text, or reasoning is placed in events,
and failures store nothing from the exception in state.

## No persistence

M06 performs no PostgreSQL write, no OpenSearch write, no Airflow trigger, no
embedding, and no canonical-cache write. The only filesystem writes are the
per-paper temporary directories, which are removed.

## W7-M07 insertion point

Change one routing entry after `live_document_processing`:
`"evidence": "graph_complete"`. M07 consumes, as separate typed state,
`local_retrieval_result`, `evidence_grade`, `live_search_result`, and
`transient_live_evidence`.

## Deferred

Local/live normalization, merge, rerank, final evidence selection, generation,
answer grounding, prompt management, cost tracking, cache integration,
observability changes, an overall request deadline, agentic HTTP/SSE, and
Gradio integration.

## Verification

- Agent tests: 622 passed (413 after W7-M05).
- Existing arXiv client, PDF parser, chunking, and metadata-fetcher tests: 79 passed
  (includes 4 new client tests for `cache_dir` and `max_bytes` and 4 parser tests for `PDFNoTextError`).
- Focused arXiv/PDF/chunking/LLM/retrieval/evidence/RAG/cache/observability/API/lifecycle regression: 469 passed.
- Full suite isolated with `REDIS_ENABLED=false LANGFUSE_ENABLED=false`: 1295 passed
  (W7-M05 baseline: 1074), 10 pre-existing warnings.
- `ruff check .`, `git diff --check`, and `uv lock --check` pass.
- Tests use fakes or an in-process HTTP mock transport; no PDF is downloaded and Docling does not run.
