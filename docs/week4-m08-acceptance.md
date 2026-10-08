# Week 4 final acceptance

## Decision

**READY FOR WEEK 5.** Chunking, persisted embeddings, chunk BM25, vector search, filtered RRF fusion, fallback behavior, and the hybrid API have passed the final acceptance gate. Week 5 can build `query -> hybrid retrieval -> context builder -> Groq -> cited answer` without a retrieval blocker.

The deployment setting is **one API worker**. A recruiter/demo workload does not justify duplicating roughly 1 GiB of loaded model/runtime memory per worker. This minimizes memory use; a worker restart will temporarily make the API unavailable while BGE reloads.

## Accepted architecture

```text
paper raw_text/sections -> deterministic 600-word chunks with 100-word overlap
query -> chunk BM25 -----------------------------------+
      -> normalized BGE-small 384-d -> Lucene HNSW ---+-> RRF(k=60) -> top chunks
```

BM25 searches `paper_title^3`, `section_title^2`, and `chunk_text`. Vector search uses `BAAI/bge-small-en-v1.5` locally on CPU with normalized embeddings. Both paths search the 513-document `arxiv-papers-chunks` index and receive the same category/date filters. Final candidate depth is `min(100, size * 4)`.

RRF uses one-based ranks: `score = sum(1 / (60 + rank))`. Raw BM25 and cosine scores are retained only as diagnostics because their scales and meanings are incompatible. Results are deduplicated by `chunk_id`.

Retrieval modes are `hybrid`, `bm25_fallback`, and `vector_fallback`. One-path failures produce a successful degraded response with `degradation_reason`; two-path failure maps to HTTP 503.

## Corpus and chunk acceptance

All 15 PostgreSQL papers contain `raw_text`, while the schema does not persist structured Docling sections. Therefore all live papers use deterministic raw-text fallback. The policy targets 600 words, prefers at least 100, uses paragraph boundaries where possible, and introduces 100-word overlap only while splitting oversized/fallback content. Structured sections remain preferred when supplied, coherent sections stay intact where sensible, and adjacent small sections may be combined.

| Metric | Live value |
|---|---:|
| papers / distinct chunk arXiv IDs | 15 / 15 |
| chunks | 513 |
| chunks per paper | min 8, max 83, average 34.2 |
| words per chunk | min 100, max 600, average 494.43 |
| overlap-sequence chunks | 498 (97.08%) |

The high overlap flag percentage means most chunks after the first participate in a raw-text sliding sequence; it does **not** mean 97% of the text is duplicated. Every chunk passed checks for contiguous indices, deterministic IDs, non-empty text, and propagated paper metadata.

All 513 embeddings are 384-dimensional. Observed L2 norms ranged from 0.99999991 to 1.00000011 (average 1.00000002), confirming normalization. The Hugging Face cache occupies 635 MiB. Cold initialization was 15.68 seconds in the final benchmark; 32 warm passage embeddings took 11.71 seconds (365.84 ms/passage), and one provider handled the whole process with `model_load_count=1`.

## Vector index

`arxiv-papers-chunks` has `index.knn=true`, one shard, zero replicas, and a 384-dimensional `knn_vector` using Lucene HNSW with cosine similarity. It contains 513 logical documents and occupies 13,845,795 bytes (about 13.2 MiB). Its 1,026 deleted-document tombstones came from earlier delete-and-replace backfills; they are not logical duplicates, did not increase during acceptance, and were not force-merged.

## Retrieval comparison

| Query | Expected topic | BM25 top | Vector top | Hybrid top | BM25/vector top-5 overlap | Observation |
|---|---|---|---|---|---:|---|
| underserved patients | accessible healthcare AI | smart-glasses 028 | smart-glasses 028 | smart-glasses 028 | 2 | both reinforce the same broad health/accessibility result |
| clinical bias | clinical triage fairness | triage 000 | triage 002 | triage 000 | 2 | correct paper; vector chooses the equity passage |
| robot world representations | RoboJEPA / Long-WAM | Long-WAM 000 | RoboJEPA 008 | Long-WAM 000 | 0 | vector is more passage-specific; repeated title metadata drives hybrid |
| low-income student adoption | FGLI LLM adoption | FGLI 000 | FGLI 001 | FGLI 000 | 3 | strong exact-title and semantic agreement |
| smart buildings | cloud/fog/edge building | building 000 | building 001 | building 000 | 3 | both paths identify the correct paper |
| disadvantaged patients | equity and healthcare access | triage 002 | triage 002 | triage 002 | 3 | both paths reinforce the disparities/undertriage passage |

Lexical strength is clearest for the FGLI and smart-building titles. Semantic strength is clearest for the robotics query, where vector search identifies RoboJEPA passages absent from BM25's top five. Agreement is clearest for `2610.01963::chunk::002` on the paraphrased healthcare query.

For the underserved-patients query, chunk `2610.01963::chunk::002` had BM25 rank 2 and vector rank 9:

```text
1/(60+2) + 1/(60+9) = 0.030621785881252923
```

This exactly matched the live fused score. Unit coverage additionally proves one-based ranks, strict deduplication, deterministic ties, and independence from raw scores.

Live `cs.RO` filtering returned only `cs.RO` chunks with candidates from both paths. A `2026-09-28` range on the FGLI query returned only that day's chunks, with both BM25 and vector ranks present.

## API, latency, and workers

Run the standalone demo with:

```bash
uv run python scripts/test_hybrid_search_api.py --base-url http://localhost:8000/api/v1
```

It sends all six representative queries to `POST /api/v1/hybrid-search`, fails on non-2xx responses, and prints retrieval mode plus five ranked chunks.

Twelve warm component benchmarks at candidate depth 20:

| Stage | Average | p50 | p95 | Min-max |
|---|---:|---:|---:|---:|
| BM25 | 27.98 ms | 25.82 | 38.76 | 20.58-49.64 |
| query embedding | 263.35 ms | 201.38 | 495.86 | 52.61-585.21 |
| vector OpenSearch | 24.91 ms | 24.12 | 32.67 | 18.68-33.20 |
| RRF/normalization | 0.88 ms | 0.78 | 1.03 | 0.70-1.75 |
| total | 317.10 ms | 252.76 | 547.28 | 92.66-648.14 |

Twelve warm HTTP requests across two loaded workers averaged 211.53 ms (p50 164.95, observed p95 322.66, range 112.91-391.37). This is acceptable for a recruiter demo and interactive research prototype, but is not a production SLA.

| Workers | Idle worker RSS | Loaded worker RSS | Health / cold request |
|---:|---:|---:|---|
| 1 | 95,196 KiB | 1,019,468 KiB | healthy; demo completed |
| 2 | 95,276 + 95,544 KiB | 1,012,080 + 1,011,984 KiB | healthy; concurrent cold requests 19.53/19.58 s |

The Docker command changed from four workers to one. The host had 11 GiB total and 7.3 GiB available before loaded-worker tests, no swap, and 129 GiB disk available. Four loaded workers would waste approximately 4 GiB on duplicated model/runtime state for little demo benefit.

## Technical debt

All discovered items are **non-blocking for Week 5**:

- PostgreSQL does not persist Docling structured sections, so the live corpus uses raw-text fallback.
- BGE model/runtime memory is duplicated per API process.
- Cold model startup takes tens of seconds and causes a cold-request delay.
- Delete-and-replace indexing leaves Lucene tombstones until normal segment merges reclaim them.
- Earlier discovery-profile precision remains imperfect.
- Repeated paper-title metadata can dominate chunk BM25, demonstrated by the robotics query.
- M05 replacement is not transactional after deletion; a subsequent bulk failure can temporarily leave an incomplete paper.

These should be addressed incrementally, but none prevents context construction over the current stable hybrid results and source metadata.
