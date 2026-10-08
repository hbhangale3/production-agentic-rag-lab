# W4-M06 semantic vector search

## Retrieval pipeline

The read-only retrieval path is:

```text
query -> BGE query embedding -> normalized 384-d vector
      -> OpenSearch Lucene HNSW/cosine search -> ranked chunk hits
```

`VectorSearchService` validates input, asks the shared `EmbeddingProvider` to encode the query, executes a k-NN query only against `arxiv-papers-chunks`, and normalizes hits into chunk-level application models. The query explicitly selects metadata and text fields, so stored embeddings are not returned.

Indexed passages and queries must use the same `BAAI/bge-small-en-v1.5` model because similarity is meaningful only when both are coordinates in the same learned vector space. The provider normalizes vectors to unit length, making cosine similarity stable and allowing the index to compare direction rather than text-length-driven magnitude.

Lucene supports exact metadata filters inside its k-NN clause. This implementation permits only application-built category terms and publication-date ranges; callers cannot submit arbitrary OpenSearch DSL.

Vector scores are useful for diagnostics and ordering within vector retrieval, but they are not on the same scale as BM25 scores. W4-M07 should combine the independent rankings with reciprocal rank fusion (RRF), not raw-score addition.

## Live validation

Validated against OpenSearch 2.19.0 with 513 chunks from 15 papers on 2026-10-08. All observed `section_title` values in these top-five results were absent in the stored documents.

| Query | Rank | Score | arXiv / chunk | Paper | Preview |
|---|---:|---:|---|---|---|
| AI systems helping underserved patients get medical care | 1 | 0.876570 | 2609.19793 / 028 | AI Smart Glasses for Wearable Intelligence | representative workflow across these domains... wearable intelligence... domain-specific constraints |
| | 2 | 0.862943 | 2609.19793 / 006 | AI Smart Glasses for Wearable Intelligence | Application Scenarios... Healthcare... Accessibility |
| | 3 | 0.859118 | 2610.01963 / 001 | Counterfactual Auditing of Bias...Clinical Triage | safety, and equity at scale... clinical triage... privacy-preserving inference |
| | 4 | 0.857752 | 2610.01963 / 004 | Counterfactual Auditing of Bias...Clinical Triage | expert-curated and silver-standard ESI vignettes... |
| | 5 | 0.857322 | 2609.24814 / 002 | Mobile Imaging Solutions for Medical Diagnosis | health issues... cloud computing... |
| bias and fairness in clinical decision making | 1 | 0.877829 | 2610.01963 / 002 | Counterfactual Auditing of Bias...Clinical Triage | mistriage... reduce critical undertriage while improving equity... disparities in triage |
| | 2 | 0.875300 | 2610.10203 / 022 | How to train your model organism | biased-scenario ratio... biased option... counterfactual partition |
| | 3 | 0.872364 | 2610.01963 / 000 | Counterfactual Auditing of Bias...Clinical Triage | paper title and author block |
| | 4 | 0.869146 | 2610.10203 / 007 | How to train your model organism | demographic reasoning shortcuts in clinical QA |
| | 5 | 0.864588 | 2610.10203 / 009 | How to train your model organism | trace the behavior... age-bias... validation criterion |
| robots learning representations of the physical world | 1 | 0.892488 | 2610.10515 / 008 | RoboJEPA: Scaling Robotic Latent World Models | synchronized with actions... multiple cameras... Open-X Embodiment |
| | 2 | 0.891173 | 2610.10528 / 022 | Long-WAM: Scaling the Context of World-Action Models | WorldScape Policy... steerable... |
| | 3 | 0.889689 | 2610.10515 / 002 | RoboJEPA: Scaling Robotic Latent World Models | robot control... scale policies rather than world models |
| | 4 | 0.888009 | 2610.10515 / 050 | RoboJEPA: Scaling Robotic Latent World Models | joint angle control... world model... action-level augmentation |
| | 5 | 0.886921 | 2610.10515 / 007 | RoboJEPA: Scaling Robotic Latent World Models | different views, actions, and states... local attention |
| technology adoption barriers for low income college students | 1 | 0.907837 | 2609.36129 / 001 | Accessible, but Not Adopted | barriers that prevent adoption by FGLI students |
| | 2 | 0.878136 | 2609.36129 / 000 | Accessible, but Not Adopted | first-generation, low-income students... Abstract |
| | 3 | 0.876001 | 2609.36129 / 007 | Accessible, but Not Adopted | intent to learn and use the system... deepen adoption |
| | 4 | 0.873189 | 2609.36129 / 002 | Accessible, but Not Adopted | depth of tool use... FGLI students' LLM adoption |
| | 5 | 0.860470 | 2609.36129 / 006 | Accessible, but Not Adopted | tools despite their availability... concrete value... career relevance |
| smart buildings using cloud and edge computing | 1 | 0.931407 | 2610.01647 / 001 | Towards a Cloud Fog Edge System for Smart Building | electricity use in buildings... energy-intensive computing |
| | 2 | 0.917667 | 2610.01647 / 000 | Towards a Cloud Fog Edge System for Smart Building | paper title and author block |
| | 3 | 0.888824 | 2610.01647 / 010 | Towards a Cloud Fog Edge System for Smart Building | SLA management... deployed within the building... sensors |
| | 4 | 0.879811 | 2610.01647 / 002 | Towards a Cloud Fog Edge System for Smart Building | fast data processing... IoT... devices... exchange data |
| | 5 | 0.873067 | 2610.01647 / 011 | Towards a Cloud Fog Edge System for Smart Building | sensors or microcontrollers... resource-constrained settings |

The rankings are semantically plausible overall. The first healthcare query is diffuse in this small corpus, but the other four strongly concentrate on the expected paper and topic.

### Semantic-versus-lexical observation

For the deliberately paraphrased query `helping disadvantaged patients obtain healthcare`, vector retrieval ranked clinical-triage chunk `2610.01963::chunk::002` first (0.834538); its text discusses undertriage, equity, and disparities without using the query's exact phrase. Independent paper-level BM25 ranked the mobile-imaging paper first and the clinical-triage paper second. This is a modest, genuine semantic win, not proof that vector retrieval is universally better.

### Performance and reuse

Cold local provider initialization was 18,091 ms. Across five warm queries, query embedding averaged 174.58 ms (88.58-343.36), OpenSearch k-NN averaged 19.94 ms (14.55-36.81), and embedding plus search averaged 194.51 ms (103.34-359.82). The same provider served every query and reported `model_load_count == 1`.

Counts remained PostgreSQL papers 15, `arxiv-papers` 15, and `arxiv-papers-chunks` 513. The chunk index had 1,026 pre-existing Lucene tombstones before this validation; searches created none.
