# Airflow Configuration

This directory contains Apache Airflow configuration and DAGs for the arXiv Paper Curator project.

## Week 1 Setup

For Week 1, we have a basic setup with:

- **hello_world_dag.py**: Simple test DAG to verify Airflow is working
- **init-db.sql**: Database initialization script

## Directory Structure

```
airflow/
├── README.md           # This file
├── init-db.sql         # Database initialization
└── dags/
    └── hello_world_dag.py  # Test DAG for Week 1
```

## Usage

The Airflow service is configured to run via Docker Compose and can be accessed at:
- Web UI: http://localhost:8080
- Default credentials: admin/[auto-generated password]

## Future Weeks

In later weeks, this directory will contain:
- arXiv paper fetching DAGs
- PDF processing workflows
- Data pipeline orchestration

## Week 2 paper ingestion

The `arxiv_paper_ingestion` DAG runs the existing application ingestion service
once per day at 03:00 UTC and can also be triggered manually. Historical runs
are not backfilled.

Its policy is controlled through environment variables:

- `ARXIV_INGESTION_PROFILES` (default `ai,healthcare_ai,health_equity_tech`)
- `ARXIV_INGESTION_BATCH_SIZE` (default `1` paper per profile)
- `ARXIV_INGESTION_SCHEDULE` (default `0 3 * * *`)
- `ARXIV_INGESTION_PROCESS_PDFS` (default `true`)

Discovery policy lives in `src/services/arxiv/discovery_profiles.py`. The three
defaults cover general AI, healthcare AI, and health-equity technology research.
Profiles run sequentially. A paper matching multiple profiles remains one row
because PostgreSQL upserts on `arxiv_id`; the PDF cache avoids duplicate downloads,
but a cached PDF may still be parsed again if profiles overlap within one run.

After each profile commits its papers to PostgreSQL, the DAG reloads those
canonical rows and indexes them in OpenSearch using `arxiv_id` as the document
ID. PostgreSQL remains the source of truth. Per-paper work remains isolated, but
the Airflow task fails after processing when any ingestion or indexing failure
was recorded, so partial failures are visible in the final task state.

The arXiv client supports both category filters and application-owned advanced
queries. The host `data/` directory is mounted into Airflow so cached PDFs can
be reused.
