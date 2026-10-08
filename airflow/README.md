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

- `ARXIV_INGESTION_CATEGORIES` (default `cs.AI,q-bio.QM`)
- `ARXIV_INGESTION_BATCH_SIZE` (default `2` papers per category)
- `ARXIV_INGESTION_SCHEDULE` (default `0 3 * * *`)
- `ARXIV_INGESTION_PROCESS_PDFS` (default `true`)

The current arXiv client supports category filters, not free-text healthcare
queries. The default therefore provides one AI category and one quantitative
biology category without attempting to ingest all of arXiv. The host `data/`
directory is mounted into Airflow so cached PDFs can be reused.
