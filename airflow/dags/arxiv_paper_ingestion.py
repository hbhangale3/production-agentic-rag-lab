"""Daily orchestration DAG for the existing arXiv ingestion service."""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DAG_ID = "arxiv_paper_ingestion"
DEFAULT_SCHEDULE = "0 3 * * *"


async def _run_ingestion(settings, database, categories: list[str], batch_size: int, process_pdfs: bool):
    # Keep heavyweight application imports out of Airflow's DAG parse loop.
    from src.schemas.ingestion import IngestionResult
    from src.services.arxiv.factory import make_arxiv_client
    from src.services.metadata_fetcher.factory import make_metadata_fetcher
    from src.services.pdf_parser.factory import make_pdf_parser_service

    arxiv_client = make_arxiv_client(settings)
    pdf_parser = make_pdf_parser_service(settings)
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)
    combined = IngestionResult()

    try:
        with database.get_session() as session:
            for category in categories:
                result = await metadata_fetcher.fetch_and_process_papers(
                    max_results=batch_size,
                    category=category,
                    process_pdfs=process_pdfs,
                    store_to_db=True,
                    db_session=session,
                )
                combined.papers_fetched += result.papers_fetched
                combined.pdfs_downloaded += result.pdfs_downloaded
                combined.pdfs_parsed += result.pdfs_parsed
                combined.papers_stored += result.papers_stored
                combined.processing_time += result.processing_time
                combined.errors.extend(result.errors)
    finally:
        await arxiv_client.aclose()

    return combined


def run_ingestion_task(
    *,
    categories: list[str] | None = None,
    batch_size: int | None = None,
    process_pdfs: bool | None = None,
) -> dict:
    """Synchronous Airflow task boundary for the async ingestion service."""
    from src.config import Settings
    from src.db.factory import make_database

    settings = Settings()
    selected_categories = categories or [
        category.strip() for category in settings.arxiv_ingestion_categories.split(",") if category.strip()
    ]
    selected_batch_size = batch_size if batch_size is not None else settings.arxiv_ingestion_batch_size
    selected_process_pdfs = process_pdfs if process_pdfs is not None else settings.arxiv_ingestion_process_pdfs
    if not selected_categories:
        raise AirflowException("At least one arXiv ingestion category is required")
    if selected_batch_size < 1:
        raise AirflowException("arXiv ingestion batch size must be at least 1")

    database = make_database()
    try:
        result = asyncio.run(
            _run_ingestion(
                settings,
                database,
                selected_categories,
                selected_batch_size,
                selected_process_pdfs,
            )
        )
    finally:
        database.teardown()

    logger.info(
        "Ingestion summary: fetched=%d PDFs_obtained=%d parsed=%d stored=%d duration=%.2fs errors=%d",
        result.papers_fetched,
        result.pdfs_downloaded,
        result.pdfs_parsed,
        result.papers_stored,
        result.processing_time,
        len(result.errors),
    )
    for error in result.errors:
        logger.warning(
            "Ingestion error: arxiv_id=%s stage=%s message=%s",
            error.arxiv_id or "unknown",
            error.stage,
            error.message,
        )

    # A metadata-fetch failure means the scheduled batch did not run. Per-paper
    # errors remain a successful task so later papers retain failure isolation.
    if any(error.stage == "fetch" for error in result.errors):
        raise AirflowException("arXiv metadata fetch failed; see task logs")
    return result.model_dump(mode="json")


default_args = {
    "owner": "rag",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description="Fetch, parse, and persist a small configurable arXiv batch",
    schedule=os.getenv("ARXIV_INGESTION_SCHEDULE", DEFAULT_SCHEDULE),
    catchup=False,
    max_active_runs=1,
    tags=["week2", "arxiv", "ingestion"],
)

ingest_papers = PythonOperator(
    task_id="ingest_arxiv_papers",
    python_callable=run_ingestion_task,
    dag=dag,
)
