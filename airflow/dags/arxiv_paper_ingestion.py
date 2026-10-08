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


async def _run_ingestion(settings, database, profiles, batch_size: int, process_pdfs: bool):
    # Keep heavyweight application imports out of Airflow's DAG parse loop.
    from src.repositories.paper import PaperRepository
    from src.schemas.discovery import DiscoveryProfileResult, DiscoveryRunResult
    from src.schemas.indexing import PaperIndexingError, PaperIndexingResult
    from src.services.arxiv.factory import make_arxiv_client
    from src.services.indexing.paper_service import PaperIndexingService
    from src.services.metadata_fetcher.factory import make_metadata_fetcher
    from src.services.opensearch.factory import make_opensearch_client
    from src.services.pdf_parser.factory import make_pdf_parser_service

    arxiv_client = make_arxiv_client(settings)
    pdf_parser = make_pdf_parser_service(settings)
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)
    opensearch_client = make_opensearch_client(settings)
    indexing_service = PaperIndexingService(opensearch_client)
    run_result = DiscoveryRunResult()

    try:
        with database.get_session() as session:
            for profile in profiles:
                result = await metadata_fetcher.fetch_and_process_papers(
                    max_results=batch_size,
                    category=profile.category,
                    search_query=profile.search_query,
                    process_pdfs=process_pdfs,
                    store_to_db=True,
                    db_session=session,
                )
                repository = PaperRepository(session)
                indexing = PaperIndexingResult(attempted=len(result.stored_arxiv_ids))
                persisted_papers = []
                for arxiv_id in result.stored_arxiv_ids:
                    paper = repository.get_by_arxiv_id(arxiv_id)
                    if paper is None:
                        indexing.failed += 1
                        indexing.errors.append(
                            PaperIndexingError(
                                arxiv_id=arxiv_id,
                                message="Persisted paper could not be reloaded for indexing",
                            )
                        )
                    else:
                        persisted_papers.append(paper)
                if persisted_papers:
                    batch_indexing = indexing_service.index_papers(persisted_papers)
                    indexing.indexed += batch_indexing.indexed
                    indexing.failed += batch_indexing.failed
                    indexing.errors.extend(batch_indexing.errors)
                profile_result = DiscoveryProfileResult(
                    profile_name=profile.name,
                    result=result,
                    indexing=indexing,
                )
                run_result.indexing.attempted += indexing.attempted
                run_result.indexing.indexed += indexing.indexed
                run_result.indexing.failed += indexing.failed
                run_result.indexing.errors.extend(indexing.errors)
                run_result.failed += len(result.errors) + profile_result.indexing.failed
                run_result.profiles.append(profile_result)
                combined = run_result.aggregate
                combined.papers_fetched += result.papers_fetched
                combined.pdfs_downloaded += result.pdfs_downloaded
                combined.pdfs_parsed += result.pdfs_parsed
                combined.papers_stored += result.papers_stored
                combined.stored_arxiv_ids.extend(result.stored_arxiv_ids)
                combined.processing_time += result.processing_time
                combined.errors.extend(result.errors)
                logger.info(
                    "Profile summary: profile=%s fetched=%d PDFs_obtained=%d parsed=%d stored=%d indexed=%d failed=%d",
                    profile.name,
                    result.papers_fetched,
                    result.pdfs_downloaded,
                    result.pdfs_parsed,
                    result.papers_stored,
                    profile_result.indexing.indexed,
                    len(result.errors) + profile_result.indexing.failed,
                )
    finally:
        await arxiv_client.aclose()
        opensearch_client.close()

    return run_result


def run_ingestion_task(
    *,
    profile_names: list[str] | None = None,
    batch_size: int | None = None,
    process_pdfs: bool | None = None,
    dag_run=None,
) -> dict:
    """Synchronous Airflow task boundary for the async ingestion service."""
    from src.config import Settings
    from src.db.factory import make_database
    from src.services.arxiv.discovery_profiles import get_discovery_profiles

    settings = Settings()
    run_config = dag_run.conf if dag_run is not None else {}
    profile_names = profile_names or run_config.get("profile_names")
    batch_size = batch_size if batch_size is not None else run_config.get("batch_size")
    process_pdfs = process_pdfs if process_pdfs is not None else run_config.get("process_pdfs")
    selected_profiles = get_discovery_profiles(profile_names or settings.arxiv_ingestion_profiles)
    selected_batch_size = batch_size if batch_size is not None else settings.arxiv_ingestion_batch_size
    selected_process_pdfs = process_pdfs if process_pdfs is not None else settings.arxiv_ingestion_process_pdfs
    if not selected_profiles:
        raise AirflowException("At least one arXiv discovery profile is required")
    if selected_batch_size < 1:
        raise AirflowException("arXiv ingestion batch size must be at least 1")

    database = make_database()
    try:
        run_result = asyncio.run(
            _run_ingestion(
                settings,
                database,
                selected_profiles,
                selected_batch_size,
                selected_process_pdfs,
            )
        )
    finally:
        database.teardown()

    result = run_result.aggregate
    logger.info(
        "Ingestion summary: fetched=%d PDFs_obtained=%d parsed=%d stored=%d indexed=%d failed=%d "
        "duration=%.2fs",
        result.papers_fetched,
        result.pdfs_downloaded,
        result.pdfs_parsed,
        result.papers_stored,
        run_result.indexing.indexed,
        run_result.failed,
        result.processing_time,
    )
    for error in result.errors:
        logger.warning(
            "Ingestion error: arxiv_id=%s stage=%s message=%s",
            error.arxiv_id or "unknown",
            error.stage,
            error.message,
        )

    for error in run_result.indexing.errors:
        logger.warning(
            "Indexing error: arxiv_id=%s message=%s",
            error.arxiv_id or "unknown",
            error.message,
        )
    if run_result.failed > 0:
        raise AirflowException(f"Ingestion completed with {run_result.failed} failure(s); see task logs")
    return run_result.model_dump(mode="json")


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
    description="Discover, parse, and persist small configurable arXiv research batches",
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
