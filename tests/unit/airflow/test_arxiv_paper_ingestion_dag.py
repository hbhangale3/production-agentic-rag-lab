import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from src.schemas.discovery import DiscoveryProfileResult, DiscoveryRunResult
from src.schemas.indexing import PaperIndexingError, PaperIndexingResult
from src.schemas.ingestion import IngestionError, IngestionResult
from src.services.arxiv.discovery_profiles import get_discovery_profiles

DAG_PATH = Path(__file__).parents[3] / "airflow" / "dags" / "arxiv_paper_ingestion.py"


class FakeDAG:
    def __init__(self, dag_id, **kwargs):
        self.dag_id = dag_id
        self.schedule = kwargs.get("schedule")
        self.catchup = kwargs.get("catchup")
        self.max_active_runs = kwargs.get("max_active_runs")
        self.task_dict = {}


class FakePythonOperator:
    def __init__(self, *, task_id, python_callable, dag):
        self.task_id = task_id
        self.python_callable = python_callable
        dag.task_dict[task_id] = self


class FakeAirflowException(Exception):
    pass


@pytest.fixture
def dag_module(monkeypatch):
    airflow_module = ModuleType("airflow")
    airflow_module.DAG = FakeDAG
    exceptions_module = ModuleType("airflow.exceptions")
    exceptions_module.AirflowException = FakeAirflowException
    operators_module = ModuleType("airflow.operators")
    python_module = ModuleType("airflow.operators.python")
    python_module.PythonOperator = FakePythonOperator
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.exceptions", exceptions_module)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators_module)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", python_module)

    spec = importlib.util.spec_from_file_location("test_arxiv_paper_ingestion_dag", DAG_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dag_imports_with_expected_configuration(dag_module) -> None:
    assert dag_module.dag.dag_id == "arxiv_paper_ingestion"
    assert dag_module.dag.schedule == "0 3 * * *"
    assert dag_module.dag.catchup is False
    assert dag_module.dag.max_active_runs == 1
    assert set(dag_module.dag.task_dict) == {"ingest_arxiv_papers"}
    assert dag_module.dag.task_dict["ingest_arxiv_papers"].python_callable is dag_module.run_ingestion_task


@pytest.mark.anyio
async def test_async_runner_forwards_policy_and_closes_client(monkeypatch, dag_module) -> None:
    result = IngestionResult(
        papers_fetched=1,
        pdfs_downloaded=1,
        pdfs_parsed=1,
        papers_stored=1,
        stored_arxiv_ids=["2601.00001"],
    )
    fetcher = Mock()
    fetcher.fetch_and_process_papers = AsyncMock(return_value=result)
    arxiv_client = Mock()
    arxiv_client.aclose = AsyncMock()
    pdf_parser = Mock()
    opensearch_client = Mock()
    indexing_service = Mock()
    indexing_service.index_papers.return_value = PaperIndexingResult(attempted=1, indexed=1)
    persisted_paper = SimpleNamespace(arxiv_id="2601.00001")
    repository = Mock()
    repository.get_by_arxiv_id.return_value = persisted_paper
    session = Mock()
    session_closed = False

    @contextmanager
    def get_session():
        nonlocal session_closed
        try:
            yield session
        finally:
            session_closed = True

    database = Mock()
    database.get_session = get_session
    monkeypatch.setattr("src.services.arxiv.factory.make_arxiv_client", lambda _settings: arxiv_client)
    monkeypatch.setattr("src.services.pdf_parser.factory.make_pdf_parser_service", lambda _settings: pdf_parser)
    monkeypatch.setattr("src.services.metadata_fetcher.factory.make_metadata_fetcher", lambda *_args: fetcher)
    monkeypatch.setattr("src.services.opensearch.factory.make_opensearch_client", lambda _settings: opensearch_client)
    monkeypatch.setattr("src.services.indexing.paper_service.PaperIndexingService", lambda _client: indexing_service)
    monkeypatch.setattr("src.repositories.paper.PaperRepository", lambda _session: repository)

    profiles = get_discovery_profiles(["ai", "healthcare_ai"])
    combined = await dag_module._run_ingestion(Mock(), database, profiles, 2, True)

    assert fetcher.fetch_and_process_papers.await_count == 2
    assert fetcher.fetch_and_process_papers.await_args_list[0].kwargs == {
        "max_results": 2,
        "category": "cs.AI",
        "search_query": None,
        "process_pdfs": True,
        "store_to_db": True,
        "db_session": session,
    }
    assert fetcher.fetch_and_process_papers.await_args_list[1].kwargs["category"] is None
    assert fetcher.fetch_and_process_papers.await_args_list[1].kwargs["search_query"] == profiles[1].search_query
    arxiv_client.aclose.assert_awaited_once()
    opensearch_client.close.assert_called_once_with()
    assert indexing_service.index_papers.call_count == 2
    indexing_service.index_papers.assert_called_with([persisted_paper])
    assert session_closed is True
    assert combined.aggregate.papers_fetched == 2
    assert combined.aggregate.papers_stored == 2
    assert combined.indexing.indexed == 2
    assert combined.failed == 0
    assert [item.profile_name for item in combined.profiles] == ["ai", "healthcare_ai"]


@pytest.mark.anyio
async def test_persistence_failure_is_not_indexed_and_later_profile_continues(monkeypatch, dag_module) -> None:
    failed = IngestionResult(
        papers_fetched=1,
        errors=[IngestionError(arxiv_id="2601.00001", stage="store", message="database error")],
    )
    succeeded = IngestionResult(
        papers_fetched=1,
        papers_stored=1,
        stored_arxiv_ids=["2601.00002"],
    )
    fetcher = Mock()
    fetcher.fetch_and_process_papers = AsyncMock(side_effect=[failed, succeeded])
    arxiv_client = Mock(aclose=AsyncMock())
    opensearch_client = Mock()
    indexing_service = Mock()
    indexing_service.index_papers.return_value = PaperIndexingResult(attempted=1, indexed=1)
    persisted_paper = SimpleNamespace(arxiv_id="2601.00002")
    repository = Mock()
    repository.get_by_arxiv_id.return_value = persisted_paper

    @contextmanager
    def get_session():
        yield Mock()

    database = Mock(get_session=get_session)
    monkeypatch.setattr("src.services.arxiv.factory.make_arxiv_client", lambda _settings: arxiv_client)
    monkeypatch.setattr("src.services.pdf_parser.factory.make_pdf_parser_service", lambda _settings: Mock())
    monkeypatch.setattr("src.services.metadata_fetcher.factory.make_metadata_fetcher", lambda *_args: fetcher)
    monkeypatch.setattr("src.services.opensearch.factory.make_opensearch_client", lambda _settings: opensearch_client)
    monkeypatch.setattr("src.services.indexing.paper_service.PaperIndexingService", lambda _client: indexing_service)
    monkeypatch.setattr("src.repositories.paper.PaperRepository", lambda _session: repository)

    result = await dag_module._run_ingestion(
        Mock(), database, get_discovery_profiles(["ai", "healthcare_ai"]), 1, False
    )

    indexing_service.index_papers.assert_called_once_with([persisted_paper])
    repository.get_by_arxiv_id.assert_called_once_with("2601.00002")
    assert result.aggregate.papers_stored == 1
    assert result.indexing.indexed == 1
    assert result.failed == 1


def test_task_fails_after_partial_error_and_closes_database(monkeypatch, dag_module, caplog) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_profiles="ai,healthcare_ai",
        arxiv_ingestion_batch_size=2,
        arxiv_ingestion_process_pdfs=True,
    )
    database = Mock()
    ingestion_result = IngestionResult(
        papers_fetched=2, pdfs_downloaded=1, pdfs_parsed=1, papers_stored=1, processing_time=1.25,
        errors=[IngestionError(arxiv_id="2601.00002", stage="parse", message="bad PDF")],
    )
    result = DiscoveryRunResult(
        profiles=[DiscoveryProfileResult(profile_name="ai", result=ingestion_result)],
        aggregate=ingestion_result,
        failed=1,
    )
    async_runner = AsyncMock(return_value=result)
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", async_runner)

    with caplog.at_level(logging.INFO), pytest.raises(FakeAirflowException, match="1 failure"):
        dag_module.run_ingestion_task()

    database.teardown.assert_called_once()
    selected = get_discovery_profiles(["ai", "healthcare_ai"])
    async_runner.assert_awaited_once_with(settings, database, selected, 2, True)
    assert "fetched=2" in caplog.text
    assert "arxiv_id=2601.00002" in caplog.text


def test_manual_dag_run_config_controls_safe_ingestion(monkeypatch, dag_module) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_profiles="ai,healthcare_ai,health_equity_tech",
        arxiv_ingestion_batch_size=2,
        arxiv_ingestion_process_pdfs=True,
    )
    database = Mock()
    result = DiscoveryRunResult()
    async_runner = AsyncMock(return_value=result)
    dag_run = SimpleNamespace(
        conf={"profile_names": ["health_equity_tech"], "batch_size": 1, "process_pdfs": False}
    )
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", async_runner)

    task_result = dag_module.run_ingestion_task(dag_run=dag_run)

    async_runner.assert_awaited_once_with(
        settings,
        database,
        get_discovery_profiles(["health_equity_tech"]),
        1,
        False,
    )
    assert task_result["failed"] == 0
    database.teardown.assert_called_once()


def test_all_profile_fetch_failures_mark_airflow_task_failed(monkeypatch, dag_module) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_profiles="ai",
        arxiv_ingestion_batch_size=1,
        arxiv_ingestion_process_pdfs=False,
    )
    database = Mock()
    failed = IngestionResult(errors=[IngestionError(stage="fetch", message="API unavailable")])
    result = DiscoveryRunResult(
        profiles=[DiscoveryProfileResult(profile_name="ai", result=failed)], aggregate=failed, failed=1
    )
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", AsyncMock(return_value=result))

    with pytest.raises(FakeAirflowException, match="1 failure"):
        dag_module.run_ingestion_task()

    database.teardown.assert_called_once()


def test_one_profile_fetch_failure_allows_later_work_then_fails_task(monkeypatch, dag_module) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_profiles="ai,healthcare_ai",
        arxiv_ingestion_batch_size=1,
        arxiv_ingestion_process_pdfs=False,
    )
    database = Mock()
    failed = IngestionResult(errors=[IngestionError(stage="fetch", message="temporary")])
    succeeded = IngestionResult(papers_fetched=1, papers_stored=1)
    aggregate = IngestionResult(papers_fetched=1, papers_stored=1, errors=failed.errors)
    result = DiscoveryRunResult(
        profiles=[
            DiscoveryProfileResult(profile_name="ai", result=failed),
            DiscoveryProfileResult(profile_name="healthcare_ai", result=succeeded),
        ],
        aggregate=aggregate,
        failed=1,
    )
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", AsyncMock(return_value=result))

    with pytest.raises(FakeAirflowException, match="1 failure"):
        dag_module.run_ingestion_task()

    database.teardown.assert_called_once()


def test_indexing_failure_preserves_persistence_and_fails_task(monkeypatch, dag_module) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_profiles="ai",
        arxiv_ingestion_batch_size=1,
        arxiv_ingestion_process_pdfs=False,
    )
    database = Mock()
    ingestion = IngestionResult(
        papers_fetched=1,
        papers_stored=1,
        stored_arxiv_ids=["2601.00001"],
    )
    indexing = PaperIndexingResult(
        attempted=1,
        failed=1,
        errors=[PaperIndexingError(arxiv_id="2601.00001", message="OpenSearch indexing failed")],
    )
    result = DiscoveryRunResult(
        profiles=[DiscoveryProfileResult(profile_name="ai", result=ingestion, indexing=indexing)],
        aggregate=ingestion,
        indexing=indexing,
        failed=1,
    )
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", AsyncMock(return_value=result))

    with pytest.raises(FakeAirflowException, match="1 failure"):
        dag_module.run_ingestion_task()

    assert result.aggregate.papers_stored == 1
    assert result.indexing.indexed == 0
    database.teardown.assert_called_once()
