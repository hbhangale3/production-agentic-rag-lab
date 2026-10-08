import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from src.schemas.ingestion import IngestionError, IngestionResult

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
    result = IngestionResult(papers_fetched=1, pdfs_downloaded=1, pdfs_parsed=1, papers_stored=1)
    fetcher = Mock()
    fetcher.fetch_and_process_papers = AsyncMock(return_value=result)
    arxiv_client = Mock()
    arxiv_client.aclose = AsyncMock()
    pdf_parser = Mock()
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

    combined = await dag_module._run_ingestion(Mock(), database, ["cs.AI", "q-bio.QM"], 2, True)

    assert fetcher.fetch_and_process_papers.await_count == 2
    assert fetcher.fetch_and_process_papers.await_args_list[0].kwargs == {
        "max_results": 2,
        "category": "cs.AI",
        "process_pdfs": True,
        "store_to_db": True,
        "db_session": session,
    }
    assert fetcher.fetch_and_process_papers.await_args_list[1].kwargs["category"] == "q-bio.QM"
    arxiv_client.aclose.assert_awaited_once()
    assert session_closed is True
    assert combined.papers_fetched == 2
    assert combined.papers_stored == 2


def test_task_executes_async_service_closes_database_and_logs_errors(monkeypatch, dag_module, caplog) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_categories="cs.AI,q-bio.QM",
        arxiv_ingestion_batch_size=2,
        arxiv_ingestion_process_pdfs=True,
    )
    database = Mock()
    result = IngestionResult(
        papers_fetched=2,
        pdfs_downloaded=1,
        pdfs_parsed=1,
        papers_stored=1,
        processing_time=1.25,
        errors=[IngestionError(arxiv_id="2601.00002", stage="parse", message="bad PDF")],
    )
    async_runner = AsyncMock(return_value=result)
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", async_runner)

    with caplog.at_level(logging.INFO):
        task_result = dag_module.run_ingestion_task()

    database.teardown.assert_called_once()
    async_runner.assert_awaited_once_with(settings, database, ["cs.AI", "q-bio.QM"], 2, True)
    assert task_result["papers_stored"] == 1
    assert "fetched=2" in caplog.text
    assert "arxiv_id=2601.00002" in caplog.text


def test_fetch_failure_marks_airflow_task_failed(monkeypatch, dag_module) -> None:
    settings = SimpleNamespace(
        arxiv_ingestion_categories="cs.AI",
        arxiv_ingestion_batch_size=1,
        arxiv_ingestion_process_pdfs=False,
    )
    database = Mock()
    result = IngestionResult(errors=[IngestionError(stage="fetch", message="API unavailable")])
    monkeypatch.setattr("src.config.Settings", lambda: settings)
    monkeypatch.setattr("src.db.factory.make_database", lambda: database)
    monkeypatch.setattr(dag_module, "_run_ingestion", AsyncMock(return_value=result))

    with pytest.raises(FakeAirflowException, match="metadata fetch failed"):
        dag_module.run_ingestion_task()

    database.teardown.assert_called_once()
