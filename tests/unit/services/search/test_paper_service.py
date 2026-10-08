from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from src.services.search.paper_service import PaperSearchService


def raw_response() -> dict:
    return {
        "took": 7,
        "hits": {
            "total": {"value": 1, "relation": "eq"},
            "hits": [
                {
                    "_id": "2610.01963",
                    "_score": 4.25,
                    "_source": {
                        "arxiv_id": "2610.01963",
                        "title": "Clinical Triage",
                        "authors": ["Ada Lovelace"],
                        "abstract": "Clinical AI abstract.",
                        "categories": ["cs.AI"],
                        "published_date": "2026-10-02T12:30:00Z",
                    },
                }
            ],
        },
    }


def test_simple_search_executes_dsl_and_normalizes_response() -> None:
    client = Mock()
    client.search.return_value = raw_response()

    result = PaperSearchService(client).search_simple("clinical triage", size=3)

    body = client.search.call_args.args[0]
    assert body["query"]["multi_match"]["query"] == "clinical triage"
    assert body["size"] == 3
    assert result.query == "clinical triage"
    assert result.total == 1
    assert result.took_ms == 7
    assert result.hits[0].arxiv_id == "2610.01963"
    assert result.hits[0].score == 4.25
    assert result.hits[0].published_date == datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("method", "kwargs", "path"),
    [
        ("search_multi_match", {"fields": ["title", "authors"]}, ("multi_match",)),
        ("search_boosting", {"negative_query": "survey"}, ("boosting",)),
        ("search_filtered", {"categories": ["cs.AI"]}, ("bool",)),
        ("search_sorted", {"sort_by": "published_date"}, ("multi_match",)),
        ("search_combined", {"categories": ["cs.AI"], "preferred_category": "cs.CL"}, ("bool",)),
    ],
)
def test_named_strategy_executes_expected_query(method: str, kwargs: dict, path: tuple[str, ...]) -> None:
    client = Mock()
    client.search.return_value = {"hits": {"total": 0, "hits": []}}
    service = PaperSearchService(client)

    result = getattr(service, method)("machine learning", **kwargs)

    query = client.search.call_args.args[0]["query"]
    assert path[0] in query
    assert result.total == 0
    assert result.hits == []
    assert result.successful is True


def test_missing_optional_source_fields_and_score_are_safe() -> None:
    client = Mock()
    client.search.return_value = {
        "hits": {"total": {"value": 1}, "hits": [{"_id": "2610.10393", "_source": {"title": "Minimal"}}]}
    }

    hit = PaperSearchService(client).search_simple("minimal").hits[0]

    assert hit.arxiv_id == "2610.10393"
    assert hit.authors == []
    assert hit.categories == []
    assert hit.published_date is None
    assert hit.score is None


def test_search_execution_failure_is_normalized() -> None:
    client = Mock()
    client.search.return_value = None

    result = PaperSearchService(client).search_simple("clinical")

    assert result.successful is False
    assert result.error == "OpenSearch search failed"
    assert result.total == 0
    assert result.hits == []
