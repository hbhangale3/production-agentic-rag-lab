from datetime import date

import pytest
from src.services.opensearch.query_builder import (
    DEFAULT_SEARCH_FIELDS,
    build_boosting_query,
    build_combined_query,
    build_filtered_query,
    build_match_query,
    build_multi_match_query,
    build_simple_query,
    build_sorted_query,
)


def test_simple_query_uses_weighted_bm25_fields() -> None:
    body = build_simple_query("clinical triage", size=5)

    multi_match = body["query"]["multi_match"]
    assert multi_match == {
        "query": "clinical triage",
        "fields": list(DEFAULT_SEARCH_FIELDS),
        "type": "best_fields",
    }
    assert body["size"] == 5
    assert "raw_text" not in body["_source"]


def test_match_query_targets_allowed_single_field() -> None:
    assert build_match_query("counterfactual", field="title")["query"] == {
        "match": {"title": "counterfactual"}
    }


def test_multi_match_uses_explicit_allowed_fields() -> None:
    clause = build_multi_match_query("language model", fields=["title^4", "authors"])["query"]["multi_match"]

    assert clause["query"] == "language model"
    assert clause["fields"] == ["title^4", "authors"]
    assert clause["type"] == "best_fields"


def test_boosting_query_contains_positive_negative_and_boost() -> None:
    boosting = build_boosting_query("clinical", "survey", negative_boost=0.3)["query"]["boosting"]

    assert boosting["positive"]["multi_match"]["query"] == "clinical"
    assert boosting["negative"]["multi_match"]["query"] == "survey"
    assert boosting["negative_boost"] == 0.3


def test_filtered_query_keeps_keywords_scoring_and_filters_non_scoring() -> None:
    body = build_filtered_query(
        "machine learning",
        categories=["cs.AI", "cs.CL"],
        published_from=date(2026, 1, 1),
        published_to=date(2026, 12, 31),
    )
    bool_query = body["query"]["bool"]

    assert bool_query["must"][0]["multi_match"]["query"] == "machine learning"
    assert {"terms": {"categories": ["cs.AI", "cs.CL"]}} in bool_query["filter"]
    assert {"range": {"published_date": {"gte": "2026-01-01", "lte": "2026-12-31"}}} in bool_query[
        "filter"
    ]


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "expected_field"),
    [("relevance", "desc", "_score"), ("published_date", "asc", "published_date"), ("published_date", "desc", "published_date")],
)
def test_sorted_query_uses_safe_sort_options(sort_by: str, sort_order: str, expected_field: str) -> None:
    body = build_sorted_query("risk control", sort_by=sort_by, sort_order=sort_order)

    assert body["sort"] == [{expected_field: {"order": sort_order}}]


def test_combined_query_composes_must_filter_should_and_sort() -> None:
    body = build_combined_query(
        "healthcare",
        categories=["cs.AI"],
        preferred_category="cs.CL",
        sort_by="published_date",
        sort_order="desc",
    )
    bool_query = body["query"]["bool"]

    assert bool_query["must"][0]["multi_match"]["query"] == "healthcare"
    assert bool_query["filter"] == [{"terms": {"categories": ["cs.AI"]}}]
    assert bool_query["should"] == [{"term": {"categories": {"value": "cs.CL", "boost": 2.0}}}]
    assert body["sort"] == [{"published_date": {"order": "desc"}}]


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (lambda: build_simple_query("  "), "query cannot be blank"),
        (lambda: build_simple_query("ai", size=0), "size must be between"),
        (lambda: build_multi_match_query("ai", fields=["secret_field"]), "unsupported searchable field"),
        (lambda: build_sorted_query("ai", sort_by="unknown"), "sort_by"),
        (lambda: build_sorted_query("ai", sort_order="sideways"), "sort_order"),
        (lambda: build_boosting_query("ai", "survey", negative_boost=1.1), "negative_boost"),
    ],
)
def test_invalid_inputs_are_rejected(builder, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        builder()
