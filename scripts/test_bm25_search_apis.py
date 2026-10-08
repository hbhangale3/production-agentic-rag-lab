#!/usr/bin/env python3
"""Run the Week 3 BM25 API demo against a live FastAPI application."""

import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx

SEPARATOR = "=" * 60
DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_RESULT_SIZE = 5


@dataclass(frozen=True)
class Strategy:
    name: str
    path: str
    params: dict[str, Any]
    explanation: tuple[str, ...]


def _strategies(result_size: int) -> list[Strategy]:
    return [
        Strategy(
            "SIMPLE SEARCH",
            "/api/v1/search/simple",
            {"q": "artificial intelligence healthcare", "size": result_size},
            ("Purpose: broad BM25 retrieval across the default searchable fields.",),
        ),
        Strategy(
            "MATCH SEARCH",
            "/api/v1/search/match",
            {"q": "clinical decision support", "field": "abstract", "size": result_size},
            ("Field: abstract", "Purpose: field-specific matching."),
        ),
        Strategy(
            "MULTI-MATCH SEARCH",
            "/api/v1/search/multi-match",
            {
                "q": "large language models healthcare",
                "fields": ["title", "abstract", "raw_text"],
                "size": result_size,
            },
            ("Fields: title, abstract, raw_text", "Purpose: one query across selected fields."),
        ),
        Strategy(
            "BOOSTING SEARCH",
            "/api/v1/search/boosting",
            {
                "positive": "artificial intelligence healthcare",
                "negative": "survey",
                "negative_boost": 0.2,
                "size": result_size,
            },
            (
                "Positive query: artificial intelligence healthcare",
                "Negative query: survey",
                "Negative boost: 0.2",
            ),
        ),
        Strategy(
            "FILTERED SEARCH",
            "/api/v1/search/filter",
            {"q": "machine learning", "category": "cs.AI", "size": result_size},
            ("Query: machine learning", "Category filter: cs.AI"),
        ),
        Strategy(
            "SORTED SEARCH",
            "/api/v1/search/sort",
            {
                "q": "artificial intelligence",
                "sort_by": "published_date",
                "sort_order": "desc",
                "size": result_size,
            },
            (
                "Query: artificial intelligence",
                "Sort: published_date DESC",
                "Scores may be null when OpenSearch performs explicit field sorting.",
            ),
        ),
        Strategy(
            "COMBINED SEARCH",
            "/api/v1/search/combined",
            {
                "q": "healthcare artificial intelligence",
                "category": "cs.AI",
                "published_from": "2026-01-01",
                "preferred_category": "cs.CL",
                "sort_by": "relevance",
                "sort_order": "desc",
                "size": result_size,
            },
            (
                "Query: healthcare artificial intelligence",
                "Category filter: cs.AI",
                "Published from: 2026-01-01",
                "Preferred category: cs.CL",
                "Sort: relevance DESC",
            ),
        ),
    ]


def _format_score(score: float | None) -> str:
    return "null" if score is None else f"{score:.6f}"


def _preview(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def _print_result(result: dict[str, Any]) -> None:
    print(f"Successful: {result.get('successful')}")
    print(f"Total matches: {result.get('total', 0)}")
    print(f"Took: {result.get('took_ms')} ms")
    print("\nResults:")
    hits = result.get("hits", [])
    if not hits:
        print("No matching papers.")
        return

    for rank, hit in enumerate(hits, start=1):
        print(f"\n#{rank}")
        print(f"arXiv ID: {hit.get('arxiv_id', '')}")
        print(f"Score: {_format_score(hit.get('score'))}")
        print(f"Title: {hit.get('title', '')}")
        print(f"Authors: {', '.join(hit.get('authors') or [])}")
        print(f"Categories: {', '.join(hit.get('categories') or [])}")
        print(f"Published: {hit.get('published_date') or 'unknown'}")
        print(f"Parser: {hit.get('parser_used') or 'unknown'}")
        print(f"Pages: {hit.get('page_count') if hit.get('page_count') is not None else 'unknown'}")
        if hit.get("abstract"):
            print(f"Abstract: {_preview(hit['abstract'])}")


def _request(
    client: httpx.Client,
    path: str,
    params: dict[str, Any],
    *,
    display: bool = True,
) -> tuple[bool, dict[str, Any] | None]:
    try:
        response = client.get(path, params=params)
    except httpx.RequestError as exc:
        print(f"HTTP request failed: {exc}")
        return False, None

    if display:
        print(f"\nRequest:\nGET {path}")
        print("\nParameters:")
        for key, value in params.items():
            rendered = ", ".join(value) if isinstance(value, list) else value
            print(f"{key} = {rendered}")
        print(f"\nHTTP Status: {response.status_code}")

    try:
        body = response.json()
    except ValueError:
        if display:
            print(f"API returned a non-JSON response: {_preview(response.text)}")
        return False, None

    if response.status_code != httpx.codes.OK:
        if display:
            print(f"API error: {body}")
        return False, body

    if display:
        _print_result(body)
    return bool(body.get("successful")), body


def _ranking(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    return [str(hit.get("arxiv_id", "")) for hit in result.get("hits", [])]


def _print_comparisons(
    client: httpx.Client,
    result_size: int,
    boosted: dict[str, Any] | None,
    sorted_result: dict[str, Any] | None,
) -> bool:
    print(f"\n{SEPARATOR}\nSTRATEGY COMPARISONS\n{SEPARATOR}")
    baseline_ok, baseline = _request(
        client,
        "/api/v1/search/simple",
        {"q": "artificial intelligence healthcare", "size": result_size},
        display=False,
    )
    relevance_ok, relevance = _request(
        client,
        "/api/v1/search/simple",
        {"q": "artificial intelligence", "size": result_size},
        display=False,
    )

    print("\nBaseline vs negative boosting")
    print(f"Baseline IDs: {_ranking(baseline)}")
    print(f"Boosted IDs:  {_ranking(boosted)}")
    print("Ranking changed:", _ranking(baseline) != _ranking(boosted))

    print("\nBM25 relevance vs publication-date sorting")
    print(f"Relevance IDs: {_ranking(relevance)}")
    print(f"Date DESC IDs: {_ranking(sorted_result)}")
    print("Ordering changed:", _ranking(relevance) != _ranking(sorted_result))
    return baseline_ok and relevance_ok


def main() -> int:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    try:
        result_size = int(os.getenv("RESULT_SIZE", str(DEFAULT_RESULT_SIZE)))
    except ValueError:
        print("ERROR: RESULT_SIZE must be an integer.")
        return 2
    if not 1 <= result_size <= 100:
        print("ERROR: RESULT_SIZE must be between 1 and 100.")
        return 2

    print(SEPARATOR)
    print("WEEK 3 BM25 SEARCH API DEMO")
    print(SEPARATOR)
    print(f"\nAPI:\n{api_base_url}")
    print("\nCorpus populated through Airflow; individual searches report totals.")

    with httpx.Client(base_url=api_base_url, timeout=30.0) as client:
        try:
            health = client.get("/api/v1/ping")
            health.raise_for_status()
        except httpx.HTTPError:
            print(f"\nERROR: FastAPI is not reachable at {api_base_url}")
            return 2

        passed = 0
        results: dict[str, dict[str, Any] | None] = {}
        strategies = _strategies(result_size)
        for number, strategy in enumerate(strategies, start=1):
            print(f"\n{SEPARATOR}\n{number}. {strategy.name}\n{SEPARATOR}")
            for line in strategy.explanation:
                print(line)
            successful, result = _request(client, strategy.path, strategy.params)
            results[strategy.name] = result
            passed += int(successful)

        comparisons_ok = _print_comparisons(
            client,
            result_size,
            results.get("BOOSTING SEARCH"),
            results.get("SORTED SEARCH"),
        )

    failed = len(strategies) - passed
    print(f"\n{SEPARATOR}\nDEMO SUMMARY\n{SEPARATOR}")
    print(f"Passed strategies: {passed}/{len(strategies)}")
    print(f"Failed strategies: {failed}/{len(strategies)}")
    if not comparisons_ok:
        print("One or more supplemental comparison requests failed.")
    return 0 if failed == 0 and comparisons_ok else 1


if __name__ == "__main__":
    sys.exit(main())
