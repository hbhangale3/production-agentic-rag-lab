"""Strict parsing for the single-field JSON contracts used by agent LLM steps."""

import json
import re

_FENCED_JSON = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL)


class StructuredOutputError(ValueError):
    """Model output did not match the exact single-field JSON contract."""


class ScoreOutputError(StructuredOutputError):
    """Model output did not match the exact ``{"score": <0-100>}`` contract."""


def _parse_single_field(content: object, field: str, error: type[StructuredOutputError]) -> object:
    if not isinstance(content, str) or not content.strip():
        raise error("response was empty")
    candidate = content.strip()
    fenced = _FENCED_JSON.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise error("response was not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {field}:
        raise error(f"response must contain only {field}")
    return payload[field]


def parse_score_object(content: object) -> int:
    """Return the validated score, tolerating only one surrounding code fence."""

    score = _parse_single_field(content, "score", ScoreOutputError)
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ScoreOutputError("score must be an integer between 0 and 100")
    return score


def parse_query_object(content: object) -> str:
    """Return the raw string from ``{"query": "..."}`` under the same fence policy."""

    query = _parse_single_field(content, "query", StructuredOutputError)
    if not isinstance(query, str):
        raise StructuredOutputError("query must be a string")
    return query


def parse_string_list_object(content: object, field: str) -> list[str]:
    """Return the list from ``{"<field>": ["...", ...]}`` under the same fence policy."""

    values = _parse_single_field(content, field, StructuredOutputError)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise StructuredOutputError(f"{field} must be a list of strings")
    return values
