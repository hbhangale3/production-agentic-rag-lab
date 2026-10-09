"""Strict parsing for the single-score JSON contract shared by agent graders."""

import json
import re

_FENCED_JSON = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL)


class ScoreOutputError(ValueError):
    """Model output did not match the exact ``{"score": <0-100>}`` contract."""


def parse_score_object(content: object) -> int:
    """Return the validated score, tolerating only one surrounding code fence."""

    if not isinstance(content, str) or not content.strip():
        raise ScoreOutputError("response was empty")
    candidate = content.strip()
    fenced = _FENCED_JSON.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ScoreOutputError("response was not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"score"}:
        raise ScoreOutputError("response must contain only score")
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ScoreOutputError("score must be an integer between 0 and 100")
    return score
