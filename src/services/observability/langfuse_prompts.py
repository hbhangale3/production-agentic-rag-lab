"""Fail-open managed-prompt resolution through the Langfuse SDK client."""

import asyncio
import logging
from typing import Any

from src.services.prompts.base import DEFAULT_PROMPT_LABEL, PromptDefinition, ResolvedPrompt, local_prompt

logger = logging.getLogger(__name__)


class LangfusePromptResolver:
    """Fetch a text prompt by name and label, falling back to the local definition on any problem.

    The SDK caches fetched prompts in process for ``cache_ttl_seconds``, so a
    managed prompt edit takes effect within that window and most resolutions
    make no network request.
    """

    def __init__(self, *, client: Any, timeout_seconds: float, cache_ttl_seconds: int = 60) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if isinstance(cache_ttl_seconds, bool) or not isinstance(cache_ttl_seconds, int) or cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be a non-negative integer")
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._cache_ttl_seconds = cache_ttl_seconds

    async def resolve(self, definition: PromptDefinition, *, label: str = DEFAULT_PROMPT_LABEL) -> ResolvedPrompt:
        try:
            managed = await asyncio.wait_for(
                asyncio.to_thread(self._fetch, definition.name, label),
                timeout=self._timeout_seconds,
            )
            content = getattr(managed, "prompt", None)
            version = getattr(managed, "version", None)
            if getattr(managed, "is_fallback", False) or not isinstance(content, str) or not content.strip():
                raise ValueError("managed prompt is not a usable text prompt")
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ValueError("managed prompt has no usable version")
            if any(marker not in content for marker in definition.required_markers):
                raise ValueError("managed prompt is missing a required contract marker")
            return ResolvedPrompt(
                name=definition.name,
                content=content,
                version=f"langfuse-v{version}",
                label=label,
                source="langfuse",
            )
        except Exception as exc:
            # The exception may carry SDK detail; only its type is logged.
            logger.warning(
                "Managed prompt %s unavailable (%s); using local fallback", definition.name, type(exc).__name__
            )
            return local_prompt(definition, label=label)

    def _fetch(self, name: str, label: str) -> Any:
        # No SDK-level fallback and no retries: failures surface here and resolve to our local prompt.
        return self._client.get_prompt(
            name,
            label=label,
            type="text",
            cache_ttl_seconds=self._cache_ttl_seconds,
            max_retries=0,
            fetch_timeout_seconds=max(1, int(self._timeout_seconds)),
        )
