"""Provider-neutral prompt identity and resolution."""

from src.services.prompts.base import (
    DEFAULT_PROMPT_LABEL,
    LOCAL_PROMPT_VERSION,
    LocalPromptResolver,
    PromptDefinition,
    PromptIdentity,
    PromptResolver,
    ResolvedPrompt,
    canonicalize_prompt,
    fingerprint_prompt,
    local_prompt,
)

__all__ = [
    "DEFAULT_PROMPT_LABEL",
    "LOCAL_PROMPT_VERSION",
    "LocalPromptResolver",
    "PromptDefinition",
    "PromptIdentity",
    "PromptResolver",
    "ResolvedPrompt",
    "canonicalize_prompt",
    "fingerprint_prompt",
    "local_prompt",
]
