"""Provider-neutral prompt identity and resolution.

A prompt's identity describes its template: name, version, label, and a
fingerprint of the template text. It never includes runtime data such as a
question, evidence, or a previous answer.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

DEFAULT_PROMPT_LABEL = "production"
LOCAL_PROMPT_VERSION = "local-v1"
PromptSource = Literal["local", "langfuse"]


def canonicalize_prompt(content: str) -> str:
    """Normalize line endings and outer whitespace so equivalent templates hash equally."""

    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def fingerprint_prompt(content: str) -> str:
    """SHA-256 hex digest of the canonical template text."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("prompt content must be a non-empty string")
    return hashlib.sha256(canonicalize_prompt(content).encode("utf-8")).hexdigest()


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prompt {name} must be a non-empty string")


@dataclass(frozen=True)
class PromptIdentity:
    """Content-free identity of one prompt template."""

    name: str
    version: str
    label: str
    fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "label", "fingerprint"):
            _require_text(getattr(self, field_name), field_name)

    def as_metadata(self) -> dict[str, object]:
        """Telemetry-safe metadata: identity only, never template or rendered text."""

        return {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_label": self.label,
            "prompt_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PromptDefinition:
    """A named prompt with its authoritative local fallback."""

    name: str
    fallback_content: str
    fallback_version: str = LOCAL_PROMPT_VERSION
    required_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "fallback_content", "fallback_version"):
            _require_text(getattr(self, field_name), field_name)
        if any(marker not in self.fallback_content for marker in self.required_markers):
            raise ValueError("prompt fallback content must contain its own required markers")


@dataclass(frozen=True)
class ResolvedPrompt:
    """The effective template for one prompt plus where it came from."""

    name: str
    content: str
    version: str
    label: str
    source: PromptSource = "local"

    def __post_init__(self) -> None:
        for field_name in ("name", "content", "version", "label"):
            _require_text(getattr(self, field_name), field_name)
        if self.source not in ("local", "langfuse"):
            raise ValueError("prompt source must be local or langfuse")

    @property
    def fingerprint(self) -> str:
        return fingerprint_prompt(self.content)

    @property
    def identity(self) -> PromptIdentity:
        return PromptIdentity(name=self.name, version=self.version, label=self.label, fingerprint=self.fingerprint)


def local_prompt(definition: PromptDefinition, *, label: str = DEFAULT_PROMPT_LABEL) -> ResolvedPrompt:
    """Resolve a definition to its local fallback."""

    return ResolvedPrompt(
        name=definition.name,
        content=definition.fallback_content,
        version=definition.fallback_version,
        label=label,
        source="local",
    )


class PromptResolver(Protocol):
    """Resolve a prompt template; implementations must fall back to the local definition, never raise."""

    async def resolve(self, definition: PromptDefinition, *, label: str = DEFAULT_PROMPT_LABEL) -> ResolvedPrompt: ...


class LocalPromptResolver:
    """Always use the local fallback; the behaviour when prompt management is disabled."""

    async def resolve(self, definition: PromptDefinition, *, label: str = DEFAULT_PROMPT_LABEL) -> ResolvedPrompt:
        return local_prompt(definition, label=label)
