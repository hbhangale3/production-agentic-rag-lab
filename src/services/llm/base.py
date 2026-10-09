from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

ChatRole = Literal["system", "user", "assistant"]
VALID_CHAT_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class ChatMessage:
    """Provider-neutral immutable chat message."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_CHAT_ROLES:
            raise ValueError("role must be one of: system, user, assistant")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be a non-empty string")


@dataclass(frozen=True)
class LLMCompletion:
    """Normalized completion data without provider-specific response types."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("completion content must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("completion model must be a non-empty string")
        for field_name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")


@dataclass(frozen=True)
class LLMStreamEvent:
    """Normalized provider stream update.

    Text may be absent on metadata-only terminal chunks.
    """

    text: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("stream text must be a string or None")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("stream model must be a non-empty string or None")
        for field_name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")


class LLMProvider(Protocol):
    """Asynchronous provider-neutral chat completion boundary."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        """Create one non-streaming chat completion."""
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream provider-neutral chat completion updates."""
        ...

    async def close(self) -> None:
        """Release resources owned by the provider."""
        ...
