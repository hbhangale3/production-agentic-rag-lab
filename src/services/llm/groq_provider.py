from collections.abc import AsyncIterator, Sequence
from typing import Any

import groq
from groq import AsyncGroq
from src.exceptions import LLMRequestError, LLMResponseError
from src.services.llm.base import ChatMessage, LLMCompletion, LLMStreamEvent


class GroqLLMProvider:
    """Provider-neutral adapter over the asynchronous Groq chat client."""

    def __init__(self, *, client: AsyncGroq, model: str, owns_client: bool = False) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("model must not be blank")
        self._client = client
        self.model = normalized_model
        self._owns_client = owns_client
        self._closed = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        if self._closed:
            raise LLMRequestError("LLM provider is closed")
        if not messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise TypeError("messages must contain only ChatMessage values")
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
        }
        if temperature is not None:
            request["temperature"] = float(temperature)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        try:
            response = await self._client.chat.completions.create(**request)
        except groq.APITimeoutError as exc:
            raise LLMRequestError("Groq request timed out") from exc
        except groq.RateLimitError as exc:
            raise LLMRequestError("Groq rate limit exceeded") from exc
        except groq.APIConnectionError as exc:
            raise LLMRequestError("Could not connect to Groq") from exc
        except groq.APIStatusError as exc:
            raise LLMRequestError(f"Groq request failed with status {exc.status_code}") from exc
        except groq.APIError as exc:
            raise LLMRequestError("Groq request failed") from exc

        try:
            choice = response.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError("Groq returned a malformed completion response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("Groq returned empty completion content")

        response_model = getattr(response, "model", None)
        if not isinstance(response_model, str) or not response_model.strip():
            response_model = self.model
        usage = getattr(response, "usage", None)
        return LLMCompletion(
            content=content,
            model=response_model,
            prompt_tokens=self._optional_token_count(usage, "prompt_tokens"),
            completion_tokens=self._optional_token_count(usage, "completion_tokens"),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        request = self._build_request(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        stream = None
        try:
            stream = await self._client.chat.completions.create(**request)
            async for chunk in stream:
                model = getattr(chunk, "model", None)
                if not isinstance(model, str) or not model.strip():
                    model = self.model
                usage = getattr(chunk, "usage", None)
                choices = getattr(chunk, "choices", None)
                if not isinstance(choices, list):
                    raise LLMResponseError("Groq returned a malformed streaming response")
                text = None
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    text = getattr(delta, "content", None)
                    if text is not None and not isinstance(text, str):
                        raise LLMResponseError("Groq returned malformed streaming content")
                    if text == "":
                        text = None
                prompt_tokens = self._optional_token_count(usage, "prompt_tokens")
                completion_tokens = self._optional_token_count(usage, "completion_tokens")
                if text is not None or usage is not None:
                    yield LLMStreamEvent(
                        text=text,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
        except LLMResponseError:
            raise
        except groq.APITimeoutError as exc:
            raise LLMRequestError("Groq request timed out") from exc
        except groq.RateLimitError as exc:
            raise LLMRequestError("Groq rate limit exceeded") from exc
        except groq.APIConnectionError as exc:
            raise LLMRequestError("Could not connect to Groq") from exc
        except groq.APIStatusError as exc:
            raise LLMRequestError(f"Groq request failed with status {exc.status_code}") from exc
        except groq.APIError as exc:
            raise LLMRequestError("Groq request failed") from exc
        finally:
            if stream is not None:
                await stream.close()

    def _build_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        if self._closed:
            raise LLMRequestError("LLM provider is closed")
        if not messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise TypeError("messages must contain only ChatMessage values")
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": stream,
        }
        if temperature is not None:
            request["temperature"] = float(temperature)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        return request

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.close()

    @staticmethod
    def _optional_token_count(usage: Any, field_name: str) -> int | None:
        value = getattr(usage, field_name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
