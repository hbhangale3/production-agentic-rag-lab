from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import groq
import httpx
import pytest
from src.exceptions import LLMRequestError, LLMResponseError
from src.services.llm.base import ChatMessage, LLMCompletion
from src.services.llm.groq_provider import GroqLLMProvider


def completion(
    content: str | None = "Grounded answer",
    *,
    model: str | None = "groq-model",
    prompt_tokens: int | None = 12,
    completion_tokens: int | None = 7,
):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=usage,
    )


def provider(*, response=None, owns_client: bool = False):
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=response if response is not None else completion()
    )
    client.close = AsyncMock()
    return GroqLLMProvider(client=client, model="configured-model", owns_client=owns_client), client


@pytest.mark.anyio
async def test_forwards_ordered_messages_model_and_non_streaming_mode() -> None:
    instance, client = provider()
    messages = [
        ChatMessage(role="system", content="Use supplied evidence."),
        ChatMessage(role="user", content="What did the paper find?"),
        ChatMessage(role="assistant", content="Earlier response."),
    ]

    result = await instance.complete(messages)

    client.chat.completions.create.assert_awaited_once_with(
        model="configured-model",
        messages=[
            {"role": "system", "content": "Use supplied evidence."},
            {"role": "user", "content": "What did the paper find?"},
            {"role": "assistant", "content": "Earlier response."},
        ],
        stream=False,
    )
    assert result == LLMCompletion(
        content="Grounded answer",
        model="groq-model",
        prompt_tokens=12,
        completion_tokens=7,
    )


@pytest.mark.anyio
async def test_forwards_optional_generation_parameters() -> None:
    instance, client = provider()

    await instance.complete(
        [ChatMessage(role="user", content="Summarize")],
        temperature=0.2,
        max_tokens=256,
    )

    assert client.chat.completions.create.await_args.kwargs["temperature"] == 0.2
    assert client.chat.completions.create.await_args.kwargs["max_tokens"] == 256


@pytest.mark.anyio
async def test_configured_model_is_used_when_response_model_is_missing() -> None:
    instance, _ = provider(response=completion(model=None, prompt_tokens=None, completion_tokens=None))

    result = await instance.complete([ChatMessage(role="user", content="Question")])

    assert result.model == "configured-model"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


@pytest.mark.parametrize(
    ("role", "content", "message"),
    [
        ("tool", "content", "role must be one of"),
        ("user", "   ", "content must be a non-empty"),
    ],
)
def test_chat_message_validation(role, content, message) -> None:
    with pytest.raises(ValueError, match=message):
        ChatMessage(role=role, content=content)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("messages", "kwargs", "message"),
    [
        ([], {}, "messages must not be empty"),
        (["not-a-message"], {}, "only ChatMessage"),
        ([ChatMessage(role="user", content="q")], {"temperature": -0.1}, "temperature"),
        ([ChatMessage(role="user", content="q")], {"temperature": 2.1}, "temperature"),
        ([ChatMessage(role="user", content="q")], {"max_tokens": 0}, "max_tokens"),
    ],
)
async def test_invalid_request_is_rejected_before_client_call(messages, kwargs, message) -> None:
    instance, client = provider()

    with pytest.raises((TypeError, ValueError), match=message):
        await instance.complete(messages, **kwargs)

    client.chat.completions.create.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (SimpleNamespace(choices=[], model="m", usage=None), "malformed"),
        (completion(content=None), "empty completion"),
        (completion(content="  "), "empty completion"),
    ],
)
async def test_malformed_or_empty_response_is_rejected(response, message) -> None:
    instance, _ = provider(response=response)

    with pytest.raises(LLMResponseError, match=message):
        await instance.complete([ChatMessage(role="user", content="Question")])


def sdk_errors():
    request = httpx.Request("POST", "https://api.groq.test/chat/completions")
    return [
        (groq.APITimeoutError(request), "timed out"),
        (groq.APIConnectionError(message="connection secret-value", request=request), "connect"),
        (
            groq.RateLimitError(
                "rate limit secret-value",
                response=httpx.Response(429, request=request),
                body={"sensitive": "secret-value"},
            ),
            "rate limit",
        ),
        (
            groq.APIStatusError(
                "server secret-value",
                response=httpx.Response(503, request=request),
                body={"sensitive": "secret-value"},
            ),
            "status 503",
        ),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(("error", "message"), sdk_errors())
async def test_sdk_errors_are_mapped_without_sensitive_details(error, message) -> None:
    instance, client = provider()
    client.chat.completions.create.side_effect = error

    with pytest.raises(LLMRequestError, match=message) as caught:
        await instance.complete([ChatMessage(role="user", content="Question")])

    assert "secret-value" not in str(caught.value)


@pytest.mark.anyio
async def test_owned_client_is_closed_once_and_cannot_be_reused() -> None:
    instance, client = provider(owns_client=True)

    await instance.close()
    await instance.close()

    client.close.assert_awaited_once_with()
    with pytest.raises(LLMRequestError, match="closed"):
        await instance.complete([ChatMessage(role="user", content="Question")])


@pytest.mark.anyio
async def test_externally_owned_client_is_reused_and_not_closed() -> None:
    instance, client = provider(owns_client=False)

    await instance.complete([ChatMessage(role="user", content="First")])
    await instance.complete([ChatMessage(role="user", content="Second")])
    await instance.close()

    assert client.chat.completions.create.await_count == 2
    client.close.assert_not_awaited()
