import json

import httpx
import pytest
from src.gradio_client import (
    IncompleteStreamError,
    JSONSSEParser,
    RAGStreamClient,
    RAGUIClientError,
    SSEEvent,
    SSEProtocolError,
)


def frame(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def test_parser_handles_one_event_per_chunk_and_all_event_types() -> None:
    parser = JSONSSEParser()
    events = []
    for name, data in (
        ("metadata", {"retrieval_mode": "hybrid", "source_count": 1}),
        ("delta", {"text": "hello"}),
        ("sources", {"sources": []}),
        ("done", {"model": "model"}),
        ("error", {"code": "failed"}),
    ):
        events.extend(parser.feed(frame(name, data)))

    assert [event.event for event in events] == [
        "metadata",
        "delta",
        "sources",
        "done",
        "error",
    ]


def test_parser_preserves_optional_cache_and_unknown_metadata_fields() -> None:
    parser = JSONSSEParser()

    events = parser.feed(
        frame(
            "metadata",
            {"cache_status": "hit", "retrieval_mode": "hybrid", "future_optional": 7},
        )
    )

    assert events == [
        SSEEvent(
            event="metadata",
            data={"cache_status": "hit", "retrieval_mode": "hybrid", "future_optional": 7},
        )
    ]


def test_parser_handles_split_utf8_and_multiple_events_in_one_chunk() -> None:
    parser = JSONSSEParser()
    payload = frame("delta", {"text": "café 世界"}) + frame("done", {"model": None})
    split = payload.index("é".encode()) + 1

    first = parser.feed(payload[:split])
    second = parser.feed(payload[split:])

    assert first == []
    assert second == [
        SSEEvent(event="delta", data={"text": "café 世界"}),
        SSEEvent(event="done", data={"model": None}),
    ]


def test_parser_ignores_blank_and_comment_frames_and_accepts_final_unterminated_event() -> None:
    parser = JSONSSEParser()

    events = parser.feed(b"\n\n: keepalive\n\n")
    events.extend(parser.feed(b'event: done\ndata: {"model":null}'))
    events.extend(parser.finish())

    assert events == [SSEEvent(event="done", data={"model": None})]


@pytest.mark.parametrize(
    "payload",
    [
        b"event: delta\ndata: not-json\n\n",
        b'event: unexpected\ndata: {"value":1}\n\n',
        b"event: delta\n\n",
    ],
)
def test_parser_rejects_malformed_or_unknown_events(payload) -> None:
    with pytest.raises(SSEProtocolError):
        JSONSSEParser().feed(payload)


@pytest.mark.anyio
async def test_client_posts_contract_and_streams_accumulated_protocol_events() -> None:
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["accept"] = request.headers["accept"]
        seen["body"] = json.loads((await request.aread()).decode())
        payload = frame("metadata", {"source_count": 1}) + frame("delta", {"text": "A [S1]."})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=ChunkStream([payload[:17], payload[17:], frame("sources", {"sources": []}), frame("done", {})]),
        )

    client = RAGStreamClient(base_url="http://api.test/root/", transport=httpx.MockTransport(handler))

    events = [event async for event in client.stream("  question  ")]

    assert seen == {
        "url": "http://api.test/root/api/v1/ask/stream",
        "accept": "text/event-stream",
        "body": {"question": "question"},
    }
    assert [event.event for event in events] == ["metadata", "delta", "sources", "done"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (422, "question was rejected"),
        (502, "could not be validated"),
        (503, "temporarily unavailable"),
        (500, "request failed"),
    ],
)
async def test_client_maps_http_failures_without_response_body(status, message) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="internal secret traceback")

    client = RAGStreamClient(base_url="http://api.test", transport=httpx.MockTransport(handler))

    with pytest.raises(RAGUIClientError, match=message) as caught:
        _ = [event async for event in client.stream("question")]

    assert "secret" not in str(caught.value)


@pytest.mark.anyio
async def test_client_maps_connection_failure_and_timeout() -> None:
    request = httpx.Request("POST", "http://api.test/api/v1/ask/stream")
    for failure, message in (
        (httpx.ConnectError("private host", request=request), "unavailable"),
        (httpx.ReadTimeout("private timeout", request=request), "timed out"),
    ):

        def handler(_request: httpx.Request, failure=failure):
            raise failure

        client = RAGStreamClient(base_url="http://api.test", transport=httpx.MockTransport(handler))
        with pytest.raises(RAGUIClientError, match=message) as caught:
            _ = [event async for event in client.stream("question")]
        assert "private" not in str(caught.value)


@pytest.mark.anyio
async def test_client_rejects_premature_stream_close_after_partial_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream([frame("delta", {"text": "partial"})]),
        )

    client = RAGStreamClient(base_url="http://api.test", transport=httpx.MockTransport(handler))
    iterator = client.stream("question")

    assert await anext(iterator) == SSEEvent(event="delta", data={"text": "partial"})
    with pytest.raises(IncompleteStreamError):
        await anext(iterator)


@pytest.mark.anyio
async def test_client_rejects_blank_question_without_transport_call() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    client = RAGStreamClient(base_url="http://api.test", transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError, match="blank"):
        _ = [event async for event in client.stream("  ")]
    assert called is False
