import codecs
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

SUPPORTED_EVENTS = frozenset({"metadata", "status", "delta", "answer", "sources", "outcome", "done", "error"})


class RAGUIClientError(Exception):
    """Safe presentation-layer failure with no backend response details."""


class SSEProtocolError(RAGUIClientError):
    """The API stream did not follow the documented SSE contract."""


class IncompleteStreamError(RAGUIClientError):
    """The API stream closed without a terminal done or error event."""


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


class JSONSSEParser:
    """Incrementally frame UTF-8 SSE bytes and decode JSON data payloads."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        if not isinstance(chunk, bytes):
            raise TypeError("SSE chunks must be bytes")
        self._buffer += self._decoder.decode(chunk)
        return self._drain(allow_unterminated=False)

    def finish(self) -> list[SSEEvent]:
        self._buffer += self._decoder.decode(b"", final=True)
        return self._drain(allow_unterminated=True)

    def _drain(self, *, allow_unterminated: bool) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        while True:
            boundary = self._next_boundary()
            if boundary is None:
                break
            index, width = boundary
            frame, self._buffer = self._buffer[:index], self._buffer[index + width :]
            if self._has_event_fields(frame):
                events.append(self._parse_frame(frame))
        if allow_unterminated and self._has_event_fields(self._buffer):
            frame, self._buffer = self._buffer, ""
            events.append(self._parse_frame(frame))
        return events

    def _next_boundary(self) -> tuple[int, int] | None:
        candidates = [
            (index, len(delimiter)) for delimiter in ("\n\n", "\r\n\r\n") if (index := self._buffer.find(delimiter)) >= 0
        ]
        return min(candidates) if candidates else None

    @staticmethod
    def _has_event_fields(frame: str) -> bool:
        return any(line and not line.startswith(":") for line in frame.splitlines())

    @staticmethod
    def _parse_frame(frame: str) -> SSEEvent:
        event_name: str | None = None
        data_lines: list[str] = []
        for line in frame.splitlines():
            if not line or line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value
            elif field == "data":
                data_lines.append(value)
        if event_name not in SUPPORTED_EVENTS:
            raise SSEProtocolError("RAG API returned an unsupported stream event.")
        if not data_lines:
            raise SSEProtocolError("RAG API returned a stream event without data.")
        try:
            data = json.loads("\n".join(data_lines))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SSEProtocolError("RAG API returned malformed stream data.") from exc
        if not isinstance(data, dict):
            raise SSEProtocolError("RAG API returned a non-object stream payload.")
        return SSEEvent(event=event_name, data=data)


class RAGStreamClient:
    """Small HTTP/SSE client for the authoritative FastAPI RAG endpoint."""

    endpoint_path = "/api/v1/ask/stream"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be blank")
        self.endpoint = f"{normalized}{self.endpoint_path}"
        self.timeout = httpx.Timeout(timeout_seconds)
        self.transport = transport

    async def stream(self, question: str) -> AsyncIterator[SSEEvent]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be blank")
        parser = JSONSSEParser()
        terminal = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    json={"question": normalized_question},
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    if response.status_code != 200:
                        raise self._http_error(response.status_code)
                    content_type = response.headers.get("content-type", "")
                    if not content_type.lower().startswith("text/event-stream"):
                        raise SSEProtocolError("RAG API returned an unexpected response type.")
                    async for chunk in response.aiter_bytes():
                        for event in parser.feed(chunk):
                            yield event
                            if event.event in {"done", "error"}:
                                terminal = True
                                return
                    for event in parser.finish():
                        yield event
                        if event.event in {"done", "error"}:
                            terminal = True
                            return
        except RAGUIClientError:
            raise
        except httpx.TimeoutException as exc:
            raise RAGUIClientError("RAG API request timed out.") from exc
        except httpx.RequestError as exc:
            raise RAGUIClientError("RAG API is unavailable.") from exc
        if not terminal:
            raise IncompleteStreamError("RAG API stream ended before completion.")

    @staticmethod
    def _http_error(status_code: int) -> RAGUIClientError:
        if status_code == 422:
            return RAGUIClientError("The question was rejected by the RAG API.")
        if status_code == 502:
            return RAGUIClientError("The generated answer could not be validated.")
        if status_code == 503:
            return RAGUIClientError("RAG API is temporarily unavailable.")
        if status_code == 504:
            return RAGUIClientError("RAG API took too long to respond.")
        return RAGUIClientError("RAG API request failed.")


class AgentStreamClient(RAGStreamClient):
    """Client for the agent SSE endpoint; a live fallback can take minutes, so the timeout is longer."""

    endpoint_path = "/api/v1/agent/ask/stream"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 240.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds, transport=transport)
