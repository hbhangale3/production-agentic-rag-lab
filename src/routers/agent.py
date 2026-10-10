import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from src.dependencies import AgentServiceDep, ObservabilityDep
from src.schemas.agent import AgentAskResponse, AgentStatus
from src.schemas.ask import AskRequest
from src.services.agent.cache_identity import AGENT_PIPELINE_VERSION
from src.services.agent.service import AgentExecutionFailure, AgentFailureKind
from src.services.agent.status import CACHE_HIT_STATUS
from src.services.observability import SafeObservation, safe_content_capture, safe_start_trace
from src.services.rag.validation import VALID_CITATION
from starlette.responses import StreamingResponse

router = APIRouter()

# Execution failures only. Semantic outcomes (out of scope, insufficient evidence, grounding failed)
# are normal 200 responses carrying a fixed safe message and no model-written answer.
FAILURE_RESPONSES: dict[AgentFailureKind, tuple[int, str, str]] = {
    AgentFailureKind.RETRIEVAL_FAILED: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "retrieval_failed",
        "Research retrieval is temporarily unavailable.",
    ),
    AgentFailureKind.GENERATION_FAILED: (
        status.HTTP_502_BAD_GATEWAY,
        "generation_failed",
        "Language model generation failed.",
    ),
    AgentFailureKind.TIMEOUT: (
        status.HTTP_504_GATEWAY_TIMEOUT,
        "timeout",
        "The research assistant took too long to answer this question.",
    ),
    AgentFailureKind.INTERNAL_ERROR: (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "The research assistant could not complete this request.",
    ),
}
_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# A live fallback can be silent for minutes while a PDF is parsed; comment frames keep the connection open.
KEEPALIVE_SECONDS = 15.0
KEEPALIVE_FRAME = ": keepalive\n\n"


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _start_trace(observability: ObservabilityDep, transport: str) -> SafeObservation:
    return safe_start_trace(
        observability,
        name="agent.request",
        metadata={
            "transport": transport,
            "content_capture": safe_content_capture(observability),
            "pipeline_version": AGENT_PIPELINE_VERSION,
        },
    )


def _finish_failure(root: SafeObservation, classification: str) -> None:
    root.update(
        metadata={"status": "error", "failure_classification": classification},
        error_type=classification,
    )
    root.end()


def _finish_success(root: SafeObservation, service: AgentServiceDep, response: AgentAskResponse) -> None:
    root.update(metadata=service.trace_metadata(response))
    root.end()


def _done_payload(response: AgentAskResponse) -> dict[str, object]:
    return {
        "outcome": response.outcome,
        "terminal_reason": response.terminal_reason,
        "grounding_passed": response.grounding_passed,
        "cache_status": response.cache_status,
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "citations": list(dict.fromkeys(VALID_CITATION.findall(response.answer)))
        if response.outcome == "answered"
        else [],
        "execution": response.execution.model_dump(mode="json"),
        "latency_ms": response.latency_ms,
    }


def _result_frames(response: AgentAskResponse) -> list[str]:
    """Frames after the status events: a grounded answer with sources, or a safe outcome message."""
    if response.outcome == "answered":
        return [
            _sse("answer", {"text": response.answer}),
            _sse("sources", {"sources": [source.model_dump(mode="json") for source in response.sources]}),
            _sse("done", _done_payload(response)),
        ]
    return [
        _sse("outcome", {"outcome": response.outcome, "message": response.answer}),
        _sse("done", _done_payload(response)),
    ]


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(
    request: AskRequest,
    service: AgentServiceDep,
    observability: ObservabilityDep,
) -> AgentAskResponse:
    """Answer one research question with the bounded agent, or return a safe controlled outcome."""
    root = _start_trace(observability, "ask")
    try:
        response = await service.answer(request.question, observation=root)
    except AgentExecutionFailure as exc:
        status_code, code, detail = FAILURE_RESPONSES[exc.kind]
        _finish_failure(root, code)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except asyncio.CancelledError:
        _finish_failure(root, "cancelled")
        raise
    except Exception as exc:
        _finish_failure(root, "internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=FAILURE_RESPONSES[AgentFailureKind.INTERNAL_ERROR][2],
        ) from exc
    _finish_success(root, service, response)
    return response


@router.post("/agent/ask/stream")
async def agent_ask_stream(
    request: AskRequest,
    service: AgentServiceDep,
    observability: ObservabilityDep,
) -> StreamingResponse:
    """Stream safe execution statuses, then the answer only after it has passed grounding."""
    root = _start_trace(observability, "stream")
    try:
        prepared = await service.prepare(request.question, observation=root)
    except asyncio.CancelledError:
        _finish_failure(root, "cancelled")
        raise
    except Exception as exc:
        _finish_failure(root, "internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=FAILURE_RESPONSES[AgentFailureKind.INTERNAL_ERROR][2],
        ) from exc
    metadata = _sse(
        "metadata",
        {"cache_status": prepared.cache_status, "pipeline_version": AGENT_PIPELINE_VERSION},
    )
    cached = service.cached_response(prepared)

    async def cached_events() -> AsyncIterator[str]:
        try:
            yield metadata
            yield _sse("status", CACHE_HIT_STATUS.model_dump())
            for frame in _result_frames(cached):
                if frame.startswith("event: done\n"):
                    _finish_success(root, service, cached)
                yield frame
        except (asyncio.CancelledError, GeneratorExit):
            _finish_failure(root, "cancelled")
            raise

    async def events() -> AsyncIterator[str]:
        statuses: asyncio.Queue[AgentStatus] = asyncio.Queue()
        run = asyncio.create_task(service.execute(prepared, observation=root, on_status=statuses.put))
        try:
            yield metadata
            while True:
                next_status = asyncio.create_task(statuses.get())
                await asyncio.wait({next_status, run}, timeout=KEEPALIVE_SECONDS, return_when=asyncio.FIRST_COMPLETED)
                if next_status.done():
                    yield _sse("status", next_status.result().model_dump())
                    continue
                next_status.cancel()
                if run.done():
                    break
                yield KEEPALIVE_FRAME
            while not statuses.empty():
                yield _sse("status", statuses.get_nowait().model_dump())
            try:
                response = run.result()
            except AgentExecutionFailure as exc:
                _, code, message = FAILURE_RESPONSES[exc.kind]
                _finish_failure(root, code)
                yield _sse("error", {"code": code, "message": message})
                return
            except Exception:
                _, code, message = FAILURE_RESPONSES[AgentFailureKind.INTERNAL_ERROR]
                _finish_failure(root, code)
                yield _sse("error", {"code": code, "message": message})
                return
            for frame in _result_frames(response):
                if frame.startswith("event: done\n"):
                    _finish_success(root, service, response)
                yield frame
        except (asyncio.CancelledError, GeneratorExit):
            _finish_failure(root, "cancelled")
            raise
        finally:
            if not run.done():
                run.cancel()

    return StreamingResponse(
        cached_events() if cached is not None else events(),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )
