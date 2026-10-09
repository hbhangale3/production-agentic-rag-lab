import json
from collections.abc import AsyncIterator
from functools import partial

from fastapi import APIRouter, HTTPException, status
from src.dependencies import HybridSearchServiceDep, RAGGenerationServiceDep, RequestSettingsDep
from src.exceptions import (
    HybridSearchError,
    InsufficientEvidenceError,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    RAGPromptBudgetError,
)
from src.schemas.ask import AskRequest, AskResponse, build_ask_response, build_ask_source
from src.services.rag import PreparedRAGGeneration, RAGStreamComplete, RAGStreamDelta
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

router = APIRouter()
INSUFFICIENT_EVIDENCE_ANSWER = "The available indexed evidence is insufficient to answer this question."


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    hybrid_service: HybridSearchServiceDep,
    rag_service: RAGGenerationServiceDep,
    settings: RequestSettingsDep,
) -> AskResponse:
    """Retrieve ranked evidence and generate one grounded research answer."""
    search = partial(
        hybrid_service.search,
        request.question,
        size=settings.rag_retrieval_size,
    )
    try:
        retrieval_result = await run_in_threadpool(search)
    except HybridSearchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research retrieval is temporarily unavailable.",
        ) from exc

    try:
        generation = await rag_service.generate(
            question=request.question,
            retrieval_result=retrieval_result,
        )
    except InsufficientEvidenceError:
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            retrieval_mode=retrieval_result.retrieval_mode,
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
        )
    except RAGPromptBudgetError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The question and retrieved evidence exceed the generation budget.",
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model service is not configured.",
        ) from exc
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model service is temporarily unavailable.",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model service returned an unusable response.",
        ) from exc
    return build_ask_response(generation)


@router.post("/ask/stream")
async def stream_ask_question(
    request: AskRequest,
    hybrid_service: HybridSearchServiceDep,
    rag_service: RAGGenerationServiceDep,
    settings: RequestSettingsDep,
) -> StreamingResponse:
    """Retrieve once, then stream grounded generation as JSON SSE events."""
    search = partial(hybrid_service.search, request.question, size=settings.rag_retrieval_size)
    try:
        retrieval_result = await run_in_threadpool(search)
    except HybridSearchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research retrieval is temporarily unavailable.",
        ) from exc

    prepared: PreparedRAGGeneration | None
    try:
        prepared = rag_service.prepare(question=request.question, retrieval_result=retrieval_result)
    except InsufficientEvidenceError:
        prepared = None
    except RAGPromptBudgetError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The question and retrieved evidence exceed the generation budget.",
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model service is not configured.",
        ) from exc

    async def events() -> AsyncIterator[str]:
        source_count = len(prepared.sources) if prepared is not None else 0
        yield _sse(
            "metadata",
            {"retrieval_mode": retrieval_result.retrieval_mode, "source_count": source_count},
        )
        if prepared is None:
            yield _sse("delta", {"text": INSUFFICIENT_EVIDENCE_ANSWER})
            yield _sse("sources", {"sources": []})
            yield _sse(
                "done",
                {
                    "model": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "citations": [],
                },
            )
            return
        try:
            async for event in rag_service.stream_generate(prepared):
                if isinstance(event, RAGStreamDelta):
                    yield _sse("delta", {"text": event.text})
                elif isinstance(event, RAGStreamComplete):
                    sources = [build_ask_source(source).model_dump(mode="json") for source in prepared.sources]
                    yield _sse("sources", {"sources": sources})
                    yield _sse(
                        "done",
                        {
                            "model": event.model,
                            "prompt_tokens": event.prompt_tokens,
                            "completion_tokens": event.completion_tokens,
                            "citations": list(event.cited_labels),
                        },
                    )
        except (LLMConfigurationError, LLMRequestError, LLMResponseError):
            yield _sse(
                "error",
                {
                    "code": "generation_failed",
                    "message": "Language model generation failed.",
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
