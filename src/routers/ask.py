import asyncio
import json
import re
from collections.abc import AsyncIterator
from functools import partial

from fastapi import APIRouter, HTTPException, status
from src.dependencies import (
    HybridSearchServiceDep,
    ObservabilityDep,
    RAGGenerationServiceDep,
    RAGResponseCacheDep,
    RequestSettingsDep,
)
from src.exceptions import (
    GroundingValidationError,
    HybridSearchError,
    InsufficientEvidenceError,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    RAGPromptBudgetError,
)
from src.schemas.ask import AskRequest, AskResponse, build_ask_response, build_ask_source
from src.services.cache import RAGCacheLookup, RAGCacheWriteOutcome
from src.services.observability import SafeObservation, safe_content_capture, safe_start_trace
from src.services.rag import PreparedRAGGeneration, RAGStreamComplete, RAGStreamDelta
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

router = APIRouter()
INSUFFICIENT_EVIDENCE_ANSWER = "The available indexed evidence is insufficient to answer this question."
CITATION = re.compile(r"\[S[1-9]\d*\]")


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _answer_citations(answer: str) -> list[str]:
    """Recover ordered unique citations for the existing SSE done payload."""
    return list(dict.fromkeys(CITATION.findall(answer)))


def _response_trace_metadata(
    response: AskResponse,
    *,
    cache_hit: bool = False,
    grounding_performed: bool = True,
) -> dict[str, object]:
    prefix = "artifact_" if cache_hit else ""
    metadata: dict[str, object] = {
        "status": "success",
        "retrieval_mode": response.retrieval_mode,
        "source_count": len(response.sources),
        f"{prefix}model": response.model,
    }
    if grounding_performed:
        metadata[f"{prefix}grounding"] = "passed"
    metadata[f"{prefix}prompt_tokens"] = response.prompt_tokens
    metadata[f"{prefix}completion_tokens"] = response.completion_tokens
    return metadata


def _finish_failure(root: SafeObservation, classification: str, exc: Exception | None = None) -> None:
    metadata: dict[str, object] = {"status": "error", "failure_classification": classification}
    if exc is not None:
        metadata["error_type"] = type(exc).__name__
    root.update(metadata=metadata, error_type=classification)
    root.end()


async def _store_with_trace(
    response_cache: RAGResponseCacheDep,
    lookup: RAGCacheLookup,
    response: AskResponse,
    root: SafeObservation,
) -> RAGCacheWriteOutcome:
    if lookup.key is None:
        return RAGCacheWriteOutcome.SKIPPED
    span = root.start_span(name="rag.cache.write", metadata=None)
    outcome = await response_cache.store(lookup, response)
    span.update(
        metadata={"outcome": outcome.value},
        error_type="cache_write_failure" if outcome is RAGCacheWriteOutcome.FAILED else None,
    )
    span.end()
    return outcome


async def _replay_cached_response(response: AskResponse) -> AsyncIterator[str]:
    """Replay one complete cached response through the compatible SSE protocol."""
    yield _sse(
        "metadata",
        {"retrieval_mode": response.retrieval_mode, "source_count": len(response.sources)},
    )
    yield _sse("delta", {"text": response.answer})
    yield _sse(
        "sources",
        {"sources": [source.model_dump(mode="json") for source in response.sources]},
    )
    yield _sse(
        "done",
        {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "citations": _answer_citations(response.answer),
        },
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    hybrid_service: HybridSearchServiceDep,
    rag_service: RAGGenerationServiceDep,
    response_cache: RAGResponseCacheDep,
    settings: RequestSettingsDep,
    observability: ObservabilityDep,
) -> AskResponse:
    """Retrieve ranked evidence and generate one grounded research answer."""
    root = safe_start_trace(
        observability,
        name="rag.request",
        metadata={"transport": "ask", "content_capture": safe_content_capture(observability)},
    )
    cache_span = root.start_span(
        name="rag.cache.lookup",
        metadata={"cache_enabled": response_cache.enabled, "cache_schema_version": "v1"},
    )
    cache_lookup = await response_cache.lookup(question=request.question)
    cache_span.update(
        metadata={"outcome": cache_lookup.outcome.value},
        error_type="cache_read_failure" if cache_lookup.outcome.value == "failure" else None,
    )
    cache_span.end()
    if cache_lookup.response is not None:
        root.update(
            metadata={
                "cache_outcome": "hit",
                **_response_trace_metadata(cache_lookup.response, cache_hit=True),
            }
        )
        root.end()
        return cache_lookup.response

    root.update(metadata={"cache_outcome": cache_lookup.outcome.value})
    retrieval_span = root.start_span(
        name="rag.retrieval",
        metadata={"requested_result_size": settings.rag_retrieval_size},
    )
    search = partial(
        hybrid_service.search,
        request.question,
        size=settings.rag_retrieval_size,
    )
    try:
        retrieval_result = await run_in_threadpool(search)
    except HybridSearchError as exc:
        retrieval_span.update(error_type="retrieval_failure")
        retrieval_span.end()
        _finish_failure(root, "retrieval_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research retrieval is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        retrieval_span.update(error_type="retrieval_failure")
        retrieval_span.end()
        _finish_failure(root, "internal_failure", exc)
        raise
    retrieval_span.update(
        metadata={
            "returned_result_count": retrieval_result.count,
            "retrieval_mode": retrieval_result.retrieval_mode,
        }
    )
    retrieval_span.end()

    try:
        generation = await rag_service.generate(
            question=request.question,
            retrieval_result=retrieval_result,
            observation=root,
        )
    except InsufficientEvidenceError:
        response = AskResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            retrieval_mode=retrieval_result.retrieval_mode,
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
        )
        write_outcome = await _store_with_trace(response_cache, cache_lookup, response, root)
        root.update(
            metadata={
                "cache_write_outcome": write_outcome.value,
                "valid_insufficiency": True,
                **_response_trace_metadata(response, grounding_performed=False),
            }
        )
        root.end()
        return response
    except RAGPromptBudgetError as exc:
        _finish_failure(root, "evidence_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The question and retrieved evidence exceed the generation budget.",
        ) from exc
    except LLMConfigurationError as exc:
        _finish_failure(root, "generation_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model service is not configured.",
        ) from exc
    except LLMRequestError as exc:
        _finish_failure(root, "generation_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model service is temporarily unavailable.",
        ) from exc
    except GroundingValidationError as exc:
        _finish_failure(root, "grounding_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model service returned an unusable response.",
        ) from exc
    except LLMResponseError as exc:
        _finish_failure(root, "generation_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model service returned an unusable response.",
        ) from exc
    except Exception as exc:
        _finish_failure(root, "internal_failure", exc)
        raise
    response = build_ask_response(generation)
    write_outcome = await _store_with_trace(response_cache, cache_lookup, response, root)
    root.update(
        metadata={
            "cache_write_outcome": write_outcome.value,
            **_response_trace_metadata(response),
        }
    )
    root.end()
    return response


@router.post("/ask/stream")
async def stream_ask_question(
    request: AskRequest,
    hybrid_service: HybridSearchServiceDep,
    rag_service: RAGGenerationServiceDep,
    response_cache: RAGResponseCacheDep,
    settings: RequestSettingsDep,
    observability: ObservabilityDep,
) -> StreamingResponse:
    """Retrieve once, then stream grounded generation as JSON SSE events."""
    root = safe_start_trace(
        observability,
        name="rag.request",
        metadata={"transport": "stream", "content_capture": safe_content_capture(observability)},
    )
    cache_span = root.start_span(
        name="rag.cache.lookup",
        metadata={"cache_enabled": response_cache.enabled, "cache_schema_version": "v1"},
    )
    cache_lookup = await response_cache.lookup(question=request.question)
    cache_span.update(
        metadata={"outcome": cache_lookup.outcome.value},
        error_type="cache_read_failure" if cache_lookup.outcome.value == "failure" else None,
    )
    cache_span.end()
    if cache_lookup.response is not None:
        async def cached_events() -> AsyncIterator[str]:
            try:
                async for frame in _replay_cached_response(cache_lookup.response):
                    if frame.startswith("event: done\n"):
                        root.update(
                            metadata={
                                "cache_outcome": "hit",
                                **_response_trace_metadata(cache_lookup.response, cache_hit=True),
                            }
                        )
                        root.end()
                    yield frame
            except asyncio.CancelledError:
                _finish_failure(root, "cancelled")
                raise
            except GeneratorExit:
                _finish_failure(root, "cancelled")
                raise

        return StreamingResponse(
            cached_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    root.update(metadata={"cache_outcome": cache_lookup.outcome.value})
    retrieval_span = root.start_span(
        name="rag.retrieval",
        metadata={"requested_result_size": settings.rag_retrieval_size},
    )
    search = partial(hybrid_service.search, request.question, size=settings.rag_retrieval_size)
    try:
        retrieval_result = await run_in_threadpool(search)
    except HybridSearchError as exc:
        retrieval_span.update(error_type="retrieval_failure")
        retrieval_span.end()
        _finish_failure(root, "retrieval_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research retrieval is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        retrieval_span.update(error_type="retrieval_failure")
        retrieval_span.end()
        _finish_failure(root, "internal_failure", exc)
        raise
    retrieval_span.update(
        metadata={
            "returned_result_count": retrieval_result.count,
            "retrieval_mode": retrieval_result.retrieval_mode,
        }
    )
    retrieval_span.end()

    prepared: PreparedRAGGeneration | None
    try:
        prepared = rag_service.prepare(
            question=request.question,
            retrieval_result=retrieval_result,
            observation=root,
        )
    except InsufficientEvidenceError:
        prepared = None
    except RAGPromptBudgetError as exc:
        _finish_failure(root, "evidence_failure", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The question and retrieved evidence exceed the generation budget.",
        ) from exc
    except LLMConfigurationError as exc:
        _finish_failure(root, "generation_failure", exc)
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
            response = AskResponse(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                sources=[],
                retrieval_mode=retrieval_result.retrieval_mode,
                model=None,
                prompt_tokens=None,
                completion_tokens=None,
            )
            yield _sse("delta", {"text": INSUFFICIENT_EVIDENCE_ANSWER})
            write_outcome = await _store_with_trace(response_cache, cache_lookup, response, root)
            yield _sse("sources", {"sources": []})
            root.update(
                metadata={
                    "cache_write_outcome": write_outcome.value,
                    "valid_insufficiency": True,
                    **_response_trace_metadata(response, grounding_performed=False),
                }
            )
            root.end()
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
        answer_parts: list[str] = []
        generation_events = rag_service.stream_generate(prepared, observation=root)
        try:
            async for event in generation_events:
                if isinstance(event, RAGStreamDelta):
                    answer_parts.append(event.text)
                    yield _sse("delta", {"text": event.text})
                elif isinstance(event, RAGStreamComplete):
                    sources = [build_ask_source(source) for source in prepared.sources]
                    response = AskResponse(
                        answer="".join(answer_parts),
                        sources=sources,
                        retrieval_mode=prepared.retrieval_mode,
                        model=event.model,
                        prompt_tokens=event.prompt_tokens,
                        completion_tokens=event.completion_tokens,
                    )
                    write_outcome = await _store_with_trace(response_cache, cache_lookup, response, root)
                    yield _sse(
                        "sources",
                        {"sources": [source.model_dump(mode="json") for source in sources]},
                    )
                    root.update(
                        metadata={
                            "cache_write_outcome": write_outcome.value,
                            **_response_trace_metadata(response),
                        }
                    )
                    root.end()
                    yield _sse(
                        "done",
                        {
                            "model": event.model,
                            "prompt_tokens": event.prompt_tokens,
                            "completion_tokens": event.completion_tokens,
                            "citations": list(event.cited_labels),
                        },
                    )
        except asyncio.CancelledError:
            _finish_failure(root, "cancelled")
            raise
        except GeneratorExit:
            _finish_failure(root, "cancelled")
            raise
        except GroundingValidationError as exc:
            _finish_failure(root, "grounding_failure", exc)
            yield _sse(
                "error",
                {
                    "code": "grounding_validation_failed",
                    "message": "The generated answer could not be validated against the supplied evidence.",
                },
            )
        except (LLMConfigurationError, LLMRequestError, LLMResponseError) as exc:
            _finish_failure(root, "generation_failure", exc)
            yield _sse(
                "error",
                {
                    "code": "generation_failed",
                    "message": "Language model generation failed.",
                },
            )
        finally:
            await generation_events.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
