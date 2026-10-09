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
from src.schemas.ask import AskRequest, AskResponse, build_ask_response
from starlette.concurrency import run_in_threadpool

router = APIRouter()


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
            answer="The available indexed evidence is insufficient to answer this question.",
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
