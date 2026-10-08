from fastapi import APIRouter, HTTPException, status
from src.dependencies import HybridSearchServiceDep
from src.exceptions import HybridSearchError
from src.schemas.hybrid_search import HybridSearchRequest, HybridSearchResult

router = APIRouter(prefix="/hybrid-search", tags=["Search"])


@router.post("", response_model=HybridSearchResult, summary="Chunk-level hybrid RRF search")
def hybrid_search(
    request: HybridSearchRequest, service: HybridSearchServiceDep
) -> HybridSearchResult:
    try:
        return service.search(
            request.query,
            size=request.size,
            categories=request.categories,
            published_from=request.published_from,
            published_to=request.published_to,
        )
    except HybridSearchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
