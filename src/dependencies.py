from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, HTTPException, Request, status

# Week 1: Removed API key authentication for simplicity
from sqlalchemy.orm import Session
from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.cache import RAGResponseCacheCoordinator
from src.services.observability import ObservabilityProvider
from src.services.opensearch.factory import make_opensearch_client
from src.services.rag import RAGGenerationService
from src.services.search.hybrid_service import HybridSearchService
from src.services.search.paper_service import PaperSearchService

# Week 1: Simplified - no API key authentication needed for local learning

@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


# Week 2+: PDF parser service (not implemented in Week 1)
def get_pdf_parser_service(request: Request):
    """Get PDF parser service from app state (Week 2+ - not implemented yet)."""
    return None


# Week 1: OpenSearch service (placeholder - full implementation in Week 3+)
def get_opensearch_service(request: Request):
    """Get OpenSearch service from app state (Week 3+ - placeholder for Week 1)."""
    return getattr(request.app.state, "opensearch_service", None)


def get_paper_search_service(request: Request) -> Generator[PaperSearchService, None, None]:
    """Provide a request-scoped search service and close its OpenSearch transport."""
    client = make_opensearch_client(request.app.state.settings)
    try:
        yield PaperSearchService(client)
    finally:
        client.close()


def get_hybrid_search_service(request: Request) -> HybridSearchService:
    """Return the worker-scoped hybrid service and its shared lazy BGE provider."""
    return request.app.state.hybrid_search_service


def get_rag_generation_service(request: Request) -> RAGGenerationService:
    """Return the worker-scoped RAG service or a safe configuration failure."""
    service = request.app.state.rag_generation_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model service is not configured.",
        )
    return service


def get_rag_response_cache(request: Request) -> RAGResponseCacheCoordinator:
    """Return the worker-scoped exact-response cache coordinator."""
    return request.app.state.rag_response_cache


def get_observability_provider(request: Request) -> ObservabilityProvider:
    """Return the worker-scoped provider-neutral observability boundary."""
    return request.app.state.observability


# Dependency type aliases for better type hints
SettingsDep = Annotated[Settings, Depends(get_settings)]
RequestSettingsDep = Annotated[Settings, Depends(get_request_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
PDFParserServiceDep = Annotated[object, Depends(get_pdf_parser_service)]
OpenSearchServiceDep = Annotated[object, Depends(get_opensearch_service)]
PaperSearchServiceDep = Annotated[PaperSearchService, Depends(get_paper_search_service)]
HybridSearchServiceDep = Annotated[HybridSearchService, Depends(get_hybrid_search_service)]
RAGGenerationServiceDep = Annotated[RAGGenerationService, Depends(get_rag_generation_service)]
RAGResponseCacheDep = Annotated[RAGResponseCacheCoordinator, Depends(get_rag_response_cache)]
ObservabilityDep = Annotated[ObservabilityProvider, Depends(get_observability_provider)]
