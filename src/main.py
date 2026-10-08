import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database

# Week 1: No complex middleware needed
from src.routers import ask, hybrid_search, papers, ping, search
from src.services.embeddings.factory import make_embedding_provider
from src.services.opensearch.factory import make_chunk_index_manager
from src.services.search.chunk_bm25_service import ChunkBM25SearchService
from src.services.search.hybrid_service import HybridSearchService
from src.services.search.vector_service import VectorSearchService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Week 1: Simplified lifespan for learning purposes."""
    logger.info("Starting RAG API...")

    # Initialize settings and database (Week 1 essentials)
    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Placeholders for future weeks
    app.state.pdf_parser_service = None
    app.state.opensearch_service = None
    app.state.llm_service = None

    chunk_index = make_chunk_index_manager(settings)
    embedding_provider = make_embedding_provider(settings)
    app.state.embedding_provider = embedding_provider
    app.state.hybrid_search_service = HybridSearchService(
        bm25_service=ChunkBM25SearchService(
            chunk_index=chunk_index,
            max_results=settings.vector_search_max_results,
        ),
        vector_service=VectorSearchService(
            embedding_provider=embedding_provider,
            chunk_index=chunk_index,
            embedding_dimension=settings.embedding_dimension,
            max_results=settings.vector_search_max_results,
        ),
        rrf_k=settings.hybrid_rrf_k,
        candidate_multiplier=settings.hybrid_candidate_multiplier,
        max_results=settings.vector_search_max_results,
    )

    logger.info("API ready")
    yield

    # Cleanup
    database.teardown()
    chunk_index.close()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
for router in (ping.router, papers.router, ask.router, search.router, hybrid_search.router):
    app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
