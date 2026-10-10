"""Build the worker-scoped agent service from already-created application services."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.config import Settings
from src.schemas.parsed_pdf import ParsedPDF
from src.services.agent.config import AgentGraphConfig
from src.services.agent.graph import AgentGraphDependencies
from src.services.agent.live_documents import LiveDocumentProcessingService, LivePaperAcquisitionService
from src.services.agent.live_search import ArxivLiveSearchService
from src.services.agent.response_cache import AgentResponseCache
from src.services.agent.service import AgentService
from src.services.arxiv.client import ArxivClient
from src.services.cache.base import AsyncCache
from src.services.chunking import PaperChunkingService
from src.services.embeddings.base import EmbeddingProvider
from src.services.llm import LLMProvider
from src.services.observability import ObservabilityProvider, make_prompt_resolver
from src.services.rag.service import RAGGenerationService, llm_telemetry_from_settings
from src.services.search.hybrid_service import HybridSearchService

logger = logging.getLogger(__name__)


class LazyPDFParser:
    """Create the Docling parser on first use and parse one PDF at a time per worker.

    Importing Docling is slow and memory-heavy, so it is deferred until a live
    paper is actually processed. The lock keeps concurrent requests from
    running several CPU-bound conversions at once.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._parser: Any | None = None
        self._lock = asyncio.Lock()

    async def parse_pdf(self, pdf_path: str | Path) -> ParsedPDF:
        async with self._lock:
            if self._parser is None:
                self._parser = await asyncio.to_thread(self._make_parser)
            return await self._parser.parse_pdf(pdf_path)

    def _make_parser(self) -> Any:
        from src.services.pdf_parser.factory import make_pdf_parser_service

        return make_pdf_parser_service(self._settings)


def agent_graph_config_from_settings(settings: Settings) -> AgentGraphConfig:
    return AgentGraphConfig(retrieval_size=settings.rag_retrieval_size)


def make_agent_service(
    *,
    settings: Settings,
    llm_provider: LLMProvider,
    hybrid_search_service: HybridSearchService,
    embedding_provider: EmbeddingProvider,
    rag_generation_service: RAGGenerationService,
    cache: AsyncCache,
    observability: ObservabilityProvider,
) -> AgentService:
    """Reuse the worker's LLM, retrieval, embedding, generation, cache, and telemetry services."""

    graph_config = agent_graph_config_from_settings(settings)
    # A dedicated client so online requests use short bounds and never share ingestion's cache directory.
    arxiv_client = ArxivClient(
        base_url=settings.arxiv_api_base_url,
        max_results=graph_config.live_arxiv_max_results,
        rate_limit_delay=settings.arxiv_rate_limit_delay,
        timeout=settings.agent_live_arxiv_timeout_seconds,
        max_retries=settings.agent_live_arxiv_max_retries,
    )
    dependencies = AgentGraphDependencies(
        llm_provider=llm_provider,
        hybrid_search_service=hybrid_search_service,
        evidence_context_builder=rag_generation_service.evidence_builder,
        embedding_provider=embedding_provider,
        rag_generation_service=rag_generation_service,
        live_search_service=ArxivLiveSearchService(arxiv_client=arxiv_client),
        live_document_processor=LiveDocumentProcessingService(
            acquisition_service=LivePaperAcquisitionService(arxiv_client=arxiv_client),
            pdf_parser=LazyPDFParser(settings),
            chunking_service=PaperChunkingService.from_settings(settings),
            paper_timeout_seconds=settings.agent_live_paper_timeout_seconds,
        ),
        llm_telemetry=llm_telemetry_from_settings(settings),
    )
    return AgentService(
        dependencies=dependencies,
        graph_config=graph_config,
        prompt_resolver=make_prompt_resolver(settings, observability),
        response_cache=AgentResponseCache(cache=cache, settings=settings, graph_config=graph_config),
        prompt_label=settings.langfuse_prompt_label,
        prompt_timeout_seconds=settings.langfuse_timeout_seconds,
        request_timeout_seconds=settings.agent_request_timeout_seconds,
        on_close=arxiv_client.aclose,
    )
