from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from src.config import Settings
from src.exceptions import LLMConfigurationError
from src.main import lifespan


def startup_dependencies():
    database = Mock()
    chunk_index = Mock()
    embedding = Mock()
    provider = Mock()
    provider.close = AsyncMock()
    settings = Settings(_env_file=None, debug=False, groq_api_key="test-only-key")
    return database, chunk_index, embedding, provider, settings


@pytest.mark.anyio
async def test_lifespan_builds_worker_scoped_rag_once_and_closes_provider() -> None:
    database, chunk_index, embedding, provider, settings = startup_dependencies()
    cache = Mock()
    cache.close = AsyncMock()
    application = FastAPI()

    with (
        patch("src.main.get_settings", return_value=settings),
        patch("src.main.make_database", return_value=database),
        patch("src.main.make_chunk_index_manager", return_value=chunk_index),
        patch("src.main.make_embedding_provider", return_value=embedding),
        patch("src.main.make_cache", return_value=cache) as make_cache,
        patch("src.main.make_llm_provider", return_value=provider) as make_provider,
    ):
        async with lifespan(application):
            make_cache.assert_called_once_with(settings)
            assert application.state.cache is cache
            make_provider.assert_called_once_with(settings)
            assert application.state.llm_provider is provider
            assert application.state.rag_generation_service.llm_provider is provider
            provider.close.assert_not_awaited()
            cache.close.assert_not_awaited()

    provider.close.assert_awaited_once_with()
    cache.close.assert_awaited_once_with()
    database.teardown.assert_called_once_with()
    chunk_index.close.assert_called_once_with()


@pytest.mark.anyio
async def test_lifespan_without_credentials_keeps_non_rag_services_available() -> None:
    database, chunk_index, embedding, _, settings = startup_dependencies()
    cache = Mock()
    cache.close = AsyncMock()
    application = FastAPI()

    with (
        patch("src.main.get_settings", return_value=settings),
        patch("src.main.make_database", return_value=database),
        patch("src.main.make_chunk_index_manager", return_value=chunk_index),
        patch("src.main.make_embedding_provider", return_value=embedding),
        patch("src.main.make_cache", return_value=cache),
        patch(
            "src.main.make_llm_provider",
            side_effect=LLMConfigurationError("test missing credential"),
        ),
    ):
        async with lifespan(application):
            assert application.state.hybrid_search_service is not None
            assert application.state.llm_provider is None
            assert application.state.rag_generation_service is None
            assert application.state.cache is cache

    database.teardown.assert_called_once_with()
    chunk_index.close.assert_called_once_with()
    cache.close.assert_awaited_once_with()
