from groq import AsyncGroq
from src.config import Settings, get_settings
from src.exceptions import LLMConfigurationError
from src.services.llm.base import LLMProvider
from src.services.llm.groq_provider import GroqLLMProvider


def make_llm_provider(
    settings: Settings | None = None,
    *,
    client: AsyncGroq | None = None,
) -> LLMProvider:
    """Build the configured LLM provider without making a network request."""
    config = settings or get_settings()
    if config.llm_provider != "groq":
        raise LLMConfigurationError(f"Unsupported LLM provider: {config.llm_provider}")

    if client is not None:
        return GroqLLMProvider(client=client, model=config.groq_model, owns_client=False)
    if config.groq_api_key is None or not config.groq_api_key.get_secret_value().strip():
        raise LLMConfigurationError("GROQ_API_KEY is required for the Groq LLM provider")

    owned_client = AsyncGroq(
        api_key=config.groq_api_key.get_secret_value(),
        timeout=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )
    return GroqLLMProvider(client=owned_client, model=config.groq_model, owns_client=True)
