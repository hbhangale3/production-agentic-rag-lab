import re
from dataclasses import dataclass

from src.exceptions import InsufficientEvidenceError, LLMResponseError, RAGPromptBudgetError
from src.schemas.hybrid_search import HybridSearchResult
from src.services.evidence import EvidenceContextBuilder, EvidenceSource, TokenCounter
from src.services.llm.base import LLMProvider
from src.services.rag.prompt import RAGPromptBuilder

CITATION_LABEL = re.compile(r"\[S\d+\]")
FULLWIDTH_CITATION_LABEL = re.compile(r"【S(\d+)】")


@dataclass(frozen=True)
class RAGGenerationResult:
    """Normalized answer with application-owned source attribution."""

    answer: str
    sources: tuple[EvidenceSource, ...]
    retrieval_mode: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cited_labels: tuple[str, ...]
    estimated_prompt_tokens: int
    token_counting: str


class RAGGenerationService:
    """Build evidence and generate one grounded answer asynchronously."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        evidence_builder: EvidenceContextBuilder,
        prompt_builder: RAGPromptBuilder | None = None,
        temperature: float = 0.1,
        max_completion_tokens: int = 512,
        context_window_tokens: int = 8192,
        token_safety_margin: int = 256,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        self._require_positive_integer(max_completion_tokens, "max_completion_tokens")
        self._require_positive_integer(context_window_tokens, "context_window_tokens")
        if (
            isinstance(token_safety_margin, bool)
            or not isinstance(token_safety_margin, int)
            or token_safety_margin < 0
        ):
            raise ValueError("token_safety_margin must be a non-negative integer")
        if max_completion_tokens + token_safety_margin >= context_window_tokens:
            raise ValueError("completion allowance and safety margin must leave prompt capacity")

        self.llm_provider = llm_provider
        self.evidence_builder = evidence_builder
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.temperature = float(temperature)
        self.max_completion_tokens = max_completion_tokens
        self.context_window_tokens = context_window_tokens
        self.token_safety_margin = token_safety_margin
        self.token_counter = token_counter or evidence_builder.token_counter

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: LLMProvider,
        evidence_builder: EvidenceContextBuilder,
        settings: object,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> "RAGGenerationService":
        return cls(
            llm_provider=llm_provider,
            evidence_builder=evidence_builder,
            prompt_builder=prompt_builder,
            temperature=getattr(settings, "llm_temperature"),
            max_completion_tokens=getattr(settings, "llm_max_completion_tokens"),
            context_window_tokens=getattr(settings, "llm_context_window_tokens"),
            token_safety_margin=getattr(settings, "llm_token_safety_margin"),
        )

    async def generate(
        self,
        *,
        question: str,
        retrieval_result: HybridSearchResult,
    ) -> RAGGenerationResult:
        normalized_question = self.prompt_builder.normalize_question(question)
        evidence = self.evidence_builder.build(retrieval_result)
        if not evidence.text.strip() or not evidence.sources:
            raise InsufficientEvidenceError("No usable retrieved evidence is available")

        messages = self.prompt_builder.build(question=normalized_question, evidence=evidence)
        estimated_prompt_tokens = sum(
            self.token_counter.count(message.content) for message in messages
        )
        required_tokens = (
            estimated_prompt_tokens + self.max_completion_tokens + self.token_safety_margin
        )
        if required_tokens > self.context_window_tokens:
            raise RAGPromptBudgetError(
                "Estimated prompt, completion allowance, and safety margin exceed the context window"
            )

        completion = await self.llm_provider.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_completion_tokens,
        )
        answer = FULLWIDTH_CITATION_LABEL.sub(r"[S\1]", completion.content)
        cited_labels = tuple(dict.fromkeys(CITATION_LABEL.findall(answer)))
        available_labels = {source.label for source in evidence.sources}
        if any(label not in available_labels for label in cited_labels):
            raise LLMResponseError("LLM answer contained an unsupported citation label")

        return RAGGenerationResult(
            answer=answer,
            sources=evidence.sources,
            retrieval_mode=evidence.retrieval_mode,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cited_labels=cited_labels,
            estimated_prompt_tokens=estimated_prompt_tokens,
            token_counting=self.token_counter.description,
        )

    @staticmethod
    def _require_positive_integer(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
