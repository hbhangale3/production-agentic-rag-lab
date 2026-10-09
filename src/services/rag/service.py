import asyncio
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from src.exceptions import InsufficientEvidenceError, RAGPromptBudgetError
from src.schemas.hybrid_search import HybridSearchResult
from src.services.evidence import EvidenceContext, EvidenceContextBuilder, EvidenceSource, TokenCounter
from src.services.llm.base import ChatMessage, LLMProvider
from src.services.observability import Observation
from src.services.rag.prompt import RAGPromptBuilder
from src.services.rag.validation import FULLWIDTH_CITATION, GroundedAnswerValidator


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


@dataclass(frozen=True)
class PreparedRAGGeneration:
    messages: tuple[ChatMessage, ...]
    sources: tuple[EvidenceSource, ...]
    retrieval_mode: str
    estimated_prompt_tokens: int


@dataclass(frozen=True)
class RAGStreamDelta:
    text: str


@dataclass(frozen=True)
class RAGStreamComplete:
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cited_labels: tuple[str, ...]


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
        answer_validator: GroundedAnswerValidator | None = None,
    ) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self._require_positive_integer(max_completion_tokens, "max_completion_tokens")
        self._require_positive_integer(context_window_tokens, "context_window_tokens")
        if isinstance(token_safety_margin, bool) or not isinstance(token_safety_margin, int) or token_safety_margin < 0:
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
        self.answer_validator = answer_validator or GroundedAnswerValidator()

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
        observation: Observation | None = None,
    ) -> RAGGenerationResult:
        prepared = self.prepare(
            question=question,
            retrieval_result=retrieval_result,
            observation=observation,
        )
        return await self._generate_prepared(prepared, observation=observation)

    async def generate_from_context(
        self,
        *,
        question: str,
        evidence: EvidenceContext,
        observation: Observation | None = None,
    ) -> RAGGenerationResult:
        """Generate from an already-built context using the same prompt, budget, and validation."""
        prepared = self.prepare_from_context(question=question, evidence=evidence, observation=observation)
        return await self._generate_prepared(prepared, observation=observation)

    async def _generate_prepared(
        self,
        prepared: PreparedRAGGeneration,
        *,
        observation: Observation | None,
    ) -> RAGGenerationResult:
        generation_span = self._start_span(
            observation,
            "rag.generation",
            {
                "streaming": False,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_completion_tokens,
            },
        )
        try:
            completion = await self.llm_provider.complete(
                prepared.messages,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
            )
        except Exception as exc:
            self._finish_span(generation_span, error_type=type(exc).__name__)
            raise
        self._finish_span(
            generation_span,
            metadata={
                "model": completion.model,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
        )

        grounding_span = self._start_span(
            observation,
            "rag.grounding",
            {"source_count": len(prepared.sources)},
        )
        try:
            validated = self.answer_validator.validate(completion.content, prepared.sources)
        except Exception as exc:
            self._finish_span(
                grounding_span,
                metadata={"passed": False},
                error_type=type(exc).__name__,
            )
            raise
        self._finish_span(
            grounding_span,
            metadata={
                "passed": True,
                "citation_count": len(validated.cited_labels),
                "citation_free_insufficiency": validated.is_insufficiency,
            },
        )

        return RAGGenerationResult(
            answer=validated.answer,
            sources=prepared.sources,
            retrieval_mode=prepared.retrieval_mode,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cited_labels=validated.cited_labels,
            estimated_prompt_tokens=prepared.estimated_prompt_tokens,
            token_counting=self.token_counter.description,
        )

    def prepare(
        self,
        *,
        question: str,
        retrieval_result: HybridSearchResult,
        observation: Observation | None = None,
    ) -> PreparedRAGGeneration:
        """Build and validate all deterministic inputs before generation starts."""
        evidence_span = self._start_span(
            observation,
            "rag.evidence",
            {
                "retrieved_count": retrieval_result.count,
                "context_token_budget": self.evidence_builder.max_context_tokens,
            },
        )
        return self._prepare(
            question=question,
            build_evidence=lambda: self.evidence_builder.build(retrieval_result),
            evidence_span=evidence_span,
        )

    def prepare_from_context(
        self,
        *,
        question: str,
        evidence: EvidenceContext,
        observation: Observation | None = None,
    ) -> PreparedRAGGeneration:
        """Validate an already-built context against the same generation budget."""
        evidence_span = self._start_span(
            observation,
            "rag.evidence",
            {
                "retrieved_count": len(evidence.sources),
                "context_token_budget": evidence.max_context_tokens,
            },
        )
        return self._prepare(question=question, build_evidence=lambda: evidence, evidence_span=evidence_span)

    def _prepare(
        self,
        *,
        question: str,
        build_evidence: Callable[[], EvidenceContext],
        evidence_span: Observation | None,
    ) -> PreparedRAGGeneration:
        try:
            normalized_question = self.prompt_builder.normalize_question(question)
            evidence = build_evidence()
            evidence_metadata: dict[str, object] = {
                "selected_source_count": len(evidence.sources),
                "context_tokens": evidence.estimated_tokens,
                "truncated_source_count": sum(source.truncated for source in evidence.sources),
                "retrieval_mode": evidence.retrieval_mode,
            }
            if not evidence.text.strip() or not evidence.sources:
                evidence_metadata["insufficient_evidence"] = True
                self._finish_span(evidence_span, metadata=evidence_metadata)
                raise InsufficientEvidenceError("No usable retrieved evidence is available")
            messages = tuple(self.prompt_builder.build(question=normalized_question, evidence=evidence))
            estimated_prompt_tokens = sum(self.token_counter.count(message.content) for message in messages)
            required_tokens = estimated_prompt_tokens + self.max_completion_tokens + self.token_safety_margin
            if required_tokens > self.context_window_tokens:
                raise RAGPromptBudgetError(
                    "Estimated prompt, completion allowance, and safety margin exceed the context window"
                )
        except InsufficientEvidenceError:
            raise
        except Exception as exc:
            self._finish_span(evidence_span, error_type=type(exc).__name__)
            raise
        self._finish_span(evidence_span, metadata=evidence_metadata)
        return PreparedRAGGeneration(
            messages=messages,
            sources=evidence.sources,
            retrieval_mode=evidence.retrieval_mode,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )

    async def stream_generate(
        self,
        prepared: PreparedRAGGeneration,
        *,
        observation: Observation | None = None,
    ) -> AsyncIterator[RAGStreamDelta | RAGStreamComplete]:
        """Stream canonical text, then validate citations and emit completion metadata."""
        normalizer = _FullwidthCitationStreamNormalizer()
        answer_parts: list[str] = []
        model: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        generation_span = self._start_span(
            observation,
            "rag.generation",
            {
                "streaming": True,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_completion_tokens,
            },
        )
        generation_finished = False
        try:
            async for event in self.llm_provider.stream(
                prepared.messages,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
            ):
                model = event.model or model
                prompt_tokens = event.prompt_tokens if event.prompt_tokens is not None else prompt_tokens
                completion_tokens = event.completion_tokens if event.completion_tokens is not None else completion_tokens
                if event.text is None:
                    continue
                text = normalizer.feed(event.text)
                if text:
                    answer_parts.append(text)
                    yield RAGStreamDelta(text=text)
            generation_finished = True
        except asyncio.CancelledError:
            self._finish_span(generation_span, error_type="cancelled")
            generation_finished = True
            raise
        except Exception as exc:
            self._finish_span(generation_span, error_type=type(exc).__name__)
            generation_finished = True
            raise
        finally:
            if not generation_finished:
                # Async-generator close/client disconnect before provider completion.
                self._finish_span(generation_span, error_type="cancelled")
        tail = normalizer.finish()
        if tail:
            answer_parts.append(tail)
            yield RAGStreamDelta(text=tail)
        self._finish_span(
            generation_span,
            metadata={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        answer = "".join(answer_parts)
        grounding_span = self._start_span(
            observation,
            "rag.grounding",
            {"source_count": len(prepared.sources)},
        )
        try:
            validated = self.answer_validator.validate(answer, prepared.sources)
        except Exception as exc:
            self._finish_span(
                grounding_span,
                metadata={"passed": False},
                error_type=type(exc).__name__,
            )
            raise
        self._finish_span(
            grounding_span,
            metadata={
                "passed": True,
                "citation_count": len(validated.cited_labels),
                "citation_free_insufficiency": validated.is_insufficiency,
            },
        )
        yield RAGStreamComplete(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cited_labels=validated.cited_labels,
        )

    @staticmethod
    def _start_span(
        observation: Observation | None,
        name: str,
        metadata: dict[str, object],
    ) -> Observation | None:
        return observation.start_span(name=name, metadata=metadata) if observation is not None else None

    @staticmethod
    def _finish_span(
        observation: Observation | None,
        *,
        metadata: dict[str, object] | None = None,
        error_type: str | None = None,
    ) -> None:
        if observation is None:
            return
        observation.update(metadata=metadata, error_type=error_type)
        observation.end()

    @staticmethod
    def _require_positive_integer(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


class _FullwidthCitationStreamNormalizer:
    """Canonicalize full-width citations while retaining only a possible suffix."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        combined = FULLWIDTH_CITATION.sub(r"[S\1]", self._pending + text)
        self._pending = ""
        marker = combined.rfind("【")
        if marker >= 0 and re.fullmatch(r"【(?:S\d*)?", combined[marker:]):
            self._pending = combined[marker:]
            return combined[:marker]
        return combined

    def finish(self) -> str:
        pending, self._pending = self._pending, ""
        return pending
