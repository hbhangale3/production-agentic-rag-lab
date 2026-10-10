"""Request-level agent orchestration: prompts, cache, bounded graph execution, public mapping.

One prompt bundle is resolved per request and used for both the cache key and
the graph, so a cached answer always corresponds to the prompts that would run.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from langgraph.graph.state import CompiledStateGraph
from src.schemas.agent import (
    AgentAskResponse,
    AgentExecutionSummary,
    AgentStatus,
    build_agent_source,
)
from src.services.agent.cache_identity import AGENT_PIPELINE_VERSION
from src.services.agent.config import AgentGraphConfig
from src.services.agent.events import AgentExecutionStatus
from src.services.agent.graph import AGENT_OBSERVATION_KEY, AgentGraphDependencies, build_agent_graph
from src.services.agent.prompt_identity import AgentPromptIdentityBundle
from src.services.agent.prompts import AgentPromptBundle, resolve_agent_prompt_bundle
from src.services.agent.response_cache import AgentCachedAnswer, AgentCacheLookup, AgentResponseCache
from src.services.agent.state import AgentState, create_initial_agent_state
from src.services.agent.status import (
    ANSWER_NOT_VALIDATED_MESSAGE,
    CACHE_HIT_STATUS,
    SEMANTIC_OUTCOME_MESSAGES,
    to_public_status,
)
from src.services.agent.types import AgentErrorCategory, TerminalReason
from src.services.cache.rag_response_cache import RAGCacheOutcome, RAGCacheWriteOutcome
from src.services.llm.base import FINISH_REASON_LENGTH
from src.services.observability.base import Observation
from src.services.observability.safe import SafeObservation
from src.services.prompts.base import DEFAULT_PROMPT_LABEL, PromptResolver

logger = logging.getLogger(__name__)

StatusCallback = Callable[[AgentStatus], Awaitable[None]]
DEFAULT_COMPILED_GRAPH_CACHE_SIZE = 4


class AgentFailureKind(StrEnum):
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"
    INTERNAL_ERROR = "internal_error"
    TIMEOUT = "timeout"


class AgentExecutionFailure(Exception):
    """The agent could not produce any truthful outcome; carries only a constrained kind."""

    def __init__(self, kind: AgentFailureKind) -> None:
        super().__init__(kind.value)
        self.kind = kind


_FAILURE_KINDS = {
    TerminalReason.RETRIEVAL_FAILED: AgentFailureKind.RETRIEVAL_FAILED,
    TerminalReason.GENERATION_FAILED: AgentFailureKind.GENERATION_FAILED,
    TerminalReason.INTERNAL_ERROR: AgentFailureKind.INTERNAL_ERROR,
}
# The model responded but its answer was unusable: structurally invalid, or still cut off after the
# one length recovery. Internally these stay generation failures; publicly they are a controlled
# outcome with a fixed message, not a gateway error.
_REJECTED_ANSWER_CATEGORIES = frozenset(
    {AgentErrorCategory.ANSWER_VALIDATION_FAILURE, AgentErrorCategory.GENERATION_INCOMPLETE}
)


@dataclass(frozen=True)
class AgentPreparedRequest:
    """The exact prompt bundle and cache state one request will use."""

    question: str
    prompts: AgentPromptBundle
    lookup: AgentCacheLookup
    started_at: float

    @property
    def cache_status(self) -> str:
        return self.lookup.outcome.value


def execution_summary(state: AgentState) -> AgentExecutionSummary:
    statuses = [event.status for event in state["execution_events"]]
    generation = state["generation_result"]
    return AgentExecutionSummary(
        retrieval_attempts=state["retrieval_attempts"],
        query_rewritten=state["rewritten_query"] is not None,
        live_fallback_used=state["live_fallback_used"],
        live_papers_selected=len(state["live_selected_papers"] or ()),
        generation_attempts=statuses.count(AgentExecutionStatus.GENERATION_STARTED),
        grounding_attempts=state["grounding_attempts"],
        source_count=len(generation.sources) if generation is not None else 0,
    )


def map_agent_state(
    state: AgentState,
    *,
    cache_status: str,
    latency_ms: float | None = None,
) -> AgentAskResponse:
    """Turn a terminal state into a public response, or raise for an execution failure.

    A model-written answer is exposed only when the run ended without a
    terminal reason and the answer passed semantic grounding. An answer left
    in state after grounding was exhausted is never returned.
    """

    terminal = state["terminal_reason"]
    statuses = [to_public_status(event) for event in state["execution_events"]]
    summary = execution_summary(state)
    if terminal is TerminalReason.GENERATION_FAILED and state["error_category"] in _REJECTED_ANSWER_CATEGORIES:
        return AgentAskResponse(
            answer=ANSWER_NOT_VALIDATED_MESSAGE,
            outcome=terminal.value,
            sources=[],
            cache_status=cache_status,
            terminal_reason=terminal.value,
            grounding_passed=None,
            execution=summary.model_copy(update={"source_count": 0}),
            statuses=statuses,
            pipeline_version=AGENT_PIPELINE_VERSION,
            latency_ms=latency_ms,
        )
    if state["error_category"] is not None or terminal in _FAILURE_KINDS:
        raise AgentExecutionFailure(_FAILURE_KINDS.get(terminal, AgentFailureKind.INTERNAL_ERROR))
    if terminal in SEMANTIC_OUTCOME_MESSAGES:
        return AgentAskResponse(
            answer=SEMANTIC_OUTCOME_MESSAGES[terminal],
            outcome=terminal.value,
            sources=[],
            cache_status=cache_status,
            terminal_reason=terminal.value,
            grounding_passed=False if terminal is TerminalReason.GROUNDING_FAILED else None,
            execution=summary.model_copy(update={"source_count": 0}),
            statuses=statuses,
            pipeline_version=AGENT_PIPELINE_VERSION,
            latency_ms=latency_ms,
        )
    generation = state["generation_result"]
    if (
        terminal is not None
        or generation is None
        or not state["generated_answer"]
        or state["grounding_passed"] is not True
        or generation.answer != state["generated_answer"]
        or generation.finish_reason == FINISH_REASON_LENGTH
    ):
        raise AgentExecutionFailure(AgentFailureKind.INTERNAL_ERROR)
    return AgentAskResponse(
        answer=generation.answer,
        outcome="answered",
        # Exactly the labeled sources the final answer was generated from and grounded against.
        sources=[build_agent_source(source) for source in generation.sources],
        cache_status=cache_status,
        terminal_reason=None,
        grounding_passed=True,
        model=generation.model,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        execution=summary,
        statuses=statuses,
        pipeline_version=AGENT_PIPELINE_VERSION,
        latency_ms=latency_ms,
    )


def is_cacheable(state: AgentState, response: AgentAskResponse) -> bool:
    """Only a complete, grounded, cited answer from a clean terminal state may be cached."""

    generation = state["generation_result"]
    return (
        response.outcome == "answered"
        and state["terminal_reason"] is None
        and state["error_category"] is None
        and state["grounding_passed"] is True
        and generation is not None
        and generation.finish_reason != FINISH_REASON_LENGTH
        and bool(generation.cited_labels)
        and bool(response.sources)
    )


class AgentService:
    """Worker-scoped entry point used by the agent API."""

    def __init__(
        self,
        *,
        dependencies: AgentGraphDependencies,
        graph_config: AgentGraphConfig,
        prompt_resolver: PromptResolver,
        response_cache: AgentResponseCache,
        prompt_label: str = DEFAULT_PROMPT_LABEL,
        prompt_timeout_seconds: float = 5.0,
        request_timeout_seconds: float = 540.0,
        clock: Callable[[], float] = time.monotonic,
        compiled_graph_cache_size: int = DEFAULT_COMPILED_GRAPH_CACHE_SIZE,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if compiled_graph_cache_size < 1:
            raise ValueError("compiled_graph_cache_size must be at least 1")
        self.dependencies = dependencies
        self.graph_config = graph_config
        self.prompt_resolver = prompt_resolver
        self.response_cache = response_cache
        self.prompt_label = prompt_label
        self.prompt_timeout_seconds = prompt_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._clock = clock
        self._graphs: OrderedDict[AgentPromptIdentityBundle, CompiledStateGraph] = OrderedDict()
        self._graph_cache_size = compiled_graph_cache_size
        self._on_close = on_close

    async def aclose(self) -> None:
        if self._on_close is not None:
            await self._on_close()

    async def prepare(self, question: str, *, observation: Observation | None = None) -> AgentPreparedRequest:
        """Resolve the prompt bundle once and look the answer up under its exact identity."""

        started_at = self._clock()
        root = SafeObservation(observation) if observation is not None else None
        prompt_span = root.start_span(name="agent.prompt_resolution", metadata=None) if root else None
        prompts = await resolve_agent_prompt_bundle(
            self.prompt_resolver,
            label=self.prompt_label,
            timeout_seconds=self.prompt_timeout_seconds,
        )
        identities = prompts.identities
        if prompt_span:
            versions = {name: payload["version"] for name, payload in identities.as_cache_payload().items()}
            prompt_span.update(metadata={"prompt_label": self.prompt_label, "prompt_versions": versions})
            prompt_span.end()
        cache_span = (
            root.start_span(name="agent.cache.lookup", metadata={"cache_enabled": self.response_cache.enabled})
            if root
            else None
        )
        lookup = await self.response_cache.lookup(question=question, prompts=identities)
        if cache_span:
            cache_span.update(
                metadata={"outcome": lookup.outcome.value},
                error_type="cache_read_failure" if lookup.outcome is RAGCacheOutcome.FAILURE else None,
            )
            cache_span.end()
        return AgentPreparedRequest(question=question, prompts=prompts, lookup=lookup, started_at=started_at)

    def cached_response(self, prepared: AgentPreparedRequest) -> AgentAskResponse | None:
        """The validated cached answer for this request, if any; no graph work is involved."""

        cached = prepared.lookup.answer
        if cached is None:
            return None
        return AgentAskResponse(
            answer=cached.answer,
            outcome="answered",
            sources=list(cached.sources),
            cache_status=RAGCacheOutcome.HIT.value,
            terminal_reason=None,
            grounding_passed=True,
            model=cached.model,
            prompt_tokens=cached.prompt_tokens,
            completion_tokens=cached.completion_tokens,
            execution=cached.execution,
            statuses=[CACHE_HIT_STATUS],
            pipeline_version=AGENT_PIPELINE_VERSION,
            latency_ms=self._elapsed_ms(prepared),
        )

    async def execute(
        self,
        prepared: AgentPreparedRequest,
        *,
        observation: Observation | None = None,
        on_status: StatusCallback | None = None,
    ) -> AgentAskResponse:
        """Run the graph with the prepared bundle under the request deadline, then cache if eligible."""

        remaining = self.request_timeout_seconds - (self._clock() - prepared.started_at)
        graph = self._graph_for(prepared.prompts)
        config = {"configurable": {AGENT_OBSERVATION_KEY: observation}} if observation is not None else None
        state: AgentState | None = None
        emitted = 0
        try:
            async with asyncio.timeout(max(remaining, 0.0)):
                async for snapshot in graph.astream(
                    create_initial_agent_state(prepared.question),
                    config=config,
                    stream_mode="values",
                ):
                    state = snapshot
                    for event in snapshot["execution_events"][emitted:]:
                        emitted += 1
                        if on_status is not None:
                            await on_status(to_public_status(event))
        except TimeoutError as exc:
            logger.warning("Agent request exceeded its deadline")
            raise AgentExecutionFailure(AgentFailureKind.TIMEOUT) from exc
        except AgentExecutionFailure:
            raise
        except Exception as exc:
            logger.warning("Agent graph execution failed (%s)", type(exc).__name__)
            raise AgentExecutionFailure(AgentFailureKind.INTERNAL_ERROR) from exc
        if state is None:
            raise AgentExecutionFailure(AgentFailureKind.INTERNAL_ERROR)

        response = map_agent_state(state, cache_status=prepared.cache_status)
        if is_cacheable(state, response):
            await self._store(prepared, response, observation)
        return response.model_copy(update={"latency_ms": self._elapsed_ms(prepared)})

    async def answer(
        self,
        question: str,
        *,
        observation: Observation | None = None,
        on_status: StatusCallback | None = None,
    ) -> AgentAskResponse:
        """Cache hit when available, otherwise one bounded graph run."""

        prepared = await self.prepare(question, observation=observation)
        cached = self.cached_response(prepared)
        if cached is not None:
            return cached
        return await self.execute(prepared, observation=observation, on_status=on_status)

    @staticmethod
    def trace_metadata(response: AgentAskResponse) -> dict[str, object]:
        """Content-free request summary for the root observation."""

        hit = response.cache_status == RAGCacheOutcome.HIT.value
        prefix = "artifact_" if hit else ""
        return {
            "status": "success",
            "cache_outcome": response.cache_status,
            "outcome": response.outcome,
            "terminal_reason": response.terminal_reason,
            "grounding_passed": response.grounding_passed,
            "pipeline_version": response.pipeline_version,
            "source_count": len(response.sources),
            f"{prefix}retrieval_attempts": response.execution.retrieval_attempts,
            f"{prefix}live_fallback_used": response.execution.live_fallback_used,
            f"{prefix}generation_attempts": response.execution.generation_attempts,
            f"{prefix}grounding_attempts": response.execution.grounding_attempts,
            f"{prefix}model": response.model,
            f"{prefix}prompt_tokens": response.prompt_tokens,
            f"{prefix}completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
        }

    async def _store(
        self,
        prepared: AgentPreparedRequest,
        response: AgentAskResponse,
        observation: Observation | None,
    ) -> None:
        if prepared.lookup.key is None:
            return
        span = SafeObservation(observation).start_span(name="agent.cache.write", metadata=None) if observation else None
        try:
            cached = AgentCachedAnswer(
                answer=response.answer,
                sources=response.sources,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                execution=response.execution,
            )
            outcome = await self.response_cache.store(prepared.lookup, cached)
        except Exception:
            outcome = RAGCacheWriteOutcome.FAILED
        if span:
            span.update(
                metadata={"outcome": outcome.value},
                error_type="cache_write_failure" if outcome is RAGCacheWriteOutcome.FAILED else None,
            )
            span.end()

    def _graph_for(self, prompts: AgentPromptBundle) -> CompiledStateGraph:
        """Reuse a compiled graph per prompt identity; a small LRU, since identities rarely change."""

        identities = prompts.identities
        graph = self._graphs.get(identities)
        if graph is None:
            graph = build_agent_graph(dependencies=self.dependencies, config=self.graph_config, prompts=prompts)
            self._graphs[identities] = graph
            while len(self._graphs) > self._graph_cache_size:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(identities)
        return graph

    def _elapsed_ms(self, prepared: AgentPreparedRequest) -> float:
        return max(0.0, round((self._clock() - prepared.started_at) * 1000, 3))
