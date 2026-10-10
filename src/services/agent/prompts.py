"""The agent's prompt catalog and the per-request bundle of resolved prompts.

Resolve one bundle at request start, derive the cache key from its
identities, and build the graph with the same bundle so the prompts that ran
are exactly the prompts that were keyed.
"""

import asyncio
import logging
from dataclasses import dataclass, fields

from src.services.agent.answer_grounding import AnswerGroundingPromptBuilder
from src.services.agent.evidence_grader import EvidenceGraderPromptBuilder
from src.services.agent.guardrail import GuardrailPromptBuilder
from src.services.agent.live_selection import LivePaperSelectionPromptBuilder
from src.services.agent.prompt_identity import AgentPromptIdentityBundle
from src.services.agent.query_rewriter import QueryRewritePromptBuilder
from src.services.prompts.base import (
    DEFAULT_PROMPT_LABEL,
    PromptDefinition,
    PromptResolver,
    ResolvedPrompt,
    local_prompt,
)
from src.services.rag.prompt import RAG_GENERATION_PROMPT, RAG_REGENERATION_PROMPT

logger = logging.getLogger(__name__)

# One window for the whole bundle; pass the configured Langfuse timeout in production.
DEFAULT_BUNDLE_RESOLUTION_TIMEOUT_SECONDS = 5.0

AGENT_PROMPT_DEFINITIONS: dict[str, PromptDefinition] = {
    "guardrail": GuardrailPromptBuilder.definition(),
    "evidence_grader": EvidenceGraderPromptBuilder.definition(),
    "query_rewrite": QueryRewritePromptBuilder.definition(),
    "live_selector": LivePaperSelectionPromptBuilder.definition(),
    "generation": RAG_GENERATION_PROMPT,
    "regeneration": RAG_REGENERATION_PROMPT,
    "answer_grounding": AnswerGroundingPromptBuilder.definition(),
}


@dataclass(frozen=True)
class AgentPromptBundle:
    """The effective templates for one agent run."""

    guardrail: ResolvedPrompt
    evidence_grader: ResolvedPrompt
    query_rewrite: ResolvedPrompt
    live_selector: ResolvedPrompt
    generation: ResolvedPrompt
    regeneration: ResolvedPrompt
    answer_grounding: ResolvedPrompt

    def __post_init__(self) -> None:
        for field in fields(self):
            prompt = getattr(self, field.name)
            if not isinstance(prompt, ResolvedPrompt) or prompt.name != AGENT_PROMPT_DEFINITIONS[field.name].name:
                raise ValueError(f"agent prompt bundle field {field.name} holds the wrong prompt")

    @property
    def identities(self) -> AgentPromptIdentityBundle:
        return AgentPromptIdentityBundle(**{field.name: getattr(self, field.name).identity for field in fields(self)})


def local_agent_prompt_bundle(*, label: str = DEFAULT_PROMPT_LABEL) -> AgentPromptBundle:
    """The bundle of local fallbacks; needs no I/O."""

    return AgentPromptBundle(
        **{key: local_prompt(definition, label=label) for key, definition in AGENT_PROMPT_DEFINITIONS.items()}
    )


async def _resolve_one(resolver: PromptResolver, definition: PromptDefinition, label: str) -> ResolvedPrompt:
    """Resolve one prompt, falling back to its local definition on any failure or invalid result."""

    try:
        prompt = await resolver.resolve(definition, label=label)
        if not isinstance(prompt, ResolvedPrompt) or prompt.name != definition.name:
            raise ValueError("resolver returned the wrong prompt")
        if any(marker not in prompt.content for marker in definition.required_markers):
            raise ValueError("resolved prompt is missing a required contract marker")
    except Exception:
        return local_prompt(definition, label=label)
    return prompt


async def resolve_agent_prompt_bundle(
    resolver: PromptResolver,
    *,
    label: str = DEFAULT_PROMPT_LABEL,
    timeout_seconds: float = DEFAULT_BUNDLE_RESOLUTION_TIMEOUT_SECONDS,
) -> AgentPromptBundle:
    """Resolve every agent prompt concurrently within one deadline; never raises for a prompt problem.

    Each prompt fails open on its own. Prompts still unresolved when
    ``timeout_seconds`` elapses use their local fallback, while prompts that
    already resolved keep their result, so the wait is one timeout window
    rather than one per prompt.
    """

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    tasks = {
        key: asyncio.create_task(_resolve_one(resolver, definition, label))
        for key, definition in AGENT_PROMPT_DEFINITIONS.items()
    }
    try:
        _, pending = await asyncio.wait(tasks.values(), timeout=timeout_seconds)
    finally:
        # Also runs if the caller is cancelled, so no resolution outlives the request.
        for task in tasks.values():
            if not task.done():
                task.cancel()
    if pending:
        logger.warning("%d agent prompts unresolved at the deadline; using local fallbacks", len(pending))
    resolved = {
        key: task.result()
        if task.done() and not task.cancelled() and task.exception() is None
        else local_prompt(AGENT_PROMPT_DEFINITIONS[key], label=label)
        for key, task in tasks.items()
    }
    return AgentPromptBundle(**resolved)
