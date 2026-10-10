"""Content-free identity of the prompts one agent run uses; safe to keep in graph state."""

from dataclasses import dataclass, fields

from src.services.prompts.base import PromptIdentity


@dataclass(frozen=True)
class AgentPromptIdentityBundle:
    """Content-free identity of every prompt one agent run uses."""

    guardrail: PromptIdentity
    evidence_grader: PromptIdentity
    query_rewrite: PromptIdentity
    live_selector: PromptIdentity
    generation: PromptIdentity
    regeneration: PromptIdentity
    answer_grounding: PromptIdentity

    def as_cache_payload(self) -> dict[str, dict[str, str]]:
        return {
            field.name: {
                "name": identity.name,
                "version": identity.version,
                "label": identity.label,
                "fingerprint": identity.fingerprint,
            }
            for field in fields(self)
            for identity in (getattr(self, field.name),)
        }
