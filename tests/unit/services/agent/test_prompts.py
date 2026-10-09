import asyncio
import dataclasses
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from src.config import Settings
from src.schemas.hybrid_search import HybridSearchHit, HybridSearchResult
from src.services.agent import (
    AGENT_PROMPT_DEFINITIONS,
    AgentCacheIdentityFactory,
    AgentCacheKeyBuilder,
    AgentGraphDependencies,
    AgentPromptBundle,
    AgentPromptIdentityBundle,
    AnswerGroundingPromptBuilder,
    EvidenceGraderPromptBuilder,
    GuardrailPromptBuilder,
    LiveArxivPaper,
    LivePaperSelectionPromptBuilder,
    QueryRewritePromptBuilder,
    build_agent_graph,
    create_initial_agent_state,
    local_agent_prompt_bundle,
    resolve_agent_prompt_bundle,
)
from src.services.agent.prompts import DEFAULT_BUNDLE_RESOLUTION_TIMEOUT_SECONDS
from src.services.evidence import EvidenceContextBuilder
from src.services.llm.base import LLMCompletion
from src.services.prompts import (
    LocalPromptResolver,
    PromptDefinition,
    ResolvedPrompt,
    fingerprint_prompt,
    local_prompt,
)
from src.services.rag import RAGGenerationService
from src.services.rag.prompt import RAG_REGENERATION_INSTRUCTION, RAG_SYSTEM_PROMPT, RAGPromptBuilder

EXPECTED_NAMES = {
    "guardrail": "agent-guardrail",
    "evidence_grader": "agent-evidence-grader",
    "query_rewrite": "agent-query-rewrite",
    "live_selector": "agent-live-paper-selector",
    "generation": "rag-answer-generation",
    "regeneration": "rag-answer-regeneration",
    "answer_grounding": "agent-answer-grounding",
}
LOCAL_CONTENT = {
    "guardrail": GuardrailPromptBuilder.SYSTEM_PROMPT,
    "evidence_grader": EvidenceGraderPromptBuilder.SYSTEM_PROMPT,
    "query_rewrite": QueryRewritePromptBuilder.SYSTEM_PROMPT,
    "live_selector": LivePaperSelectionPromptBuilder.SYSTEM_PROMPT,
    "generation": RAG_SYSTEM_PROMPT,
    "regeneration": RAG_REGENERATION_INSTRUCTION,
    "answer_grounding": AnswerGroundingPromptBuilder.SYSTEM_PROMPT,
}
PRIVATE_QUESTION = "PRIVATE_QUESTION_SENTINEL about India?"
PRIVATE_EVIDENCE = "PRIVATE_EVIDENCE_SENTINEL passage"
PRIVATE_ANSWER = "PRIVATE_ANSWER_SENTINEL [S1]."


def make_context(text: str = PRIVATE_EVIDENCE):
    hit = HybridSearchHit(
        chunk_id="c1", arxiv_id="2401.00001", chunk_index=0, chunk_text=text, word_count=2, paper_title="T", rrf_score=0.5
    )
    result = HybridSearchResult(query="q", retrieval_mode="hybrid", count=1, results=[hit])
    return EvidenceContextBuilder(max_context_tokens=1000).build(result)


def test_catalog_names_every_llm_stage_with_explicit_local_versions() -> None:
    assert {key: definition.name for key, definition in AGENT_PROMPT_DEFINITIONS.items()} == EXPECTED_NAMES
    assert {definition.fallback_version for definition in AGENT_PROMPT_DEFINITIONS.values()} == {"local-v1"}
    assert {key: definition.fallback_content for key, definition in AGENT_PROMPT_DEFINITIONS.items()} == LOCAL_CONTENT
    assert AGENT_PROMPT_DEFINITIONS["guardrail"].required_markers == ('{"score"',)
    assert AGENT_PROMPT_DEFINITIONS["query_rewrite"].required_markers == ('{"query"',)
    assert AGENT_PROMPT_DEFINITIONS["live_selector"].required_markers == ('"selected_arxiv_ids"',)
    assert [field.name for field in dataclasses.fields(AgentPromptBundle)] == list(EXPECTED_NAMES)
    assert [field.name for field in dataclasses.fields(AgentPromptIdentityBundle)] == list(EXPECTED_NAMES)


def test_local_bundle_identities_fingerprint_the_templates_only() -> None:
    bundle = local_agent_prompt_bundle()
    identities = bundle.identities

    for key, name in EXPECTED_NAMES.items():
        identity = getattr(identities, key)
        assert (identity.name, identity.version, identity.label) == (name, "local-v1", "production")
        assert identity.fingerprint == fingerprint_prompt(LOCAL_CONTENT[key])
        assert getattr(bundle, key).source == "local"
    assert len({getattr(identities, key).fingerprint for key in EXPECTED_NAMES}) == 7
    assert local_agent_prompt_bundle().identities == identities
    assert "You are" not in repr(identities)
    assert identities.as_cache_payload()["guardrail"] == {
        "name": "agent-guardrail",
        "version": "local-v1",
        "label": "production",
        "fingerprint": fingerprint_prompt(GuardrailPromptBuilder.SYSTEM_PROMPT),
    }
    assert local_agent_prompt_bundle(label="staging").identities.guardrail.label == "staging"


def test_default_builders_use_the_same_identities_as_the_local_bundle() -> None:
    identities = local_agent_prompt_bundle().identities

    assert GuardrailPromptBuilder().identity == identities.guardrail
    assert EvidenceGraderPromptBuilder().identity == identities.evidence_grader
    assert QueryRewritePromptBuilder().identity == identities.query_rewrite
    assert LivePaperSelectionPromptBuilder().identity == identities.live_selector
    assert AnswerGroundingPromptBuilder().identity == identities.answer_grounding
    assert RAGPromptBuilder().identity == identities.generation
    assert RAGPromptBuilder().regeneration_identity == identities.regeneration


def test_runtime_question_evidence_and_previous_answer_never_change_prompt_identity() -> None:
    paper = LiveArxivPaper(
        arxiv_id="2501.00001",
        title=PRIVATE_EVIDENCE,
        abstract=PRIVATE_EVIDENCE,
        authors=(),
        categories=(),
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        updated_date=None,
        pdf_url="https://arxiv.org/pdf/2501.00001",
    )
    guardrail, grader, rewriter = GuardrailPromptBuilder(), EvidenceGraderPromptBuilder(), QueryRewritePromptBuilder()
    selector, grounding, rag = LivePaperSelectionPromptBuilder(), AnswerGroundingPromptBuilder(), RAGPromptBuilder()
    before = (
        guardrail.identity,
        grader.identity,
        rewriter.identity,
        selector.identity,
        grounding.identity,
        rag.identity,
        rag.regeneration_identity,
    )
    context = make_context()

    rendered = [
        guardrail.build(PRIVATE_QUESTION),
        grader.build(PRIVATE_QUESTION, context),
        rewriter.build(PRIVATE_QUESTION, "another query"),
        selector.build(PRIVATE_QUESTION, (paper,), max_papers=1),
        grounding.build(PRIVATE_QUESTION, PRIVATE_ANSWER, context.sources),
        rag.build(question=PRIVATE_QUESTION, evidence=context),
        rag.build_regeneration(question=PRIVATE_QUESTION, evidence=context, previous_answer=PRIVATE_ANSWER),
        rag.build_regeneration(question="Other question?", evidence=make_context("other"), previous_answer="Other [S1]."),
    ]

    after = (
        guardrail.identity,
        grader.identity,
        rewriter.identity,
        selector.identity,
        grounding.identity,
        rag.identity,
        rag.regeneration_identity,
    )
    assert after == before
    assert all(PRIVATE_QUESTION in messages[1].content for messages in rendered[:7])
    for identity in after:
        assert "PRIVATE" not in repr(identity)
    assert rag.regeneration_identity.fingerprint == fingerprint_prompt(RAG_REGENERATION_INSTRUCTION)
    assert PRIVATE_ANSWER in rendered[6][1].content


def test_builders_render_the_resolved_template_and_report_its_identity() -> None:
    managed = ResolvedPrompt(
        name="agent-guardrail",
        content='Managed guardrail. Return {"score": <0-100>}.',
        version="langfuse-v4",
        label="production",
        source="langfuse",
    )

    builder = GuardrailPromptBuilder(prompt=managed)

    assert builder.build("question")[0].content == managed.content
    assert builder.identity == managed.identity
    assert builder.identity.version == "langfuse-v4"
    assert builder.identity != GuardrailPromptBuilder().identity

    generation = ResolvedPrompt(name="rag-answer-generation", content="Managed RAG system.", version="langfuse-v2", label="production", source="langfuse")
    regeneration = ResolvedPrompt(name="rag-answer-regeneration", content="Managed regeneration instruction.", version="langfuse-v9", label="production", source="langfuse")
    rag = RAGPromptBuilder(generation_prompt=generation, regeneration_prompt=regeneration)
    system, user = rag.build_regeneration(question="Q?", evidence=make_context(), previous_answer="Prev [S1].")
    assert system.content == "Managed RAG system."
    assert user.content.endswith("Managed regeneration instruction.")
    assert RAG_REGENERATION_INSTRUCTION not in user.content
    assert (rag.identity.version, rag.regeneration_identity.version) == ("langfuse-v2", "langfuse-v9")


class ScriptedResolver:
    def __init__(self, overrides: dict[str, object]) -> None:
        self.overrides = overrides
        self.calls: list[tuple[str, str]] = []

    async def resolve(self, definition: PromptDefinition, *, label: str = "production") -> ResolvedPrompt:
        self.calls.append((definition.name, label))
        outcome = self.overrides.get(definition.name)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return local_prompt(definition, label=label)
        return outcome


@pytest.mark.anyio
async def test_bundle_resolution_resolves_each_prompt_once_and_mixes_sources() -> None:
    managed = ResolvedPrompt(
        name="agent-answer-grounding",
        content='Managed grounding. Return {"score": <0-100>}.',
        version="langfuse-v3",
        label="staging",
        source="langfuse",
    )
    resolver = ScriptedResolver({"agent-answer-grounding": managed})

    bundle = await resolve_agent_prompt_bundle(resolver, label="staging")

    assert resolver.calls == [(name, "staging") for name in EXPECTED_NAMES.values()]
    assert bundle.answer_grounding == managed
    assert bundle.identities.answer_grounding.version == "langfuse-v3"
    assert bundle.guardrail == local_prompt(AGENT_PROMPT_DEFINITIONS["guardrail"], label="staging")
    assert bundle.identities != local_agent_prompt_bundle(label="staging").identities
    assert await resolve_agent_prompt_bundle(LocalPromptResolver()) == local_agent_prompt_bundle()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "outcome",
    [
        RuntimeError("private resolver detail"),
        ResolvedPrompt(name="agent-query-rewrite", content="wrong prompt", version="v", label="production"),
        ResolvedPrompt(name="agent-guardrail", content="lost its output contract", version="v", label="production"),
        "not a resolved prompt",
    ],
)
async def test_bundle_resolution_is_fail_open_per_prompt(outcome) -> None:
    resolver = ScriptedResolver({"agent-guardrail": outcome})

    bundle = await resolve_agent_prompt_bundle(resolver)

    assert bundle == local_agent_prompt_bundle()


def test_bundle_rejects_a_prompt_in_the_wrong_slot() -> None:
    local = local_agent_prompt_bundle()

    with pytest.raises(ValueError, match="wrong prompt"):
        dataclasses.replace(local, guardrail=local.answer_grounding)
    with pytest.raises(ValueError):
        dataclasses.replace(local, generation="text")  # type: ignore[arg-type]


def managed_prompt(definition: PromptDefinition, label: str = "production") -> ResolvedPrompt:
    return ResolvedPrompt(
        name=definition.name,
        content=f"MANAGED[{definition.name}] " + " ".join(definition.required_markers),
        version="langfuse-v5",
        label=label,
        source="langfuse",
    )


class GatedResolver:
    """Managed prompts that resolve only after ``release`` is set; chosen names fail or never finish."""

    def __init__(self, *, failing: set[str] = frozenset(), hanging: set[str] = frozenset()) -> None:
        self.failing, self.hanging = failing, hanging
        self.release = asyncio.Event()
        self.started: list[str] = []
        self.all_started = asyncio.Event()
        self.cancelled: list[str] = []

    async def resolve(self, definition: PromptDefinition, *, label: str = "production") -> ResolvedPrompt:
        self.started.append(definition.name)
        if len(self.started) == len(EXPECTED_NAMES):
            self.all_started.set()
        try:
            if definition.name in self.hanging:
                await asyncio.Event().wait()
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(definition.name)
            raise
        if definition.name in self.failing:
            raise RuntimeError("private resolver detail")
        return managed_prompt(definition, label)


def sources(bundle: AgentPromptBundle) -> dict[str, str]:
    return {key: getattr(bundle, key).source for key in EXPECTED_NAMES}


def assert_identities_match_contents(bundle: AgentPromptBundle) -> None:
    for key in EXPECTED_NAMES:
        prompt, identity = getattr(bundle, key), getattr(bundle.identities, key)
        assert identity.fingerprint == fingerprint_prompt(prompt.content)
        assert (identity.name, identity.version, identity.label) == (prompt.name, prompt.version, prompt.label)


@pytest.mark.anyio
async def test_all_seven_prompts_are_resolved_concurrently() -> None:
    resolver = GatedResolver()

    resolution = asyncio.create_task(resolve_agent_prompt_bundle(resolver, timeout_seconds=30))
    await asyncio.wait_for(resolver.all_started.wait(), timeout=5)

    assert sorted(resolver.started) == sorted(EXPECTED_NAMES.values())
    assert not resolution.done()
    resolver.release.set()
    bundle = await resolution

    assert set(sources(bundle).values()) == {"langfuse"}
    assert_identities_match_contents(bundle)
    assert resolver.cancelled == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failing",
    [{"agent-guardrail"}, {"agent-query-rewrite", "rag-answer-generation", "agent-answer-grounding"}],
)
async def test_failed_prompts_fall_back_individually_and_the_rest_stay_managed(failing: set[str]) -> None:
    resolver = GatedResolver(failing=failing)
    resolver.release.set()

    bundle = await resolve_agent_prompt_bundle(resolver, label="staging", timeout_seconds=30)

    for key, name in EXPECTED_NAMES.items():
        prompt = getattr(bundle, key)
        if name in failing:
            assert prompt == local_prompt(AGENT_PROMPT_DEFINITIONS[key], label="staging")
        else:
            assert prompt == managed_prompt(AGENT_PROMPT_DEFINITIONS[key], "staging")
    assert_identities_match_contents(bundle)


@pytest.mark.anyio
async def test_one_prompt_that_never_resolves_is_bounded_and_only_it_falls_back() -> None:
    resolver = GatedResolver(hanging={"agent-live-paper-selector"})
    resolver.release.set()

    bundle = await asyncio.wait_for(resolve_agent_prompt_bundle(resolver, timeout_seconds=0.05), timeout=5)

    assert sources(bundle) == {
        **{key: "langfuse" for key in EXPECTED_NAMES},
        "live_selector": "local",
    }
    assert bundle.live_selector == local_prompt(AGENT_PROMPT_DEFINITIONS["live_selector"])
    assert_identities_match_contents(bundle)
    await asyncio.sleep(0)
    assert resolver.cancelled == ["agent-live-paper-selector"]


@pytest.mark.anyio
async def test_bundle_deadline_returns_the_local_bundle_without_raising(caplog) -> None:
    resolver = GatedResolver()

    bundle = await asyncio.wait_for(resolve_agent_prompt_bundle(resolver, timeout_seconds=0.05), timeout=5)

    assert bundle == local_agent_prompt_bundle()
    assert sorted(resolver.started) == sorted(EXPECTED_NAMES.values())
    assert_identities_match_contents(bundle)
    await asyncio.sleep(0)
    assert sorted(resolver.cancelled) == sorted(EXPECTED_NAMES.values())
    assert "7 agent prompts unresolved" in caplog.text
    assert "MANAGED" not in caplog.text and "You are" not in caplog.text


@pytest.mark.anyio
async def test_seven_slow_resolutions_cost_one_timeout_window_not_seven() -> None:
    class SlowResolver:
        async def resolve(self, definition, *, label="production"):
            await asyncio.sleep(0.2)
            return managed_prompt(definition, label)

    loop = asyncio.get_running_loop()
    started = loop.time()

    bundle = await resolve_agent_prompt_bundle(SlowResolver(), timeout_seconds=0.6)

    assert set(sources(bundle).values()) == {"langfuse"}
    assert loop.time() - started < 7 * 0.2


@pytest.mark.anyio
async def test_cancelling_the_caller_cancels_outstanding_resolutions() -> None:
    resolver = GatedResolver()
    resolution = asyncio.create_task(resolve_agent_prompt_bundle(resolver, timeout_seconds=30))
    await asyncio.wait_for(resolver.all_started.wait(), timeout=5)

    resolution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resolution
    await asyncio.sleep(0)

    assert sorted(resolver.cancelled) == sorted(EXPECTED_NAMES.values())


@pytest.mark.anyio
@pytest.mark.parametrize("timeout", [0, -1, True, "5"])
async def test_invalid_bundle_timeout_is_a_programming_error(timeout) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        await resolve_agent_prompt_bundle(LocalPromptResolver(), timeout_seconds=timeout)
    assert DEFAULT_BUNDLE_RESOLUTION_TIMEOUT_SECONDS == 5.0


@pytest.mark.anyio
async def test_cache_key_and_graph_use_the_same_partially_managed_bundle() -> None:
    resolver = GatedResolver(failing={"agent-evidence-grader"}, hanging={"rag-answer-regeneration"})
    resolver.release.set()
    bundle = await resolve_agent_prompt_bundle(resolver, timeout_seconds=0.05)
    assert sources(bundle)["guardrail"] == "langfuse"
    assert (sources(bundle)["evidence_grader"], sources(bundle)["regeneration"]) == ("local", "local")

    factory = AgentCacheIdentityFactory(settings=Settings(_env_file=None, debug=False))
    identity = factory.create(question="AI question", corpus_fingerprint="a" * 64, prompts=bundle.identities)
    local_identity = factory.create(
        question="AI question", corpus_fingerprint="a" * 64, prompts=local_agent_prompt_bundle().identities
    )
    assert AgentCacheKeyBuilder.build(identity) != AgentCacheKeyBuilder.build(local_identity)

    system_prompts: list[str] = []

    class RejectingLLM:
        async def complete(self, messages, **kwargs):
            system_prompts.append(messages[0].content)
            return LLMCompletion(content='{"score":5}', model="fake-model")

    llm = RejectingLLM()
    builder = EvidenceContextBuilder(max_context_tokens=1000)
    live_search, processor = Mock(), Mock()
    live_search.search, processor.process = AsyncMock(), AsyncMock()
    deps = AgentGraphDependencies(
        llm_provider=llm,
        hybrid_search_service=Mock(),
        evidence_context_builder=builder,
        embedding_provider=Mock(),
        rag_generation_service=RAGGenerationService(llm_provider=llm, evidence_builder=builder),
        live_search_service=live_search,
        live_document_processor=processor,
    )

    result = await build_agent_graph(dependencies=deps, prompts=bundle).ainvoke(
        create_initial_agent_state("AI question")
    )

    assert system_prompts == [bundle.guardrail.content]
    assert result["prompt_identities"] == bundle.identities
    assert identity.prompts == {
        key: AgentCacheIdentityFactory(settings=Settings(_env_file=None, debug=False))
        .create(question="q", corpus_fingerprint="a" * 64, prompts=result["prompt_identities"])
        .prompts[key]
        for key in EXPECTED_NAMES
    }
