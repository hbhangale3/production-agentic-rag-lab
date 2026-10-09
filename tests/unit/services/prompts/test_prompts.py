import dataclasses
import hashlib

import pytest
from src.services.prompts import (
    DEFAULT_PROMPT_LABEL,
    LOCAL_PROMPT_VERSION,
    LocalPromptResolver,
    PromptDefinition,
    PromptIdentity,
    ResolvedPrompt,
    canonicalize_prompt,
    fingerprint_prompt,
    local_prompt,
)

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_fingerprint_is_sha256_of_canonical_content_with_known_vectors() -> None:
    assert fingerprint_prompt("abc") == ABC_SHA256
    assert fingerprint_prompt("You are a grader.\nReturn JSON.") == hashlib.sha256(
        b"You are a grader.\nReturn JSON."
    ).hexdigest()
    assert fingerprint_prompt("Ünïcode prompt") == hashlib.sha256("Ünïcode prompt".encode()).hexdigest()


@pytest.mark.parametrize("equivalent", ["abc", "  abc\n", "\r\nabc\r\n", "\tabc "])
def test_same_template_has_the_same_fingerprint_regardless_of_outer_whitespace(equivalent: str) -> None:
    assert fingerprint_prompt(equivalent) == ABC_SHA256


def test_line_endings_are_canonicalized_but_inner_text_is_not() -> None:
    assert canonicalize_prompt("a\r\nb\rc\n") == "a\nb\nc"
    assert fingerprint_prompt("a\r\nb") == fingerprint_prompt("a\nb")
    assert fingerprint_prompt("a b") != fingerprint_prompt("a  b")
    assert fingerprint_prompt("Score 0-100") != fingerprint_prompt("score 0-100")


@pytest.mark.parametrize("changed", ["abd", "abc.", "abc\nmore", "ABC"])
def test_content_change_changes_fingerprint(changed: str) -> None:
    assert fingerprint_prompt(changed) != ABC_SHA256


@pytest.mark.parametrize("content", ["", "   ", None, 123])
def test_blank_or_non_text_content_cannot_be_fingerprinted(content) -> None:
    with pytest.raises(ValueError):
        fingerprint_prompt(content)


def test_local_prompt_resolution_exposes_identity_without_template_text() -> None:
    definition = PromptDefinition(name="agent-test", fallback_content="Template text {\"score\"", required_markers=('{"score"',))

    prompt = local_prompt(definition)

    assert prompt == ResolvedPrompt(
        name="agent-test",
        content='Template text {"score"',
        version=LOCAL_PROMPT_VERSION,
        label=DEFAULT_PROMPT_LABEL,
        source="local",
    )
    assert (LOCAL_PROMPT_VERSION, DEFAULT_PROMPT_LABEL) == ("local-v1", "production")
    assert prompt.identity == PromptIdentity(
        name="agent-test",
        version="local-v1",
        label="production",
        fingerprint=fingerprint_prompt('Template text {"score"'),
    )
    assert [field.name for field in dataclasses.fields(PromptIdentity)] == ["name", "version", "label", "fingerprint"]
    assert prompt.identity.as_metadata() == {
        "prompt_name": "agent-test",
        "prompt_version": "local-v1",
        "prompt_label": "production",
        "prompt_fingerprint": prompt.fingerprint,
    }
    assert "Template text" not in repr(prompt.identity)
    assert local_prompt(definition, label="staging").label == "staging"


@pytest.mark.anyio
async def test_local_resolver_always_returns_the_fallback() -> None:
    definition = PromptDefinition(name="agent-test", fallback_content="Template", fallback_version="local-v7")

    prompt = await LocalPromptResolver().resolve(definition, label="production")

    assert (prompt.content, prompt.version, prompt.source) == ("Template", "local-v7", "local")
    assert await LocalPromptResolver().resolve(definition) == prompt


def test_identity_and_prompt_types_are_immutable_and_validated() -> None:
    identity = PromptIdentity(name="n", version="v", label="l", fingerprint="f")

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.version = "other"  # type: ignore[misc]
    for invalid in ({"name": " "}, {"version": ""}, {"label": ""}, {"fingerprint": " "}):
        with pytest.raises(ValueError):
            dataclasses.replace(identity, **invalid)
    for invalid in ({"content": " "}, {"name": ""}, {"version": ""}, {"label": ""}, {"source": "remote"}):
        with pytest.raises(ValueError):
            dataclasses.replace(ResolvedPrompt(name="n", content="c", version="v", label="l"), **invalid)


def test_definition_requires_its_own_markers_in_the_fallback() -> None:
    with pytest.raises(ValueError, match="required markers"):
        PromptDefinition(name="n", fallback_content="no contract here", required_markers=('{"score"',))
    with pytest.raises(ValueError):
        PromptDefinition(name="n", fallback_content="  ")
