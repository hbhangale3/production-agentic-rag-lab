from dataclasses import replace

import pytest
from src.exceptions import GroundingValidationError
from src.services.evidence import EvidenceSource
from src.services.rag.validation import GroundedAnswerValidator


def source(label: str) -> EvidenceSource:
    return EvidenceSource(
        label=label,
        retrieval_rank=1,
        arxiv_id="2401.00001",
        chunk_id="paper::chunk::000",
        chunk_index=0,
        paper_title="Paper",
        section_title="Results",
        content="Evidence",
        authors=(),
        categories=(),
        published_date=None,
        rrf_score=0.03,
        bm25_rank=1,
        vector_rank=1,
        bm25_score=None,
        vector_score=None,
        truncated=False,
    )


SOURCES = (source("[S1]"), replace(source("[S1]"), label="[S2]"))


@pytest.mark.parametrize(
    ("answer", "canonical", "citations"),
    [
        ("A supported result [S1].", "A supported result [S1].", ("[S1]",)),
        ("One [S1], two [S2].", "One [S1], two [S2].", ("[S1]", "[S2]")),
        ("Repeat [S1], [S1], [S2].", "Repeat [S1], [S1], [S2].", ("[S1]", "[S2]")),
        ("Reverse [S2], then [S1].", "Reverse [S2], then [S1].", ("[S2]", "[S1]")),
        ("Full width 【S1】.", "Full width [S1].", ("[S1]",)),
        ("Both 【S2】 and 【S1】.", "Both [S2] and [S1].", ("[S2]", "[S1]")),
    ],
)
def test_valid_grounded_answers(answer, canonical, citations) -> None:
    result = GroundedAnswerValidator().validate(answer, SOURCES)

    assert result.answer == canonical
    assert result.cited_labels == citations
    assert result.is_insufficiency is False


@pytest.mark.parametrize(
    "answer",
    [
        "The available evidence is insufficient to answer the question.",
        "The supplied sources do not provide enough information to determine this.",
        "Cannot answer this question from the provided evidence.",
        "There is not enough evidence in the retrieved sources to conclude this.",
    ],
)
def test_explicit_insufficiency_may_be_citation_free(answer) -> None:
    result = GroundedAnswerValidator().validate(answer, SOURCES)

    assert result.cited_labels == ()
    assert result.is_insufficiency is True


@pytest.mark.parametrize(
    "answer",
    [
        "",
        " \n\t ",
        "Unknown [S99].",
        "Invalid [S0].",
        "Invalid 【S0】.",
        "Invalid [S].",
        "Invalid [Sabc].",
        "Invalid [S-1].",
        "Invalid [S 1].",
        "Invalid 【Sabc】.",
        "AI improves healthcare access.",
        "The evidence is insufficient for certainty, but AI definitely reduces mortality by 40%.",
        "The evidence is insufficient. AI definitely improves outcomes.",
    ],
)
def test_invalid_grounded_answers_fail_closed(answer) -> None:
    with pytest.raises(GroundingValidationError):
        GroundedAnswerValidator().validate(answer, SOURCES)


def test_s0_is_invalid_even_if_a_malformed_source_list_contains_it() -> None:
    with pytest.raises(GroundingValidationError, match="malformed"):
        GroundedAnswerValidator().validate("Invalid [S0].", (source("[S0]"),))


def test_normal_bracketed_words_and_acronyms_are_not_malformed_citations() -> None:
    result = GroundedAnswerValidator().validate("A summary [Summary] from SQL [SQL] [S1].", SOURCES)

    assert result.cited_labels == ("[S1]",)
