import re
from dataclasses import dataclass

from src.exceptions import GroundingValidationError
from src.services.evidence import EvidenceSource

VALID_CITATION = re.compile(r"\[S[1-9]\d*\]")
FULLWIDTH_CITATION = re.compile(r"【S(\d+)】")
BRACKETED_S_TEXT = re.compile(r"\[S([^\]\r\n]*)\]|【S([^】\r\n]*)】")

_INSUFFICIENCY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:the )?(?:(?:available|retrieved|supplied|provided) )?(?:evidence|sources?) "
        r"(?:is|are) (?:insufficient|not sufficient)(?: to .+)?[.!]?",
        r"(?:the )?(?:(?:available|retrieved|supplied|provided) )?(?:evidence|sources?) "
        r"(?:does|do) not provide (?:enough|sufficient) (?:evidence|information)(?: to .+)?[.!]?",
        r"there (?:is|are) not enough (?:evidence|information)(?: in (?:the )?"
        r"(?:retrieved|supplied|provided) (?:evidence|sources?))?(?: to .+)?[.!]?",
        r"(?:i )?cannot (?:answer|determine|conclude|establish)(?: .+)? from (?:the )?"
        r"(?:available|retrieved|supplied|provided) (?:evidence|sources?)[.!]?",
    )
)


@dataclass(frozen=True)
class ValidatedGroundedAnswer:
    answer: str
    cited_labels: tuple[str, ...]
    is_insufficiency: bool


class GroundedAnswerValidator:
    """Deterministically enforce the structural grounded-answer contract."""

    def canonicalize(self, answer: str) -> str:
        if not isinstance(answer, str):
            raise GroundingValidationError("Generated answer was not text")
        return FULLWIDTH_CITATION.sub(r"[S\1]", answer)

    def validate(self, answer: str, sources: tuple[EvidenceSource, ...]) -> ValidatedGroundedAnswer:
        canonical = self.canonicalize(answer)
        if not canonical.strip():
            raise GroundingValidationError("Generated answer was empty")
        if self._has_malformed_citation(canonical):
            raise GroundingValidationError("Generated answer contained malformed citation syntax")

        cited_labels = tuple(dict.fromkeys(VALID_CITATION.findall(canonical)))
        available_labels = {source.label for source in sources}
        if any(label not in available_labels for label in cited_labels):
            raise GroundingValidationError("Generated answer contained an unsupported citation label")

        is_insufficiency = self._is_insufficiency(canonical)
        if not cited_labels and not is_insufficiency:
            raise GroundingValidationError("Substantive generated answer did not cite supplied evidence")
        return ValidatedGroundedAnswer(
            answer=canonical,
            cited_labels=cited_labels,
            is_insufficiency=is_insufficiency,
        )

    @staticmethod
    def _has_malformed_citation(answer: str) -> bool:
        for match in BRACKETED_S_TEXT.finditer(answer):
            entire = match.group(0)
            if VALID_CITATION.fullmatch(entire):
                continue
            suffix = match.group(1) if match.group(1) is not None else match.group(2)
            # Keep detection narrow enough not to reject ordinary bracketed words
            # such as [Summary] or common uppercase acronyms such as [SQL].
            citation_like = (
                suffix == ""
                or any(character.isdigit() for character in suffix)
                or suffix[:1] in {" ", "+", "-"}
                or (suffix.islower() and suffix.isalpha() and len(suffix) <= 3)
            )
            if citation_like:
                return True
        return False

    @staticmethod
    def _is_insufficiency(answer: str) -> bool:
        normalized = " ".join(answer.split())
        return any(pattern.fullmatch(normalized) for pattern in _INSUFFICIENCY_PATTERNS)
