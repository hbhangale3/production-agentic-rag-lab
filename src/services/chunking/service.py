import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.services.chunking.models import ChunkingResult, PaperChunk

PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class _Section:
    title: str | None
    text: str
    word_count: int


@dataclass(frozen=True)
class _SectionGroup:
    titles: tuple[str, ...]
    texts: tuple[str, ...]
    word_count: int

    @property
    def title(self) -> str | None:
        return " / ".join(dict.fromkeys(self.titles)) or None

    @property
    def text(self) -> str:
        return "\n\n".join(self.texts)


class PaperChunkingService:
    """Create deterministic chunks without touching PDFs or external services."""

    def __init__(self, *, target_words: int, overlap_words: int, min_words: int) -> None:
        if target_words < 1:
            raise ValueError("target_words must be greater than zero")
        if overlap_words < 0:
            raise ValueError("overlap_words must not be negative")
        if overlap_words >= target_words:
            raise ValueError("overlap_words must be smaller than target_words")
        if min_words < 1:
            raise ValueError("min_words must be greater than zero")
        if min_words > target_words:
            raise ValueError("min_words must not exceed target_words")
        self.target_words = target_words
        self.overlap_words = overlap_words
        self.min_words = min_words

    @classmethod
    def from_settings(cls, settings: Any) -> "PaperChunkingService":
        return cls(
            target_words=settings.chunk_target_words,
            overlap_words=settings.chunk_overlap_words,
            min_words=settings.chunk_min_words,
        )

    def chunk_paper(self, paper: Any) -> ChunkingResult:
        """Chunk a Paper-like object while leaving it untouched."""
        arxiv_id = str(getattr(paper, "arxiv_id", "")).strip()
        if not arxiv_id:
            raise ValueError("paper arxiv_id must not be empty")

        sections = self._usable_sections(getattr(paper, "sections", None))
        if sections:
            pieces = [
                piece
                for group in self._group_sections(sections)
                for piece in self._split_text(group.text, group.title)
            ]
            mode = "sections"
        else:
            raw_text = self._clean_text(getattr(paper, "raw_text", None))
            pieces = self._split_text(raw_text, None) if raw_text else []
            mode = "raw_text" if pieces else "empty"

        metadata = self._metadata(paper)
        chunks = tuple(
            PaperChunk(
                chunk_id=f"{arxiv_id}::chunk::{index:03d}",
                arxiv_id=arxiv_id,
                chunk_index=index,
                section_title=section_title,
                text=text,
                word_count=len(text.split()),
                has_overlap=has_overlap,
                **metadata,
            )
            for index, (section_title, text, has_overlap) in enumerate(pieces)
        )
        return ChunkingResult(arxiv_id=arxiv_id, mode=mode, chunks=chunks)

    def _split_text(self, text: str, section_title: str | None) -> list[tuple[str | None, str, bool]]:
        paragraphs = [self._clean_text(value) for value in PARAGRAPH_BREAK.split(text)]
        paragraphs = [value for value in paragraphs if value]
        words: list[str] = []
        paragraph_ends: list[int] = []
        for paragraph in paragraphs:
            words.extend(paragraph.split())
            paragraph_ends.append(len(words))

        pieces: list[tuple[str | None, str, bool]] = []
        start = 0
        while start < len(words):
            remaining = len(words) - start
            if remaining <= self.target_words:
                end = len(words)
            else:
                limit = start + self.target_words
                eligible_ends = [
                    boundary
                    for boundary in paragraph_ends
                    if start + self.min_words <= boundary <= limit
                ]
                end = max(eligible_ends, default=limit)

            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                pieces.append((section_title, chunk_text, start > 0 and self.overlap_words > 0))
            if end >= len(words):
                break
            next_start = max(0, end - self.overlap_words)
            if next_start <= start:
                next_start = end
            start = next_start
        return pieces

    def _group_sections(self, sections: list[_Section]) -> list[_SectionGroup]:
        groups: list[_SectionGroup] = []
        buffered: list[_Section] = []
        buffered_words = 0

        def flush() -> None:
            nonlocal buffered, buffered_words
            if not buffered:
                return
            groups.append(
                _SectionGroup(
                    titles=tuple(section.title for section in buffered if section.title),
                    texts=tuple(section.text for section in buffered),
                    word_count=buffered_words,
                )
            )
            buffered = []
            buffered_words = 0

        for section in sections:
            if section.word_count >= self.min_words:
                flush()
                buffered = [section]
                buffered_words = section.word_count
                flush()
                continue
            if buffered and buffered_words + section.word_count > self.target_words:
                flush()
            buffered.append(section)
            buffered_words += section.word_count
            if buffered_words >= self.min_words:
                flush()
        flush()

        if len(groups) > 1 and groups[-1].word_count < self.min_words:
            previous, trailing = groups[-2:]
            if previous.word_count + trailing.word_count <= self.target_words:
                groups[-2:] = [
                    _SectionGroup(
                        titles=previous.titles + trailing.titles,
                        texts=previous.texts + trailing.texts,
                        word_count=previous.word_count + trailing.word_count,
                    )
                ]
        return groups

    @classmethod
    def _usable_sections(cls, value: Any) -> list[_Section]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        sections: list[_Section] = []
        for entry in value:
            if isinstance(entry, Mapping):
                title = entry.get("title")
                text = entry.get("text")
            else:
                title = getattr(entry, "title", None)
                text = getattr(entry, "text", None)
            cleaned = cls._clean_text(text)
            if not cleaned:
                continue
            cleaned_title = cls._clean_text(title) or None
            sections.append(_Section(cleaned_title, cleaned, len(cleaned.split())))
        return sections

    @staticmethod
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        paragraphs = [" ".join(part.split()) for part in PARAGRAPH_BREAK.split(value)]
        return "\n\n".join(part for part in paragraphs if part).strip()

    @staticmethod
    def _metadata(paper: Any) -> dict[str, Any]:
        published_date = getattr(paper, "published_date", None)
        return {
            "paper_title": PaperChunkingService._clean_text(getattr(paper, "title", None)) or None,
            "authors": tuple(getattr(paper, "authors", None) or ()),
            "categories": tuple(getattr(paper, "categories", None) or ()),
            "published_date": published_date if isinstance(published_date, datetime) else None,
        }
