from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from src.schemas.parsed_pdf import ParsedPDFSection
from src.services.chunking import PaperChunkingService


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def paper(**overrides):
    values = {
        "arxiv_id": "2610.01963",
        "title": "Clinical AI",
        "authors": ["Ada Lovelace"],
        "categories": ["cs.AI", "cs.CL"],
        "published_date": datetime(2026, 10, 1, tzinfo=timezone.utc),
        "raw_text": words("raw", 250),
        "sections": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def service(*, target: int = 100, overlap: int = 20, minimum: int = 30):
    return PaperChunkingService(target_words=target, overlap_words=overlap, min_words=minimum)


def test_chunk_ids_are_deterministic_and_unique() -> None:
    first = service().chunk_paper(paper())
    second = service().chunk_paper(paper())

    assert first == second
    assert [chunk.chunk_id for chunk in first.chunks] == [
        "2610.01963::chunk::000",
        "2610.01963::chunk::001",
        "2610.01963::chunk::002",
    ]
    assert len({chunk.chunk_id for chunk in first.chunks}) == len(first.chunks)


def test_uses_actual_docling_section_representation_and_preserves_order() -> None:
    sections = [
        ParsedPDFSection(title="Introduction", text=words("intro", 60)),
        ParsedPDFSection(title="Methods", text=words("method", 60)),
    ]

    result = service().chunk_paper(paper(sections=sections))

    assert result.mode == "sections"
    assert [chunk.section_title for chunk in result.chunks] == ["Introduction", "Methods"]
    assert result.chunks[0].text.startswith("intro0")
    assert result.chunks[1].text.startswith("method0")


def test_long_section_splits_with_overlap_and_forward_progress() -> None:
    result = service().chunk_paper(
        paper(sections=[{"title": "Results", "text": words("token", 250)}])
    )

    assert [chunk.word_count for chunk in result.chunks] == [100, 100, 90]
    assert [chunk.has_overlap for chunk in result.chunks] == [False, True, True]
    assert result.chunks[0].text.split()[-20:] == result.chunks[1].text.split()[:20]
    assert result.chunks[1].text.split()[-20:] == result.chunks[2].text.split()[:20]


def test_zero_overlap_produces_disjoint_chunks() -> None:
    result = service(overlap=0).chunk_paper(paper(raw_text=words("word", 220)))

    assert [chunk.word_count for chunk in result.chunks] == [100, 100, 20]
    assert not any(chunk.has_overlap for chunk in result.chunks)
    assert result.chunks[0].text.split()[-1] != result.chunks[1].text.split()[0]


def test_adjacent_small_sections_are_combined_without_discarding_them() -> None:
    result = service(minimum=50).chunk_paper(
        paper(
            sections=[
                {"title": "A", "text": words("a", 20)},
                {"title": "B", "text": words("b", 25)},
                {"title": "C", "text": words("c", 20)},
            ]
        )
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].section_title == "A / B / C"
    assert result.chunks[0].word_count == 65


def test_missing_sections_falls_back_to_paragraph_aware_raw_text() -> None:
    raw_text = f"{words('first', 70)}\n\n{words('second', 70)}\n\n{words('third', 40)}"

    result = service().chunk_paper(paper(raw_text=raw_text, sections=[]))

    assert result.mode == "raw_text"
    assert result.chunks[0].word_count == 70
    assert result.chunks[1].text.startswith("first50")


@pytest.mark.parametrize("raw_text", [None, "", " \n\t "])
def test_empty_text_returns_empty_result(raw_text) -> None:
    result = service().chunk_paper(paper(raw_text=raw_text, sections=None))

    assert result.mode == "empty"
    assert result.chunks == ()


def test_malformed_sections_are_ignored_and_valid_entries_are_used() -> None:
    result = service().chunk_paper(
        paper(
            sections=[None, 42, {}, {"title": "Empty", "text": " "}, {"title": "Valid", "text": "naïve café 東京"}]
        )
    )

    assert result.mode == "sections"
    assert len(result.chunks) == 1
    assert result.chunks[0].section_title == "Valid"
    assert result.chunks[0].text == "naïve café 東京"


def test_all_malformed_sections_fall_back_to_raw_text() -> None:
    result = service().chunk_paper(paper(sections=[None, {"unexpected": "value"}]))

    assert result.mode == "raw_text"
    assert result.chunks


def test_metadata_is_propagated_to_every_non_empty_chunk() -> None:
    result = service().chunk_paper(paper())

    assert all(chunk.text and chunk.word_count > 0 for chunk in result.chunks)
    assert all(chunk.arxiv_id == "2610.01963" for chunk in result.chunks)
    assert all(chunk.paper_title == "Clinical AI" for chunk in result.chunks)
    assert all(chunk.authors == ("Ada Lovelace",) for chunk in result.chunks)
    assert all(chunk.categories == ("cs.AI", "cs.CL") for chunk in result.chunks)
    assert all(chunk.published_date == datetime(2026, 10, 1, tzinfo=timezone.utc) for chunk in result.chunks)


def test_chunking_does_not_mutate_paper() -> None:
    source = paper(sections=[{"title": "Intro", "text": words("x", 120)}])
    original = deepcopy(source.__dict__)

    service().chunk_paper(source)

    assert source.__dict__ == original


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller"):
        service(target=100, overlap=100)
    with pytest.raises(ValueError, match="must not exceed"):
        service(target=100, minimum=101)
