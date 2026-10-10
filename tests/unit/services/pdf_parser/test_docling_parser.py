from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import DocItemLabel
from src.exceptions import PDFNoTextError, PDFParserError
from src.services.pdf_parser.docling_parser import BYTES_PER_MEGABYTE, DoclingPDFParser


class FakeDocument:
    def export_to_markdown(self) -> str:
        return "# Example Paper\n\n## Abstract\n\nSummary text.\n\n## Methods\n\nMethod text."

    def iterate_items(self):
        yield SimpleNamespace(label=DocItemLabel.TITLE, text="Example Paper"), 0
        yield SimpleNamespace(label=DocItemLabel.SECTION_HEADER, text="Abstract"), 1
        yield SimpleNamespace(label=DocItemLabel.TEXT, text="Summary text."), 2
        yield SimpleNamespace(label=DocItemLabel.SECTION_HEADER, text="Methods"), 1
        yield SimpleNamespace(label=DocItemLabel.TEXT, text="Method text."), 2


class FakeConverter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[object, dict[str, object]]] = []

    def convert(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            document=FakeDocument(),
            pages=[object(), object(), object()],
        )


def write_pdf(path, content: bytes = b"%PDF-1.7\nmock") -> None:
    path.write_bytes(content)


@pytest.mark.anyio
async def test_parse_pdf_rejects_nonexistent_file(tmp_path) -> None:
    parser = DoclingPDFParser(converter=FakeConverter())

    with pytest.raises(PDFParserError, match="does not exist"):
        await parser.parse_pdf(tmp_path / "missing.pdf")


@pytest.mark.anyio
async def test_parse_pdf_rejects_directory(tmp_path) -> None:
    parser = DoclingPDFParser(converter=FakeConverter())

    with pytest.raises(PDFParserError, match="not a regular file"):
        await parser.parse_pdf(tmp_path)


@pytest.mark.anyio
async def test_parse_pdf_rejects_empty_file(tmp_path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()
    parser = DoclingPDFParser(converter=FakeConverter())

    with pytest.raises(PDFParserError, match="empty"):
        await parser.parse_pdf(pdf_path)


@pytest.mark.anyio
async def test_parse_pdf_rejects_non_pdf_file(tmp_path) -> None:
    pdf_path = tmp_path / "invalid.pdf"
    write_pdf(pdf_path, b"not a PDF")
    parser = DoclingPDFParser(converter=FakeConverter())

    with pytest.raises(PDFParserError, match="valid PDF signature"):
        await parser.parse_pdf(pdf_path)


@pytest.mark.anyio
async def test_parse_pdf_enforces_file_size_limit(tmp_path) -> None:
    pdf_path = tmp_path / "large.pdf"
    write_pdf(pdf_path, b"%PDF-" + b"x" * BYTES_PER_MEGABYTE)
    parser = DoclingPDFParser(max_file_size_mb=1, converter=FakeConverter())

    with pytest.raises(PDFParserError, match="size limit"):
        await parser.parse_pdf(pdf_path)


@pytest.mark.anyio
async def test_parse_pdf_maps_docling_result_to_application_schema(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    write_pdf(pdf_path)
    converter = FakeConverter()
    parser = DoclingPDFParser(max_file_size_mb=12, max_pages=25, converter=converter)

    result = await parser.parse_pdf(pdf_path)

    assert result.raw_text.startswith("# Example Paper")
    assert result.parser_used == "docling"
    assert result.page_count == 3
    assert [(section.title, section.text) for section in result.sections] == [
        ("Abstract", "Summary text."),
        ("Methods", "Method text."),
    ]
    assert converter.calls == [
        (
            pdf_path,
            {
                "raises_on_error": True,
                "max_num_pages": 25,
                "max_file_size": 12 * BYTES_PER_MEGABYTE,
            },
        )
    ]


@pytest.mark.anyio
async def test_parse_pdf_translates_docling_failure(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    write_pdf(pdf_path)
    parser = DoclingPDFParser(converter=FakeConverter(error=RuntimeError("conversion exploded")))

    with pytest.raises(PDFParserError, match="Docling failed to convert") as exc_info:
        await parser.parse_pdf(pdf_path)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


class EmptyTextDocument(FakeDocument):
    def export_to_markdown(self) -> str:
        return "  \n\n  "


@pytest.mark.anyio
async def test_parse_pdf_signals_no_extractable_text_with_a_typed_parser_error(tmp_path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    write_pdf(pdf_path)
    converter = FakeConverter()
    converter.convert = lambda path, **kwargs: SimpleNamespace(
        status=ConversionStatus.SUCCESS, document=EmptyTextDocument(), pages=[object()]
    )

    with pytest.raises(PDFNoTextError, match="produced no text") as caught:
        await DoclingPDFParser(converter=converter).parse_pdf(pdf_path)

    assert isinstance(caught.value, PDFParserError)


@pytest.mark.anyio
async def test_conversion_failure_is_not_reported_as_no_text(tmp_path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    write_pdf(pdf_path)

    with pytest.raises(PDFParserError) as caught:
        await DoclingPDFParser(converter=FakeConverter(error=RuntimeError("boom"))).parse_pdf(pdf_path)

    assert not isinstance(caught.value, PDFNoTextError)
