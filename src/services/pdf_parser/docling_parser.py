import asyncio
import logging
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel
from src.exceptions import PDFNoTextError, PDFParserError
from src.schemas.parsed_pdf import ParsedPDF, ParsedPDFSection

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"
BYTES_PER_MEGABYTE = 1024 * 1024
SUCCESSFUL_STATUSES = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}


class DoclingPDFParser:
    """Asynchronous application adapter for Docling PDF conversion."""

    def __init__(
        self,
        *,
        max_file_size_mb: int = 50,
        max_pages: int = 100,
        converter: DocumentConverter | None = None,
    ) -> None:
        if max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be at least 1")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")

        self.max_file_size_mb = max_file_size_mb
        self.max_pages = max_pages
        self._converter = converter or self._make_converter()

    async def parse_pdf(self, pdf_path: str | Path) -> ParsedPDF:
        """Validate and parse a local PDF without blocking the asyncio event loop."""
        path = Path(pdf_path)
        self._validate_input(path)
        logger.info("Parsing PDF with Docling: %s", path)

        try:
            return await asyncio.to_thread(self._convert, path)
        except PDFParserError:
            raise
        except Exception as exc:
            raise PDFParserError(f"Docling failed to parse PDF {path}: {exc}") from exc

    def _validate_input(self, path: Path) -> None:
        if not path.exists():
            raise PDFParserError(f"PDF file does not exist: {path}")
        if not path.is_file():
            raise PDFParserError(f"PDF path is not a regular file: {path}")

        try:
            file_size = path.stat().st_size
            if file_size == 0:
                raise PDFParserError(f"PDF file is empty: {path}")

            max_bytes = self.max_file_size_mb * BYTES_PER_MEGABYTE
            if file_size > max_bytes:
                raise PDFParserError(f"PDF file exceeds the {self.max_file_size_mb} MB size limit: {path}")

            with path.open("rb") as pdf_file:
                if pdf_file.read(len(PDF_SIGNATURE)) != PDF_SIGNATURE:
                    raise PDFParserError(f"File does not have a valid PDF signature: {path}")
        except PDFParserError:
            raise
        except OSError as exc:
            raise PDFParserError(f"Could not inspect PDF file {path}: {exc}") from exc

    def _convert(self, path: Path) -> ParsedPDF:
        try:
            result = self._converter.convert(
                path,
                raises_on_error=True,
                max_num_pages=self.max_pages,
                max_file_size=self.max_file_size_mb * BYTES_PER_MEGABYTE,
            )
        except Exception as exc:
            raise PDFParserError(f"Docling failed to convert PDF {path}: {exc}") from exc

        if result.status not in SUCCESSFUL_STATUSES:
            raise PDFParserError(f"Docling conversion failed for {path} with status: {result.status}")

        raw_text = result.document.export_to_markdown().strip()
        if not raw_text:
            raise PDFNoTextError(f"Docling produced no text for PDF: {path}")

        sections = self._extract_sections(result.document)
        page_count = len(result.pages) if result.pages is not None else None
        return ParsedPDF(
            raw_text=raw_text,
            sections=sections,
            parser_used="docling",
            page_count=page_count,
        )

    @staticmethod
    def _make_converter() -> DocumentConverter:
        # arXiv papers are born-digital PDFs. Disabling OCR and TableFormer keeps
        # this text/section milestone headless and avoids unnecessary native OCR/CV dependencies.
        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
        )

    @staticmethod
    def _extract_sections(document: Any) -> list[ParsedPDFSection]:
        sections: list[ParsedPDFSection] = []
        current_title: str | None = None
        current_content: list[str] = []

        def append_current_section() -> None:
            if current_title is None:
                return
            text = "\n\n".join(current_content).strip()
            sections.append(ParsedPDFSection(title=current_title, text=text))

        for item, _level in document.iterate_items():
            label = getattr(item, "label", None)
            text = getattr(item, "text", "").strip()
            if label == DocItemLabel.SECTION_HEADER:
                append_current_section()
                current_title = text
                current_content = []
            elif current_title is not None and text:
                current_content.append(text)

        append_current_section()
        return sections
