from src.config import Settings, get_settings
from src.services.pdf_parser.docling_parser import DoclingPDFParser


def make_pdf_parser_service(settings: Settings | None = None) -> DoclingPDFParser:
    """Create the configured PDF parser service."""
    config = settings or get_settings()
    return DoclingPDFParser(
        max_file_size_mb=config.pdf_parser_max_file_size_mb,
        max_pages=config.pdf_parser_max_pages,
    )
