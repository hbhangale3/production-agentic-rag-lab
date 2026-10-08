import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from datetime import time as datetime_time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal

import httpx
from src.exceptions import ArxivClientError, ArxivPDFDownloadError
from src.schemas.arxiv import ArxivPaper

logger = logging.getLogger(__name__)

SortOrder = Literal["ascending", "descending"]

ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
NAMESPACES = {"atom": ATOM_NAMESPACE, "arxiv": ARXIV_NAMESPACE}
TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")
UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
PDF_SIGNATURE = b"%PDF-"


class ArxivClient:
    """Asynchronous client for querying and parsing the public arXiv API."""

    def __init__(
        self,
        *,
        base_url: str,
        search_category: str = "cs.AI",
        max_results: int = 10,
        rate_limit_delay: float = 3.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        pdf_cache_dir: str | Path = "data/arxiv_pdfs",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if rate_limit_delay < 0:
            raise ValueError("rate_limit_delay cannot be negative")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.base_url = base_url
        self.search_category = search_category
        self.max_results = max_results
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.pdf_cache_dir = Path(pdf_cache_dir)
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_at: float | None = None

    async def __aenter__(self) -> "ArxivClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally managed HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def fetch_papers(
        self,
        *,
        category: str | None = None,
        max_results: int | None = None,
        sort_order: SortOrder = "descending",
        from_date: date | datetime | None = None,
        to_date: date | datetime | None = None,
    ) -> list[ArxivPaper]:
        """Fetch paper metadata, newest first by default."""
        result_limit = max_results if max_results is not None else self.max_results
        if result_limit < 1:
            raise ValueError("max_results must be at least 1")
        if sort_order not in ("ascending", "descending"):
            raise ValueError("sort_order must be 'ascending' or 'descending'")

        query = self.build_search_query(
            category=category or self.search_category,
            from_date=from_date,
            to_date=to_date,
        )
        params = {
            "search_query": query,
            "start": 0,
            "max_results": result_limit,
            "sortBy": "submittedDate",
            "sortOrder": sort_order,
        }
        response = await self._request(params)
        return self.parse_response(response.text)

    async def download_pdf(self, paper: ArxivPaper) -> Path:
        """Download and atomically cache a paper PDF, or reuse a valid cached copy."""
        target_path = self.pdf_cache_dir / self.pdf_filename(paper.arxiv_id)
        partial_path = target_path.with_suffix(".pdf.part")

        try:
            self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
            if self._is_valid_cached_pdf(target_path):
                logger.info("Using cached arXiv PDF: %s", target_path)
                return target_path

            if target_path.exists():
                logger.warning("Removing invalid cached arXiv PDF: %s", target_path)
                target_path.unlink()

            partial_path.unlink(missing_ok=True)
            await self._download_pdf_with_retries(str(paper.pdf_url), partial_path)
            partial_path.replace(target_path)
            logger.info("Cached arXiv PDF at %s", target_path)
            return target_path
        except ArxivPDFDownloadError:
            partial_path.unlink(missing_ok=True)
            raise
        except OSError as exc:
            partial_path.unlink(missing_ok=True)
            raise ArxivPDFDownloadError(f"Could not cache arXiv PDF {paper.arxiv_id}: {exc}") from exc

    @staticmethod
    def pdf_filename(arxiv_id: str) -> str:
        """Return a deterministic filesystem-safe filename for an arXiv ID."""
        safe_id = UNSAFE_FILENAME_CHARACTERS.sub("_", arxiv_id.strip()).strip("._-")
        if not safe_id:
            raise ArxivPDFDownloadError("Cannot create a PDF filename from an empty arXiv ID")
        return f"{safe_id}.pdf"

    @classmethod
    def build_search_query(
        cls,
        *,
        category: str,
        from_date: date | datetime | None = None,
        to_date: date | datetime | None = None,
    ) -> str:
        """Build an arXiv query with an optional inclusive submission-date range."""
        if not category.strip():
            raise ValueError("category cannot be empty")

        start = cls._normalise_boundary(from_date, end_of_day=False) if from_date else None
        end = cls._normalise_boundary(to_date, end_of_day=True) if to_date else None
        if start and end and start > end:
            raise ValueError("from_date cannot be later than to_date")

        query = f"cat:{category.strip()}"
        if start or end:
            start_value = cls._format_arxiv_date(start) if start else "000001010000"
            end_value = cls._format_arxiv_date(end) if end else "999912312359"
            query += f" AND submittedDate:[{start_value} TO {end_value}]"
        return query

    async def _request(self, params: dict[str, str | int]) -> httpx.Response:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            await self._respect_rate_limit()
            response: httpx.Response | None = None
            try:
                logger.info("Querying arXiv API (attempt %d/%d)", attempt + 1, attempts)
                response = await self._client.get(self.base_url, params=params)
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    response.raise_for_status()
                    return response
                error: Exception = httpx.HTTPStatusError(
                    f"Transient arXiv response: {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                error = exc
            except httpx.HTTPStatusError as exc:
                raise ArxivClientError(f"arXiv API returned HTTP {exc.response.status_code}") from exc

            if attempt == attempts - 1:
                raise ArxivClientError(f"arXiv API request failed after {attempts} attempts") from error

            retry_delay = self._retry_delay(response, attempt)
            logger.warning("Transient arXiv API failure; retrying in %.1f seconds", retry_delay)
            await asyncio.sleep(retry_delay)

        raise AssertionError("unreachable")

    async def _download_pdf_with_retries(self, url: str, partial_path: Path) -> None:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            await self._respect_rate_limit()
            response: httpx.Response | None = None
            try:
                logger.info("Downloading arXiv PDF (attempt %d/%d): %s", attempt + 1, attempts, url)
                async with self._client.stream("GET", url) as response:
                    if response.status_code in TRANSIENT_STATUS_CODES:
                        error: Exception = httpx.HTTPStatusError(
                            f"Transient arXiv PDF response: {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    else:
                        response.raise_for_status()
                        await self._stream_pdf_to_file(response, partial_path)
                        return
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                error = exc
            except httpx.HTTPStatusError as exc:
                partial_path.unlink(missing_ok=True)
                raise ArxivPDFDownloadError(f"arXiv PDF request returned HTTP {exc.response.status_code}") from exc
            except ArxivPDFDownloadError:
                partial_path.unlink(missing_ok=True)
                raise
            except OSError as exc:
                partial_path.unlink(missing_ok=True)
                raise ArxivPDFDownloadError(f"Could not write arXiv PDF: {exc}") from exc

            partial_path.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise ArxivPDFDownloadError(f"arXiv PDF download failed after {attempts} attempts") from error

            retry_delay = self._retry_delay(response, attempt)
            logger.warning("Transient arXiv PDF failure; retrying in %.1f seconds", retry_delay)
            await asyncio.sleep(retry_delay)

        raise AssertionError("unreachable")

    @staticmethod
    async def _stream_pdf_to_file(response: httpx.Response, partial_path: Path) -> None:
        signature = bytearray()
        bytes_written = 0
        with partial_path.open("wb") as pdf_file:
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if len(signature) < len(PDF_SIGNATURE):
                    needed = len(PDF_SIGNATURE) - len(signature)
                    signature.extend(chunk[:needed])
                pdf_file.write(chunk)
                bytes_written += len(chunk)

        content_type = response.headers.get("Content-Type", "").lower()
        if bytes_written == 0:
            raise ArxivPDFDownloadError("arXiv returned an empty PDF response")
        if bytes(signature) != PDF_SIGNATURE:
            detail = f" (Content-Type: {content_type})" if content_type else ""
            raise ArxivPDFDownloadError(f"arXiv response is not a valid PDF{detail}")

    @staticmethod
    def _is_valid_cached_pdf(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < len(PDF_SIGNATURE):
                return False
            with path.open("rb") as pdf_file:
                return pdf_file.read(len(PDF_SIGNATURE)) == PDF_SIGNATURE
        except OSError:
            return False

    async def _respect_rate_limit(self) -> None:
        async with self._rate_limit_lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                remaining = self.rate_limit_delay - (now - self._last_request_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_request_at = time.monotonic()

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), self.rate_limit_delay)
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                        return max(seconds, self.rate_limit_delay, 0.0)
                    except (TypeError, ValueError):
                        pass
        return max(self.rate_limit_delay, float(2**attempt))

    @staticmethod
    def parse_response(xml_text: str) -> list[ArxivPaper]:
        """Parse an arXiv Atom feed into validated metadata objects."""
        try:
            root = ET.fromstring(xml_text)
            return [ArxivClient._parse_entry(entry) for entry in root.findall("atom:entry", NAMESPACES)]
        except (ET.ParseError, TypeError, ValueError) as exc:
            raise ArxivClientError("Could not parse the arXiv API response") from exc

    @staticmethod
    def _parse_entry(entry: ET.Element) -> ArxivPaper:
        entry_id = ArxivClient._required_text(entry, "atom:id")
        arxiv_id = ARXIV_VERSION_SUFFIX.sub("", entry_id.rstrip("/").rsplit("/", 1)[-1])
        pdf_url = next(
            (
                link.attrib["href"]
                for link in entry.findall("atom:link", NAMESPACES)
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf"
            ),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        updated_text = entry.findtext("atom:updated", namespaces=NAMESPACES)

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=ArxivClient._clean_text(ArxivClient._required_text(entry, "atom:title")),
            authors=[ArxivClient._clean_text(name.text or "") for name in entry.findall("atom:author/atom:name", NAMESPACES)],
            abstract=ArxivClient._clean_text(ArxivClient._required_text(entry, "atom:summary")),
            categories=[category.attrib["term"] for category in entry.findall("atom:category", NAMESPACES)],
            published_date=ArxivClient._parse_datetime(ArxivClient._required_text(entry, "atom:published")),
            updated_date=ArxivClient._parse_datetime(updated_text) if updated_text else None,
            pdf_url=pdf_url,
        )

    @staticmethod
    def _required_text(entry: ET.Element, path: str) -> str:
        value = entry.findtext(path, namespaces=NAMESPACES)
        if not value or not value.strip():
            raise ValueError(f"Missing required arXiv field: {path}")
        return value.strip()

    @staticmethod
    def _clean_text(value: str) -> str:
        return " ".join(value.split())

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _normalise_boundary(value: date | datetime, *, end_of_day: bool) -> datetime:
        if isinstance(value, datetime):
            return value
        boundary_time = datetime_time(23, 59) if end_of_day else datetime_time.min
        return datetime.combine(value, boundary_time)

    @staticmethod
    def _format_arxiv_date(value: datetime) -> str:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.strftime("%Y%m%d%H%M")
