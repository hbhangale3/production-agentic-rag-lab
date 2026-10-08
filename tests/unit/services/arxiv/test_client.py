from datetime import date, datetime, timezone

import httpx
import pytest
from src.exceptions import ArxivClientError, ArxivPDFDownloadError
from src.schemas.arxiv import ArxivPaper
from src.services.arxiv.client import ArxivClient

ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2412.12345v2</id>
    <updated>2025-01-02T03:04:05Z</updated>
    <published>2024-12-20T10:11:12Z</published>
    <title>  A Paper\n      About Agents  </title>
    <summary> An abstract with\n      whitespace. </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <link href="https://arxiv.org/pdf/2412.12345v2" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>
"""


@pytest.fixture
def paper() -> ArxivPaper:
    return ArxivPaper(
        arxiv_id="2412.12345",
        title="A Paper About Agents",
        authors=["Ada Lovelace"],
        abstract="An abstract.",
        categories=["cs.AI"],
        published_date=datetime(2024, 12, 20, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/2412.12345v2",
    )


@pytest.mark.anyio
async def test_fetch_papers_parses_structured_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=ARXIV_RESPONSE, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        http_client=http_client,
    )
    try:
        papers = await client.fetch_papers()
    finally:
        await http_client.aclose()

    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2412.12345"
    assert paper.title == "A Paper About Agents"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.abstract == "An abstract with whitespace."
    assert paper.categories == ["cs.AI", "cs.CL"]
    assert paper.published_date == datetime(2024, 12, 20, 10, 11, 12, tzinfo=timezone.utc)
    assert paper.updated_date == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/2412.12345v2"


@pytest.mark.anyio
async def test_fetch_papers_builds_date_query_and_sort_parameters() -> None:
    captured_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, text=ARXIV_RESPONSE, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        search_category="cs.LG",
        max_results=7,
        rate_limit_delay=0,
        http_client=http_client,
    )
    try:
        await client.fetch_papers(
            from_date=date(2025, 1, 1),
            to_date=date(2025, 1, 31),
            sort_order="ascending",
        )
    finally:
        await http_client.aclose()

    assert captured_request is not None
    assert captured_request.url.params["search_query"] == ("cat:cs.LG AND submittedDate:[202501010000 TO 202501312359]")
    assert captured_request.url.params["max_results"] == "7"
    assert captured_request.url.params["sortBy"] == "submittedDate"
    assert captured_request.url.params["sortOrder"] == "ascending"


@pytest.mark.anyio
async def test_fetch_papers_raises_useful_error_for_http_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad query", request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        http_client=http_client,
    )
    try:
        with pytest.raises(ArxivClientError, match="HTTP 400"):
            await client.fetch_papers()
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_fetch_papers_retries_transient_failure() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        status_code = 503 if attempts == 1 else 200
        return httpx.Response(status_code, text=ARXIV_RESPONSE, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        max_retries=1,
        http_client=http_client,
    )
    try:
        papers = await client.fetch_papers()
    finally:
        await http_client.aclose()

    assert attempts == 2
    assert len(papers) == 1


def test_build_search_query_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="from_date"):
        ArxivClient.build_search_query(
            category="cs.AI",
            from_date=date(2025, 2, 1),
            to_date=date(2025, 1, 1),
        )


@pytest.mark.anyio
async def test_download_pdf_creates_valid_cached_file(tmp_path, paper: ArxivPaper) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.7\nmock pdf content",
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        pdf_cache_dir=tmp_path,
        http_client=http_client,
    )
    try:
        pdf_path = await client.download_pdf(paper)
    finally:
        await http_client.aclose()

    assert pdf_path == tmp_path / "2412.12345.pdf"
    assert pdf_path.read_bytes() == b"%PDF-1.7\nmock pdf content"
    assert not (tmp_path / "2412.12345.pdf.part").exists()


@pytest.mark.anyio
async def test_download_pdf_cache_hit_avoids_network(tmp_path, paper: ArxivPaper) -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, content=b"%PDF-1.7\nfirst download", request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        pdf_cache_dir=tmp_path,
        http_client=http_client,
    )
    try:
        first_path = await client.download_pdf(paper)
        second_path = await client.download_pdf(paper)
    finally:
        await http_client.aclose()

    assert first_path == second_path
    assert request_count == 1


@pytest.mark.anyio
async def test_download_pdf_rejects_non_pdf_and_removes_partial_file(tmp_path, paper: ArxivPaper) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=b"<html>not a PDF</html>",
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        pdf_cache_dir=tmp_path,
        http_client=http_client,
    )
    try:
        with pytest.raises(ArxivPDFDownloadError, match="not a valid PDF"):
            await client.download_pdf(paper)
    finally:
        await http_client.aclose()

    assert not (tmp_path / "2412.12345.pdf").exists()
    assert not (tmp_path / "2412.12345.pdf.part").exists()


@pytest.mark.anyio
async def test_download_pdf_raises_for_http_failure(tmp_path, paper: ArxivPaper) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        pdf_cache_dir=tmp_path,
        http_client=http_client,
    )
    try:
        with pytest.raises(ArxivPDFDownloadError, match="HTTP 404"):
            await client.download_pdf(paper)
    finally:
        await http_client.aclose()

    assert not (tmp_path / "2412.12345.pdf.part").exists()


def test_pdf_filename_is_safe_for_legacy_and_malicious_ids() -> None:
    assert ArxivClient.pdf_filename("hep-th/9901001") == "hep-th_9901001.pdf"
    assert ArxivClient.pdf_filename("../../dangerous/id") == "dangerous_id.pdf"
