from datetime import date, datetime, timezone

import httpx
import pytest
from src.exceptions import ArxivClientError
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
