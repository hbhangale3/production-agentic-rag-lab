import dataclasses
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from src.exceptions import ArxivClientError
from src.schemas.arxiv import ArxivPaper
from src.services.agent import ArxivLiveSearchService, LiveArxivPaper, LiveArxivSearchResult, LiveSearchError
from src.services.agent.live_search import MAX_LIVE_QUERY_TERMS, build_arxiv_search_query, normalize_arxiv_id
from src.services.arxiv.client import ArxivClient

QUERY = "India healthcare challenges artificial intelligence health equity"


def make_paper(arxiv_id: str = "2501.00001", *, title: str = "A Live Paper", abstract: str = "An abstract.") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Ada Lovelace", "  Alan   Turing ", " "],
        abstract=abstract,
        categories=["cs.AI", " cs.CY "],
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        updated_date=datetime(2025, 2, 3, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def make_service(*papers: ArxivPaper, error: Exception | None = None) -> tuple[ArxivLiveSearchService, Mock]:
    client = Mock(spec=ArxivClient)
    client.fetch_papers = AsyncMock(return_value=list(papers), side_effect=error)
    return ArxivLiveSearchService(arxiv_client=client), client


@pytest.mark.anyio
async def test_normalizes_existing_arxiv_metadata_into_typed_candidates() -> None:
    service, client = make_service(make_paper("2501.00001v3", title="  A Live\n  Paper ", abstract=" An\n abstract. "))

    result = await service.search(f"  {QUERY}\n", max_results=5)

    assert result == LiveArxivSearchResult(
        query=QUERY,
        candidates=(
            LiveArxivPaper(
                arxiv_id="2501.00001",
                title="A Live Paper",
                abstract="An abstract.",
                authors=("Ada Lovelace", "Alan Turing"),
                categories=("cs.AI", "cs.CY"),
                published_date=datetime(2025, 1, 2, tzinfo=UTC),
                updated_date=datetime(2025, 2, 3, tzinfo=UTC),
                pdf_url="https://arxiv.org/pdf/2501.00001v3",
            ),
        ),
    )
    assert result.count == 1
    client.fetch_papers.assert_awaited_once_with(
        search_query=(
            "all:India OR all:healthcare OR all:challenges OR all:artificial OR all:intelligence "
            "OR all:health OR all:equity"
        ),
        max_results=5,
        sort_by="relevance",
        sort_order="descending",
    )


@pytest.mark.anyio
async def test_only_the_read_only_search_method_of_the_client_is_used() -> None:
    service, client = make_service(make_paper())

    await service.search(QUERY, max_results=5)

    assert [call[0] for call in client.method_calls] == ["fetch_papers"]
    client.download_pdf.assert_not_called()


@pytest.mark.anyio
async def test_preserves_source_order_and_keeps_first_duplicate() -> None:
    service, _ = make_service(
        make_paper("2501.00003", title="Third"),
        make_paper("2501.00001v1", title="First"),
        make_paper("2501.00003v2", title="Duplicate of third"),
        make_paper("2501.00002", title="Second"),
        make_paper("2501.00001v2", title="Duplicate of first"),
    )

    first = await service.search(QUERY, max_results=5)
    second = await service.search(QUERY, max_results=5)

    assert first == second
    assert [(paper.arxiv_id, paper.title) for paper in first.candidates] == [
        ("2501.00003", "Third"),
        ("2501.00001", "First"),
        ("2501.00002", "Second"),
    ]


@pytest.mark.anyio
async def test_entries_without_usable_abstract_or_title_are_filtered() -> None:
    service, _ = make_service(
        make_paper("2501.00001", abstract=""),
        make_paper("2501.00002", abstract=" \n\t "),
        make_paper("2501.00003", title="  "),
        make_paper("2501.00004"),
    )

    result = await service.search(QUERY, max_results=5)

    assert [paper.arxiv_id for paper in result.candidates] == ["2501.00004"]


@pytest.mark.anyio
async def test_result_limit_is_enforced_even_if_client_returns_more() -> None:
    service, client = make_service(*(make_paper(f"2501.{index:05d}") for index in range(1, 9)))

    result = await service.search(QUERY, max_results=3)

    assert [paper.arxiv_id for paper in result.candidates] == ["2501.00001", "2501.00002", "2501.00003"]
    assert client.fetch_papers.await_args.kwargs["max_results"] == 3


@pytest.mark.anyio
async def test_already_local_papers_are_excluded_and_request_stays_bounded() -> None:
    service, client = make_service(
        make_paper("2401.00001v2"),
        make_paper("2501.00002"),
        make_paper("2401.00009"),
        make_paper("2501.00004"),
    )

    result = await service.search(
        QUERY, max_results=2, exclude_arxiv_ids=("2401.00001", " 2401.00009v4 ", "2401.00001", "")
    )

    assert [paper.arxiv_id for paper in result.candidates] == ["2501.00002", "2501.00004"]
    assert client.fetch_papers.await_args.kwargs["max_results"] == 4

    await service.search(QUERY, max_results=2, exclude_arxiv_ids=[f"2401.{index:05d}" for index in range(50)])
    assert client.fetch_papers.await_args.kwargs["max_results"] == 4


@pytest.mark.anyio
async def test_zero_results_is_a_successful_empty_result() -> None:
    service, _ = make_service()

    result = await service.search(QUERY, max_results=5)

    assert result == LiveArxivSearchResult(query=QUERY)
    assert result.count == 0


@pytest.mark.anyio
async def test_query_without_search_terms_returns_empty_without_network() -> None:
    service, client = make_service(make_paper())

    result = await service.search("?? -- !! a", max_results=5)

    assert result.count == 0
    client.fetch_papers.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [ArxivClientError("private https://export.arxiv.org detail"), RuntimeError("private detail"), ValueError("private")],
)
async def test_client_failure_becomes_controlled_error_without_details(error: Exception) -> None:
    service, _ = make_service(error=error)

    with pytest.raises(LiveSearchError) as caught:
        await service.search(QUERY, max_results=5)

    assert str(caught.value) == "live arXiv search failed"


@pytest.mark.anyio
@pytest.mark.parametrize(("query", "max_results"), [("", 5), ("  ", 5), (QUERY, 0), (QUERY, True), (QUERY, 2.0)])
async def test_invalid_arguments_are_programming_errors(query: str, max_results) -> None:
    service, client = make_service()

    with pytest.raises(ValueError):
        await service.search(query, max_results=max_results)

    client.fetch_papers.assert_not_awaited()


@pytest.mark.anyio
async def test_real_client_makes_one_awaited_metadata_request_and_no_pdf_request(tmp_path) -> None:
    requests: list[httpx.Request] = []
    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v2</id>
    <updated>2025-02-03T00:00:00Z</updated>
    <published>2025-01-02T00:00:00Z</published>
    <title>Live Paper</title>
    <summary>Live abstract.</summary>
    <author><name>Ada Lovelace</name></author>
    <category term="cs.CY"/>
    <link href="https://arxiv.org/pdf/2501.00001v2" type="application/pdf" title="pdf"/>
  </entry>
</feed>"""

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=feed, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pdf_dir = tmp_path / "pdfs"
    client = ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=0,
        pdf_cache_dir=pdf_dir,
        http_client=http_client,
    )
    try:
        result = await ArxivLiveSearchService(arxiv_client=client).search('AI" OR cat:cs.* ) AND (health', max_results=5)
    finally:
        await http_client.aclose()

    assert len(requests) == 1
    assert requests[0].url.path == "/api/query"
    assert requests[0].url.params["search_query"] == "all:AI OR all:cat OR all:cs OR all:health"
    assert requests[0].url.params["sortBy"] == "relevance"
    assert requests[0].url.params["max_results"] == "5"
    assert result.candidates[0].arxiv_id == "2501.00001"
    assert result.candidates[0].pdf_url == "https://arxiv.org/pdf/2501.00001v2"
    assert not pdf_dir.exists()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What are the current healthcare issues in India and how can AI help us solve them?",
         "all:healthcare OR all:issues OR all:India OR all:AI OR all:help OR all:solve"),
        ('"quoted phrase" AND (cat:cs.AI) OR ti:x', "all:quoted OR all:phrase OR all:cat OR all:cs OR all:AI OR all:ti"),
        ("NLP nlp Nlp clinical", "all:NLP OR all:clinical"),
        ("health-equity_tech", "all:health OR all:equity OR all:tech"),
        ("\r\n\x00 ; ) ( : *", None),
    ],
)
def test_query_construction_treats_text_as_terms_not_syntax(query: str, expected: str | None) -> None:
    assert build_arxiv_search_query(query) == expected


def test_query_construction_caps_the_number_of_terms() -> None:
    built = build_arxiv_search_query(" ".join(f"term{index}" for index in range(100)))

    assert built is not None
    assert built.count("all:") == MAX_LIVE_QUERY_TERMS
    assert built.startswith("all:term0 OR all:term1 ")


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [("2501.00001", "2501.00001"), (" 2501.00001v12 ", "2501.00001"), ("hep-th/9901001v1", "hep-th/9901001")],
)
def test_arxiv_id_normalization_reuses_version_stripping(raw: str, normalized: str) -> None:
    assert normalize_arxiv_id(raw) == normalized


def test_candidate_and_result_are_immutable_value_types_without_foreign_objects() -> None:
    paper = LiveArxivPaper(
        arxiv_id="2501.00001",
        title="Title",
        abstract="Abstract",
        authors=("Ada",),
        categories=("cs.AI",),
        published_date=datetime(2025, 1, 2, tzinfo=UTC),
        updated_date=None,
        pdf_url="https://arxiv.org/pdf/2501.00001",
    )
    result = LiveArxivSearchResult(query="q", candidates=(paper,))

    assert [field.name for field in dataclasses.fields(LiveArxivPaper)] == [
        "arxiv_id",
        "title",
        "abstract",
        "authors",
        "categories",
        "published_date",
        "updated_date",
        "pdf_url",
    ]
    assert [field.name for field in dataclasses.fields(LiveArxivSearchResult)] == ["query", "candidates"]
    assert result == LiveArxivSearchResult(query="q", candidates=(dataclasses.replace(paper),))
    assert hash(paper) == hash(dataclasses.replace(paper))
    with pytest.raises(dataclasses.FrozenInstanceError):
        paper.title = "Other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.candidates = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [{"arxiv_id": " "}, {"title": ""}, {"abstract": "  "}, {"pdf_url": ""}, {"authors": ["Ada"]}, {"categories": [1]}],
)
def test_candidate_rejects_blank_required_fields_and_mutable_collections(overrides: dict) -> None:
    fields = {
        "arxiv_id": "2501.00001",
        "title": "Title",
        "abstract": "Abstract",
        "authors": ("Ada",),
        "categories": ("cs.AI",),
        "published_date": datetime(2025, 1, 2, tzinfo=UTC),
        "updated_date": None,
        "pdf_url": "https://arxiv.org/pdf/2501.00001",
    }

    with pytest.raises(ValueError):
        LiveArxivPaper(**{**fields, **overrides})


@pytest.mark.parametrize("candidates", [[], (SimpleNamespace(arxiv_id="x"),), (make_paper(),)])
def test_result_rejects_lists_and_foreign_candidate_objects(candidates) -> None:
    with pytest.raises(ValueError):
        LiveArxivSearchResult(query="q", candidates=candidates)
