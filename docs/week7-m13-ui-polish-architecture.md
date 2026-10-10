# Week 7 M13: demo UI polish and final architecture diagram

W7-M13 changes how the Gradio demo presents a response and adds the final
architecture diagram to the page. It is presentation only: the API, the SSE
contract, the agent, the cache, Nginx, and Compose are untouched. The UI
still makes exactly one API call per question and renders only what that
call returns.

Public URL: <https://agenticrag.hbapps.dedyn.io>

## Page layout

In reading order:

1. **Header.** Title, one-line description, and a small row of technology
   badges.
2. **Ask a research question.** Text box, a wide primary `Ask` button, a
   secondary `Clear` button, and four example questions that fill the box
   when clicked.
3. **Status line.** The current backend status, announced as `role="status"`.
4. **Answer** (left, wider column). The validated answer as Markdown with its
   `[S1]` citations, followed by the summary row.
5. **Execution path** (right column; below the answer on narrow screens).
6. **Sources.** One card per source.
7. **How this system works.** Collapsed by default; a short explanation,
   four steps, and the architecture diagram.

The diagram is never above the question box, and while collapsed it takes
one line.

## What each part shows

**Summary row.** Built only from fields the API returned; a missing value is
left out rather than shown as zero.

| Item | Values |
|---|---|
| Cache | `CACHE HIT`, `CACHE MISS`, `CACHE BYPASS`, `CACHE UNAVAILABLE` |
| Grounding | `GROUNDED`, `NOT VALIDATED`, `NO VALIDATED ANSWER`, `OUT OF SCOPE`, `INSUFFICIENT EVIDENCE` |
| Latency | milliseconds under one second, otherwise seconds; measured in the UI process |
| Model | the model id as the API reports it |
| Sources | number of sources returned |
| Tokens | `N in / M out`; labelled "original run" on a cache hit |

The grounding badge appears only once the request has finished.
`GROUNDED` means the API reported a validated answer. `NO VALIDATED ANSWER`
covers `grounding_failed`, `generation_failed`, and failed requests.
`NOT VALIDATED` is used only when text is on screen that did not pass
validation, which the agent endpoint never produces.

**Execution path.** An ordered list of the status messages the backend
streamed, verbatim and in order, capped at 60. The UI never adds, renames,
or infers a step. A cache hit shows the single step "Validated cached answer
found".

**Source cards.** Citation label, a `LOCAL` or `LIVE ARXIV` badge, title,
arXiv id, section, and publication date when present. An "Open arXiv" link
appears only for an `https://arxiv.org/` URL from the API. The evidence
excerpt (at most 1200 characters) sits behind "View source details" instead
of on the card face. Every value is HTML-escaped.

**Outcomes.** `out_of_scope`, `insufficient_evidence`, `grounding_failed`,
and `generation_failed` show the API's fixed safe message in the answer
area, no sources, and the matching badge. Errors show the API's safe message
in the status line and no answer. Rejected or partial model text, stack
traces, and exception details are never rendered, as before.

**Clear** resets the question, answer, summary row, execution path, sources,
and status to their initial state.

## Styling

One CSS string and a Gradio theme in `src/gradio_app.py`, passed to
`launch()`. Colours for text, borders, and surfaces come from Gradio theme
variables, so light and dark both work; the three badge colours have
explicit dark-mode values. Every badge carries its meaning as text, not only
as colour. Fonts are the system UI stack, so nothing is fetched from a
third party. The page is capped at 1180 px and the two response columns
stack on narrow screens.

## Architecture diagram

| File | Purpose |
|---|---|
| `src/assets/architecture/production-agentic-rag-final.png` | Shown in the UI and README (3200 x 2524) |
| `src/assets/architecture/production-agentic-rag-final.svg` | Editable source |

The files are under `src/` because the Docker image copies only `src/`; no
Dockerfile or Compose change was needed and the UI loads the image from the
container's own filesystem. It is a static file: nothing is generated per
request.

It keeps the layout and colour language of the earlier "End-to-End Query
Journey" diagram and replaces what had gone stale:

| Earlier diagram | Now |
|---|---|
| `bge-large-en-v1.5` | `BAAI/bge-small-en-v1.5`, 384-dimensional |
| `rag:response:v1:<sha256>` | `rag:agent-response:v1:<sha256>`, agent-aware exact validated-response cache, with what the key covers |
| `POST /api/v1/ask`, `/ask/stream` | `POST /api/v1/agent/ask`, `/agent/ask/stream` |
| PDF acquisition, Docling parse, merge/rerank marked "Planned" | Implemented path: live search, bounded selection, selective PDF acquisition, Docling parse, transient evidence |
| One "Grounding validation" box | Finish-reason check, length recovery, structural citation / answer-contract validation, and the semantic answer grounding grader as separate steps |
| No regeneration | Bounded grounding regeneration on the same evidence (at most 2 grounding attempts) |
| Local and live evidence merged loosely | Normalize, then one common BGE semantic rerank; local OpenSearch scores are never compared with live scores |
| Langfuse: traces, latency, versioning | Adds Generation observations, token usage, cost when available, content capture off by default |
| Legend: "Future / planned (Week 7)" | "Every path shown is implemented" |

A unit test checks the SVG for the current labels and for the absence of
`Planned`, `bge-large`, `CrewAI`, and the old cache namespace.

To change the diagram, edit the SVG and re-render the PNG at twice its size
with any SVG renderer, for example headless Chromium:

```bash
docker run --rm -v "$PWD/src/assets/architecture:/w" mcr.microsoft.com/playwright/python:v1.49.0-jammy \
  bash -c 'pip install -q playwright==1.49.0 && python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={\"width\": 1600, \"height\": 1262}, device_scale_factor=2)
    page.goto(\"file:///w/production-agentic-rag-final.svg\")
    page.locator(\"svg\").screenshot(path=\"/w/production-agentic-rag-final.png\")
"'
```

## Deploying a UI change

Gradio and the API share one image, so a UI change rebuilds it once and
Compose recreates both containers:

```bash
docker compose build gradio
docker compose up -d --wait
docker compose ps -a
```

The public site answers 502 for the minute or two this takes. No volume is
touched.

## Acceptance (2026-10-10)

Checked on <https://agenticrag.hbapps.dedyn.io> with headless Chromium at
1366 x 768 in light and dark themes and at 390 x 844, after rebuilding the
image and recreating `api` and `gradio`:

- The page loads over HTTPS with no console errors and no `http://` request.
- Clicking the first example fills the question; `Ask` returns the cached
  answer with `CACHE HIT`, `GROUNDED`, latency under 120 ms, the model,
  5 sources, original-run tokens, five `LOCAL` source cards, and the single
  step "Validated cached answer found".
- "How this system works" starts collapsed, expands, and loads the diagram
  (3200 x 2524, scaled to the column) from the site's own origin.
- `Clear` empties the question, source cards, execution path, and summary
  row, and the status returns to "Ready".
- No horizontal scrolling at any of the three sizes; the columns stack at
  390 px.
- All seven containers are healthy and every service port is still bound to
  `127.0.0.1`.

A cache miss was not run against the live site, to avoid model and live
arXiv usage; progressive status rendering, outcomes, and errors are covered
by `tests/unit/test_gradio_app.py`.

## Risks and follow-ups

- At laptop width the diagram's smaller labels are about 8 px tall; the
  fullscreen button on the image is the intended way to read them.
- Several sources from one paper show the same title, and a card only
  differs by its citation label when the chunk has no section title.
- Three of the four example questions were not checked against the live
  corpus; any of them may take the slow live-fallback path the first time.
- The PNG is rendered from the SVG by hand; the unit test checks the SVG's
  labels, not that the PNG matches it.
