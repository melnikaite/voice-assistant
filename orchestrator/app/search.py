"""DuckDuckGo web search — async wrapper around the synchronous DDGS API,
plus parallel page fetching so the summariser sees real content, not just
the 200-300 character preview DDG ships in results.

Localisation: DDG accepts a ``region`` (a.k.a. ``kl``) parameter that biases
results to a country/language.  We default to the ``DDG_REGION`` env var
(typically ``de-de`` for a German household) — DDG itself geo-detects from
the calling IP, but the ``ddgs`` library only honours the explicit ``region``
parameter, so we still have to send one.

Page enrichment: after DDG returns its result list we fetch the top N URLs
in parallel (httpx), then extract the *main content block* — not the whole
``<body>`` — using two extractors in tandem:

  1. **trafilatura** (primary).  Identifies the article/main-content area
     using readability-style heuristics; drops navigation, sidebars, ads,
     cookie banners, related-article rails, comments, footers.  Much
     slower than tag-stripping (~200-500 ms/page) but the quality is
     dramatically better — the LLM gets the actual content, not the
     page's UI chrome.
  2. **selectolax** (fallback).  When trafilatura can't identify a main
     block (some SPA pages, malformed HTML), we fall back to stripping
     known noise tags and taking whatever's in <body>.  Cruder, but
     better than nothing.

Each extraction runs in a thread (``asyncio.to_thread``) so CPU-bound parse
work of N pages happens concurrently rather than serialised on the loop.

``format_results`` prefers ``page_text`` over the short ``body`` snippet
when feeding the LLM and prefixes each block with ``[page]`` / ``[snippet]``
so the summarisation prompt can weight them appropriately.

Security note: search snippets AND fetched page bodies are UNTRUSTED data
from external websites and may contain prompt-injection attempts.
``sanitize_snippet`` strips the most common attack vectors before the text
is handed to the LLM, on top of the boundary-marker isolation in
``tools/web_search.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Bumped from 5 → 8 so the LLM has multiple sources to cross-reference; lone
# top hits are often a single low-quality blog and produce off-topic answers.
_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "8"))

# DDG region/language code.  Format: "<country>-<lang>" lowercase.
# Common values: de-de, ru-ru, us-en, uk-en, fr-fr, wt-wt (worldwide).
# DDG also does its own server-side geoip, but the `ddgs` library only honours
# the explicit `region` parameter — so this env value is what actually wins.
_DEFAULT_REGION = os.environ.get("DDG_REGION", "de-de")

# Region code → primary language name (in English, for prompt clarity).
# Used by web_search.py to translate the user's query INTO the locale's
# language before hitting DDG: without this, a Russian-language query on
# region=de-de still pulls Russian sites (Moscow time, Russian news) because
# DDG's region parameter biases the source pool but does not override the
# textual relevance signal — and a Russian "what time is it now" textually
# matches Russian sites best.  Translating the query to German tilts BOTH
# the region bias AND the textual match toward German sources, fixing the
# Moscow-time-in-Berlin class of bug.
# Set the language to None to skip translation (worldwide / unknown locale).
_REGION_LANGUAGE: dict[str, str | None] = {
    "de-de": "German",
    "ru-ru": "Russian",
    "us-en": "English",
    "uk-en": "English",
    "ca-en": "English",
    "au-en": "English",
    "fr-fr": "French",
    "es-es": "Spanish",
    "it-it": "Italian",
    "pl-pl": "Polish",
    "nl-nl": "Dutch",
    "pt-pt": "Portuguese",
    "br-pt": "Portuguese",
    "jp-jp": "Japanese",
    "cn-zh": "Chinese",
    "wt-wt": None,
}

# --- Page fetching config ----------------------------------------------------
# Turn the whole feature off via env if a flaky network makes fetch unreliable.
_FETCH_PAGES = os.environ.get("WEB_SEARCH_FETCH_PAGES", "true").lower() == "true"
# How many URLs we fire off in parallel from the top of the DDG result list.
# More candidates → higher chance of fast hits even when some sites are slow,
# but also more network traffic and CPU on the parser threadpool.  10 is a
# good ceiling for most home / office connections.
_FETCH_PARALLELISM = int(os.environ.get("WEB_SEARCH_FETCH_PARALLELISM", "10"))
# Once this many fetches succeed (return non-empty content) we stop waiting
# for the rest and hand whatever we have to the summariser.  Saves time on
# the long-tail slow site.  3 successful pages is usually plenty to cross-
# reference a fact.
_FETCH_TARGET_SUCCESSES = int(os.environ.get("WEB_SEARCH_FETCH_TARGET", "5"))
# Hard wall-clock budget for the entire enrichment phase.  Even if we
# haven't hit TARGET_SUCCESSES yet, we cut over to summarisation when this
# elapses — better a quick partial answer than a stuck request.
_FETCH_DEADLINE_S = float(os.environ.get("WEB_SEARCH_FETCH_DEADLINE_S", "4"))
# Per-page truncation after HTML stripping.  ~3 KB ≈ 750 tokens — enough for
# a thorough factual answer, small enough that several pages fit comfortably
# in Gemma's window with room for the summarisation prompt.
_PAGE_MAX_CHARS = int(os.environ.get("WEB_SEARCH_PAGE_MAX_CHARS", "3000"))
# Per-page hard timeout.  Slow page falls back to snippet; one slow site
# can't block the whole search.
_PAGE_TIMEOUT_S = float(os.environ.get("WEB_SEARCH_PAGE_TIMEOUT_S", "5"))

# Legacy alias used elsewhere in the codebase / docker-compose env.
_FETCH_TOP_N = int(os.environ.get("WEB_SEARCH_FETCH_TOP_N", str(_FETCH_PARALLELISM)))

# Most sites 403 a missing/obvious-bot UA.  Pretend to be a regular browser.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# HTML tags whose text content is almost never the actual page content;
# used by the selectolax FALLBACK path when trafilatura can't extract a
# main block.  Trafilatura has its own (much smarter) heuristics and
# doesn't use this list.
_NOISE_TAGS = (
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "form", "button", "svg",
)

# Minimum length (chars) below which we consider trafilatura's extraction
# "didn't really find anything" and fall back to selectolax.  Pages with
# just a title and 1-2 nav links can otherwise pass through trafilatura
# with 50-char output that's worse than the DDG snippet.
_MIN_EXTRACTED_CHARS = 120


def current_region() -> str:
    """Return the DDG region used for searches (for logging/UI)."""
    return _DEFAULT_REGION


def current_locale_language() -> str | None:
    """Return the primary language NAME for the active region (e.g. "German"),
    or ``None`` if the locale doesn't have a sensible single language to
    translate into (worldwide, unknown).  Names are in English so they can
    be dropped straight into a translation prompt.
    """
    return _REGION_LANGUAGE.get(_DEFAULT_REGION)

# ── Sanitisation patterns ──────────────────────────────────────────────────

# URLs are useless for voice and are a common injection carrier (e.g.
# "visit http://evil.com/jailbreak" embedded in a snippet).
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Common prompt-injection delimiters / role markers.
_INJECTION_RE = re.compile(
    r"(<<<|>>>|<\|im_start\|>|<\|im_end\|>|<\|system\|>"
    r"|\bSYSTEM\s*:|\bUSER\s*:|\bASSISTANT\s*:"
    r"|\bIGNORE\s+(?:ALL\s+)?(?:PREVIOUS|PRIOR)\b"
    r"|\bDISREGARD\s+(?:ALL\s+)?(?:PREVIOUS|PRIOR)\b"
    r"|#{3,}|={3,})",
    re.IGNORECASE,
)


def sanitize_snippet(text: str, max_chars: int = 300) -> str:
    """
    Strip user-hostile content from a search snippet before feeding it to
    the LLM.  This is a best-effort defence against indirect prompt injection
    via DDG results — not a cryptographic guarantee.

    Steps:
      1. Drop URLs (useless aloud, common injection vector).
      2. Strip known injection delimiter patterns.
      3. Collapse excess whitespace.
      4. Trim to ``max_chars``.
    """
    text = _URL_RE.sub("", text)
    text = _INJECTION_RE.sub("", text)
    text = " ".join(text.split())  # collapse whitespace
    return text[:max_chars].strip()


# ── DDG search ─────────────────────────────────────────────────────────────


async def ddg_search(
    query: str,
    max_results: int = _MAX_RESULTS,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Run a DuckDuckGo text search; return up to ``max_results`` results.

    ``region`` is the DDG country-language code (``de-de``, ``ru-ru``,
    ``us-en``, …).  When omitted, falls back to the ``DDG_REGION`` env
    value.  Pass ``"wt-wt"`` to disable localisation for a specific query.

    Each result dict has keys: ``title``, ``href``, ``body``. Returns an empty
    list on any error (rate limit, network issue, etc.) so the caller can
    decide how to degrade gracefully.
    """
    effective_region = region or _DEFAULT_REGION

    def _sync() -> list[dict[str, Any]]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(
                ddgs.text(query, region=effective_region, max_results=max_results)
            )

    loop = asyncio.get_running_loop()
    try:
        log.info(
            "ddg_search query=%r region=%s max=%d",
            query, effective_region, max_results,
        )
        return await loop.run_in_executor(None, _sync)
    except Exception:
        log.exception("DuckDuckGo search failed for query %r", query)
        return []


# ---------------------------------------------------------------------------
# Page fetching — parallel HTTP GET + HTML→text for the top N results
# ---------------------------------------------------------------------------


def _extract_main_content(html: str, *, max_chars: int) -> tuple[str, str]:
    """
    Sync HTML → main-content plain text.  Returns ``(text, extractor)``
    where ``extractor`` is ``"trafilatura"`` / ``"selectolax"`` / ``""``
    (the last for total failure) — useful for logging extractor coverage.

    Pure / no I/O: safe to call from ``asyncio.to_thread``.
    """
    # ── Primary: trafilatura — identifies the article/main block ─────────
    try:
        import trafilatura  # type: ignore[import]

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,     # weather grids, sports scores live here
            include_images=False,
            include_links=False,
            output_format="txt",
            # Default settings already filter ads/nav/footer/sidebar/cookie
            # banners.  Don't set favor_precision — we'd rather a bit of
            # extra prose than miss the actual answer to the question.
        )
        if text and len(text) >= _MIN_EXTRACTED_CHARS:
            text = " ".join(text.split())
            return text[:max_chars], "trafilatura"
    except Exception as exc:
        log.info(
            "trafilatura.extract failed (%s) — falling back to selectolax",
            exc.__class__.__name__,
        )

    # ── Fallback: selectolax — strip noise tags, take whatever's in <body>
    try:
        from selectolax.parser import HTMLParser  # type: ignore[import]

        tree = HTMLParser(html)
        for sel in _NOISE_TAGS:
            for node in tree.css(sel):
                node.decompose()
        body = tree.body
        if body is None:
            return "", ""
        text = body.text(separator=" ", strip=True)
        text = " ".join(text.split())
        if not text:
            return "", ""
        return text[:max_chars], "selectolax"
    except Exception as exc:
        log.warning("selectolax fallback failed (%s)", exc.__class__.__name__)
        return "", ""


async def _fetch_page_text(
    url: str,
    *,
    timeout: float = _PAGE_TIMEOUT_S,
    max_chars: int = _PAGE_MAX_CHARS,
) -> str:
    """Fetch ``url``, return cleaned plain text up to ``max_chars``.

    Returns "" on any failure (timeout, non-200, parse error, empty
    extraction).  The caller falls back to the DDG snippet in that case —
    one slow or hostile site never blocks the rest of the search.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_UA},
        ) as client:
            r = await client.get(url)
    except Exception as exc:
        log.info("fetch_page_text: %s failed (%s)", url, exc.__class__.__name__)
        return ""

    if r.status_code != 200:
        log.info("fetch_page_text: %s → HTTP %d", url, r.status_code)
        return ""
    # Skip non-HTML responses (PDFs, images served at the URL, etc.).
    ctype = r.headers.get("content-type", "").lower()
    if "html" not in ctype and "xml" not in ctype:
        log.info("fetch_page_text: %s skipping content-type=%s", url, ctype)
        return ""

    # Parse in a thread: trafilatura/selectolax are CPU-bound (50-500 ms
    # each); without to_thread they'd serialise on the loop instead of the
    # ``gather`` call running them in parallel.
    text, extractor = await asyncio.to_thread(
        _extract_main_content, r.text, max_chars=max_chars
    )
    log.info(
        "fetch_page_text: %s → %d chars via %s",
        url, len(text), extractor or "no_extraction",
    )
    return text


async def enrich_with_pages(
    results: list[dict[str, Any]],
    *,
    parallelism: int = _FETCH_PARALLELISM,
    target_successes: int = _FETCH_TARGET_SUCCESSES,
    deadline_s: float = _FETCH_DEADLINE_S,
) -> list[dict[str, Any]]:
    """Fire N URLs in parallel, return as soon as we have enough — or when
    the wall-clock deadline elapses.

    Strategy:
      • Fire `parallelism` (default 10) fetches concurrently from the top
        of the result list.
      • As each one completes, check: do we now have `target_successes`
        (default 5) non-empty page texts?  If yes, cancel the rest.
      • If `deadline_s` (default 4 s) wall-clock elapses first, take
        whatever we have.
      • Fetches that haven't completed yet (slow sites, unresponsive
        servers) are cancelled — their results were not going to arrive
        in time to be useful anyway.

    The remaining ``results`` entries (those past `parallelism`, plus
    cancelled ones) are left snippet-only.  The summariser sees a mix of
    `[page]` and `[snippet]` content, weighted in its prompt to prefer
    page content where available.
    """
    if not _FETCH_PAGES or not results:
        return results
    top = results[: parallelism]
    log.info(
        "enrich_with_pages: firing %d fetches, target=%d successes, deadline=%.1fs",
        len(top), target_successes, deadline_s,
    )

    # Map each task → index in `top` so we can attach the result back.
    tasks: dict[asyncio.Task, int] = {
        asyncio.create_task(_fetch_page_text(r.get("href", ""))): i
        for i, r in enumerate(top)
    }
    successes = 0
    elapsed_start = asyncio.get_event_loop().time()

    pending = set(tasks.keys())
    try:
        while pending and successes < target_successes:
            time_left = deadline_s - (asyncio.get_event_loop().time() - elapsed_start)
            if time_left <= 0:
                log.info(
                    "enrich_with_pages: hit %.1fs deadline with %d/%d successes — "
                    "stopping fetch, summarising on what's available",
                    deadline_s, successes, target_successes,
                )
                break
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=time_left,
            )
            for t in done:
                idx = tasks[t]
                try:
                    text = t.result()
                except Exception as exc:
                    log.info(
                        "enrich_with_pages: task %d crashed (%s)",
                        idx, exc.__class__.__name__,
                    )
                    continue
                if text:
                    top[idx]["page_text"] = text
                    successes += 1
    finally:
        # Cancel any still-pending fetches — we're moving on.
        for t in pending:
            t.cancel()
        # Drain cancellations so the event loop doesn't warn.
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    log.info(
        "enrich_with_pages: %d/%d successful in %.2fs (fired %d, %d cancelled)",
        successes,
        target_successes,
        asyncio.get_event_loop().time() - elapsed_start,
        len(top),
        len(pending),
    )
    return results


def format_results(results: list[dict[str, Any]]) -> str:
    """
    Format sanitised search results as a plain-text block for LLM context.

    For each result we prefer the full fetched ``page_text`` (typically a
    few KB of real page content) over the short DDG ``body`` snippet (200-
    300 chars).  Both are run through ``sanitize_snippet`` first to strip
    URLs, prompt-injection delimiters and excess whitespace.
    """
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = sanitize_snippet(r.get("title", ""), max_chars=120)
        # page_text > body: only set by enrich_with_pages on top N results.
        page_text = r.get("page_text") or ""
        if page_text:
            content = sanitize_snippet(page_text, max_chars=_PAGE_MAX_CHARS)
            marker = "[page]"
        else:
            content = sanitize_snippet(r.get("body", ""), max_chars=300)
            marker = "[snippet]"
        lines.append(f"{i}. {marker} {title}\n   {content}")
    return "\n\n".join(lines)
