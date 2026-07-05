"""
summarize_url tool (#54).

User says "summarise this" / "make a summary" or pastes a URL into the
UI.  The tool detects whether the URL points to a page, a single media
file, or a playlist, then routes to the appropriate pipeline:

  • HTML / text page  → trafilatura extraction → LLM summary (streamed)
  • Single media      → yt-dlp audio → Whisper → LLM summary
    – Short media  (< ASYNC_THRESHOLD_S): streams to TTS like web_search
    – Long  media  (≥ ASYNC_THRESHOLD_S): immediate ack + background job
                      → notify via Web Push when ready
  • Multiple media    → voice disambiguation, default to first item

All LLM summarization uses map-reduce when the transcript is longer than
SINGLE_PASS_CHAR_LIMIT, so even a 2-hour podcast gets a sensible answer.
"""
from __future__ import annotations

import logging

from ..i18n import t
from ..search import _fetch_page_text  # trafilatura-based text extraction
from ..url_summarizer import (
    ASYNC_THRESHOLD_S,
    ProbeResult,
    _background_summarize,
    download_and_transcribe,
    probe_url,
    summarize_text,
)
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


@tool(
    name="summarize_url",
    description=(
        "Summarise the content of a URL: web page, YouTube video, podcast, "
        "or any audio/video supported by yt-dlp.  Use when the user shares "
        "a link and asks for a summary, says 'summarise this', 'what is this "
        "about', 'give me the gist of <url>', or similar.  "
        "For SHORT content (< 3 min): streams the summary aloud immediately.  "
        "For LONG media (≥ 3 min): acknowledges and summarises in the background, "
        "notifying when ready via a push notification or spoken reply.  "
        "Do NOT use this for web searches — use `web_search` instead."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The URL to summarise.  Must be a valid http/https URL "
                    "provided by the user — never fabricate one."
                ),
            },
            "item_index": {
                "type": "integer",
                "description": (
                    "When the URL contains multiple media items (e.g. a playlist "
                    "page), which item to summarise (0-based).  Omit to let the "
                    "tool pick the most relevant one (usually item 0)."
                ),
            },
        },
        "required": ["url"],
    },
    risk="read",
    tier="system",
)
async def summarize_url(
    url: str,
    item_index: int | None = None,
    *,
    ctx=None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang

    # ── 1. Validate URL ───────────────────────────────────────────────────
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return ToolResult(
            text=t("summarize_url.bad_url", lang),
            data={"error": "bad_url", "url": url},
        )

    # ── 2. Probe for media ───────────────────────────────────────────────
    await cx.progress("probe", url[:80])
    probe: ProbeResult = await probe_url(url)
    log.info("summarize_url: probe %r → kind=%s entries=%d", url, probe.kind, len(probe.entries))

    # ── 3. Branch ────────────────────────────────────────────────────────

    # ── 3a. Plain page (no media found) ──────────────────────────────────
    if probe.kind == "page" or not probe.entries:
        await cx.progress("fetch")
        text = await _fetch_page_text(url)
        if not text or not text.strip():
            return ToolResult(
                text=t("summarize_url.no_content", lang),
                data={"error": "no_content", "url": url},
            )
        await cx.progress("summarize")
        summary = await summarize_text(
            text,
            title=url,
            user_lang=lang,
            stream_sink=cx.stream_sink,
            client_id=cx.client_id,
        )
        return ToolResult(
            text=summary or t("summarize_url.empty_summary", lang),
            data={"url": url, "kind": "page"},
        )

    # ── 3b. Multiple media → pick one ────────────────────────────────────
    if probe.kind == "multi" and item_index is None:
        # Default to item 0 (most recent / first in playlist).
        # Tell the user we picked it and how to request a different one.
        entry = probe.entries[0]
        total = len(probe.entries)
        note = (
            f" (picked the first of {total} items; "
            f"say 'summarise item 2' to choose a different one)"
            if total > 1 else ""
        )
        log.info("summarize_url: multi → defaulting to entry 0 of %d", total)
        # Fall through to single-media handling with the selected entry.
    elif probe.kind == "multi" and item_index is not None:
        idx = max(0, min(item_index, len(probe.entries) - 1))
        entry = probe.entries[idx]
        note = ""
    else:
        entry = probe.entries[0]
        note = ""

    # ── 3c. Single media — short path (sync/streaming) ───────────────────
    duration = entry.duration_s
    if duration is None or duration < ASYNC_THRESHOLD_S:
        await cx.progress("download", entry.title or entry.url[:80])
        transcript = await download_and_transcribe(
            entry,
            progress=cx.progress,
            client_id=cx.client_id,
        )
        if not transcript:
            return ToolResult(
                text=t("summarize_url.transcribe_failed", lang),
                data={"error": "transcribe_failed", "url": entry.url},
            )
        await cx.progress("summarize")
        summary = await summarize_text(
            transcript,
            title=entry.title,
            user_lang=lang,
            stream_sink=cx.stream_sink,
            client_id=cx.client_id,
        )
        return ToolResult(
            text=(summary or t("summarize_url.empty_summary", lang)) + note,
            data={
                "url": entry.url,
                "kind": "media",
                "title": entry.title,
                "duration_s": duration,
            },
        )

    # ── 3d. Long media — async path ───────────────────────────────────────
    # Kick off the background job and return an immediate ack.
    import asyncio as _asyncio
    _asyncio.create_task(
        _background_summarize(
            entry,
            client_id=cx.client_id,
            profile_id=cx.profile_id,
            user_lang=lang,
        )
    )
    title_str = f'"{entry.title}"' if entry.title else "that media"
    mins = int((duration or 0) / 60)
    ack = t(
        "summarize_url.async_started",
        lang,
        title=title_str,
        duration=f"{mins} min" if mins else "a while",
    )
    return ToolResult(
        text=ack,
        data={
            "url": entry.url,
            "kind": "media_async",
            "title": entry.title,
            "duration_s": duration,
        },
    )
