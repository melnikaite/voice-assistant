"""
Item ingest — turn raw user input into stored items.

Each ingest function:
  1. Calls storage.items.create_item() to mint the row (immediately visible).
  2. Fires background async tasks for the slow parts: embedding, LLM summary,
     vision captioning.  These complete in ~1-2 s and populate the row without
     blocking the voice response.

Kinds handled:
  text        — plain text snippet or note
  link        — web URL; fetches title + snippet via httpx
  video       — YouTube / other video URL; extracts metadata + caption snippet
  short       — same pipeline as video (YouTube Shorts, TikTok, etc.)
  screenshot  — image bytes; runs vision LLM for a textual description
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from urllib.parse import urlparse

from .. import llm_utils, memory

log = logging.getLogger(__name__)


# ── Text ─────────────────────────────────────────────────────────────────

async def ingest_text(
    *,
    owner_profile_id: int,
    created_by_profile_id: int,
    category_id: int | None,
    body: str,
    title: str | None = None,
) -> int:
    """Store a plain text snippet.

    Fires embedding in the background so the item becomes searchable ~1 s
    after this returns.  Returns the new item id.
    """
    from ..storage.items import create_item

    item_id = await create_item(
        owner_profile_id=owner_profile_id,
        created_by_profile_id=created_by_profile_id,
        category_id=category_id,
        kind="text",
        title=title,
        body=body,
    )
    log.info("ingest_text: created item %d", item_id)
    asyncio.create_task(_embed_and_store(item_id, body))
    return item_id


# ── Link ─────────────────────────────────────────────────────────────────

async def ingest_link(
    *,
    owner_profile_id: int,
    created_by_profile_id: int,
    category_id: int | None,
    url: str,
    title: str | None = None,
) -> int:
    """Fetch URL metadata, store a link item.

    Background tasks: fetch title + og:description from the URL, generate
    embedding, optionally generate a short LLM summary.  The row is
    visible immediately with whatever the caller supplied; metadata fills
    in asynchronously.
    """
    from ..storage.items import create_item

    item_id = await create_item(
        owner_profile_id=owner_profile_id,
        created_by_profile_id=created_by_profile_id,
        category_id=category_id,
        kind="link",
        title=title,
        url=url,
    )
    log.info("ingest_link: created item %d for %s", item_id, url)
    asyncio.create_task(_fetch_link_metadata(item_id, url, title, owner_profile_id))
    return item_id


async def _fetch_link_metadata(
    item_id: int,
    url: str,
    caller_title: str | None,
    owner_profile_id: int,
) -> None:
    """Background: fetch URL, extract title + snippet, embed."""
    try:
        import httpx
        from ..storage.items import set_item_summary, update_item

        try:
            async with httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; VoiceAssistant/1.0)"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            log.warning("ingest_link: failed to fetch %s: %s", url, exc)
            return

        # Extract <title> tag.
        fetched_title: str | None = None
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if m:
            fetched_title = m.group(1).strip()

        # Extract og:description or fall back to first 500 chars of text.
        snippet: str = ""
        og_m = re.search(
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if not og_m:
            # Also try reversed attribute order.
            og_m = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description["\']',
                html,
                re.IGNORECASE,
            )
        if og_m:
            snippet = og_m.group(1).strip()
        else:
            # Strip tags and collapse whitespace for a plain-text preview.
            text_only = re.sub(r"<[^>]+>", " ", html)
            text_only = re.sub(r"\s+", " ", text_only).strip()
            snippet = text_only[:500]

        snippet = snippet[:200]

        if snippet:
            await set_item_summary(item_id, snippet)

        # Prefer fetched title over caller-supplied when better.
        if fetched_title and not caller_title:
            await update_item(item_id, owner_profile_id, title=fetched_title)

        embed_text = f"{fetched_title or caller_title or ''} {snippet}".strip()
        if embed_text:
            await _embed_and_store(item_id, embed_text)

        log.info("ingest_link: metadata enriched item %d", item_id)
    except NotImplementedError as exc:
        log.warning("ingest_link: background task incomplete import: %s", exc)
    except Exception as exc:
        log.warning("ingest_link: background metadata fetch failed for item %d: %s", item_id, exc)


# ── Video / Short ─────────────────────────────────────────────────────────

async def ingest_video(
    *,
    owner_profile_id: int,
    created_by_profile_id: int,
    category_id: int | None,
    url: str,
    kind: str = "video",      # 'video' | 'short'
    title: str | None = None,
) -> int:
    """Extract video metadata and store a video item.

    Background tasks: resolve video title via oEmbed (no download —
    metadata only), generate a short summary from title + description,
    embed.  Transcript extraction is intentionally skipped for videos
    because full transcripts introduce too much noise into semantic
    search for household use.
    """
    from ..storage.items import create_item

    item_id = await create_item(
        owner_profile_id=owner_profile_id,
        created_by_profile_id=created_by_profile_id,
        category_id=category_id,
        kind=kind,
        title=title,
        url=url,
    )
    log.info("ingest_video: created item %d for %s", item_id, url)
    asyncio.create_task(_fetch_video_metadata(item_id, url, title, owner_profile_id))
    return item_id


async def _fetch_video_metadata(
    item_id: int,
    url: str,
    caller_title: str | None,
    owner_profile_id: int,
) -> None:
    """Background: resolve video title via oEmbed or URL domain fallback."""
    try:
        import httpx
        from ..storage.items import update_item

        resolved_title: str | None = None

        # Detect YouTube URLs (youtube.com/watch, youtu.be, shorts).
        is_youtube = bool(
            re.search(r"(youtube\.com|youtu\.be)", url, re.IGNORECASE)
        )

        if is_youtube:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            try:
                async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as client:
                    resp = await client.get(oembed_url)
                    resp.raise_for_status()
                    data = resp.json()
                    resolved_title = data.get("title") or None
            except Exception as exc:
                log.warning("ingest_video: oEmbed failed for %s: %s", url, exc)

        if not resolved_title:
            # Fallback: use domain + path as a human-readable hint.
            try:
                parsed = urlparse(url)
                path = parsed.path.rstrip("/") or "/"
                resolved_title = f"{parsed.netloc}{path}"
            except Exception:
                resolved_title = url

        # Only update title when the caller didn't supply one and we found something better.
        if resolved_title and not caller_title:
            await update_item(item_id, owner_profile_id, title=resolved_title)

        embed_text = resolved_title or caller_title or url
        await _embed_and_store(item_id, embed_text)

        log.info("ingest_video: metadata enriched item %d", item_id)
    except NotImplementedError as exc:
        log.warning("ingest_video: background task incomplete import: %s", exc)
    except Exception as exc:
        log.warning("ingest_video: background metadata fetch failed for item %d: %s", item_id, exc)


# ── Screenshot ────────────────────────────────────────────────────────────

async def ingest_screenshot(
    *,
    owner_profile_id: int,
    created_by_profile_id: int,
    category_id: int | None,
    image_bytes: bytes,
    mime_type: str = "image/png",
    title: str | None = None,
) -> int:
    """Write a screenshot to disk and create an item row.

    Background tasks: run the vision LLM to generate a textual caption
    (stored as summary), then embed that caption.  The vision call uses
    the existing LLM_VISION_URL / LLM_VISION_MODEL env vars (same as
    look_at_screen).

    File is written to ITEMS_DIR/<id>.<ext> — set via set_item_media_path
    after the id is known.
    """
    from ..storage.items import ITEMS_DIR, _ensure_dir, create_item, set_item_media_path

    _ensure_dir()

    item_id = await create_item(
        owner_profile_id=owner_profile_id,
        created_by_profile_id=created_by_profile_id,
        category_id=category_id,
        kind="screenshot",
        title=title,
    )

    ext = ".jpg" if mime_type == "image/jpeg" else ".png"
    filename = f"{item_id}{ext}"
    file_path = ITEMS_DIR / filename

    def _write_file() -> None:
        file_path.write_bytes(image_bytes)

    await asyncio.to_thread(_write_file)
    await set_item_media_path(item_id, filename)

    log.info("ingest_screenshot: created item %d, file=%s", item_id, filename)
    asyncio.create_task(_caption_screenshot(item_id, image_bytes))
    return item_id


async def _caption_screenshot(item_id: int, image_bytes: bytes) -> None:
    """Background: run vision LLM on image bytes, store caption as summary."""
    try:
        from ..storage.items import set_item_summary

        media_type = "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            media_type = "image/jpeg"
        data_url = (
            "data:" + media_type + ";base64,"
            + base64.standard_b64encode(image_bytes).decode("ascii")
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual assistant.  Describe the contents of the "
                    "screenshot in 1-3 concise sentences.  Focus on the most "
                    "important information visible.  Plain text only — no markdown."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Describe this screenshot."},
                ],
            },
        ]
        choice = await llm_utils.chat(
            messages,
            temperature=0.2,
            reasoning_effort="low",
            endpoint_url=llm_utils.LLM_VISION_URL,
            model=llm_utils.LLM_VISION_MODEL,
            tool_name="ingest_screenshot",
        )
        msg = choice.get("message") or {}
        caption = llm_utils.extract_text(msg).strip()
        if caption:
            await set_item_summary(item_id, caption)
            await _embed_and_store(item_id, caption)
            log.info("ingest_screenshot: captioned item %d", item_id)
        else:
            log.warning("ingest_screenshot: empty caption from vision LLM for item %d", item_id)
    except NotImplementedError as exc:
        log.warning("ingest_screenshot: background task incomplete import: %s", exc)
    except Exception as exc:
        log.warning("ingest_screenshot: vision captioning failed for item %d: %s", item_id, exc)


# ── Shared background helpers ─────────────────────────────────────────────

async def _embed_and_store(item_id: int, text: str) -> None:
    """Generate a fastembed vector for text and call set_item_embedding.

    Reuses the same model instance as semantic memory (app.memory).
    Runs in a background asyncio task — never awaited by the ingest path.
    """
    try:
        from ..storage.items import set_item_embedding

        vec = await memory.embed_passage(text)
        blob = memory.encode(vec)
        await set_item_embedding(item_id, blob)
        log.info("_embed_and_store: stored embedding for item %d", item_id)
    except NotImplementedError as exc:
        log.warning("_embed_and_store: incomplete import for item %d: %s", item_id, exc)
    except Exception as exc:
        log.warning("_embed_and_store: failed to embed item %d: %s", item_id, exc)


async def _summarise_and_store(item_id: int, text: str, *, max_words: int = 40) -> None:
    """Ask the LLM for a short summary and call set_item_summary.

    Uses 'low' reasoning_effort so it's cheap.  Only fires for link/video
    kinds where a snippet is available but would benefit from compression.
    """
    try:
        from ..storage.items import set_item_summary

        messages = [
            {
                "role": "user",
                "content": f"Summarise in {max_words} words or fewer: {text[:2000]}",
            }
        ]
        choice = await llm_utils.chat(
            messages,
            reasoning_effort="low",
            tool_name="ingest_summarise",
        )
        msg = choice.get("message") or {}
        summary = llm_utils.extract_text(msg).strip()
        if summary:
            await set_item_summary(item_id, summary)
            log.info("_summarise_and_store: stored summary for item %d", item_id)
        else:
            log.warning("_summarise_and_store: empty summary from LLM for item %d", item_id)
    except NotImplementedError as exc:
        log.warning("_summarise_and_store: incomplete import for item %d: %s", item_id, exc)
    except Exception as exc:
        log.warning("_summarise_and_store: failed to summarise item %d: %s", item_id, exc)
