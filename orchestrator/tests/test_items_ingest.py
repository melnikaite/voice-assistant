"""
Tests for orchestrator/app/items/ingest.py.

Strategy:
  * Uses an in-memory SQLite DB path via the conftest _fresh_db fixture
    (DB_PATH is overridden to a /tmp file).
  * Background tasks (embed, summarise, vision) are either mocked out or
    allowed to fail gracefully — tests should not need a live LLM.
  * Each test verifies that the item row is created correctly and that
    the immediate (synchronous) state is correct.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.items.ingest import (
    _embed_and_store,
    _summarise_and_store,
    ingest_link,
    ingest_screenshot,
    ingest_text,
    ingest_video,
)
from app.storage import get_item, init_schema


# ── Helpers ───────────────────────────────────────────────────────────────

_OWNER = 1
_CREATOR = 1
_CAT = None  # uncategorised


def _close_coro(coro):
    """side_effect for patched ``asyncio.create_task``.

    The patched call still receives the real coroutine object (because
    ``_embed_and_store(...)`` was evaluated before the patched
    create_task was invoked), so we close it explicitly to avoid
    "coroutine was never awaited" RuntimeWarnings at GC time.  Returns a
    MagicMock so any code that expects a Task back keeps working.
    """
    coro.close()
    return MagicMock()


# ── Text ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_text_creates_row():
    """ingest_text mints a row with kind='text' and the supplied body."""
    # Suppress background embedding task — no fastembed in test env.
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="Buy oat milk and coffee",
            title="Shopping note",
        )

    assert isinstance(item_id, int)
    assert item_id > 0

    row = await get_item(item_id)
    assert row is not None
    assert row["kind"] == "text"
    assert row["body"] == "Buy oat milk and coffee"
    assert row["title"] == "Shopping note"
    assert row["status"] == "active"
    assert row["owner_profile_id"] == _OWNER
    assert row["deleted_at"] is None


@pytest.mark.asyncio
async def test_ingest_text_no_title():
    """ingest_text works fine when title is not supplied."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="Just a note",
        )
    row = await get_item(item_id)
    assert row is not None
    assert row["title"] is None
    assert row["body"] == "Just a note"


@pytest.mark.asyncio
async def test_ingest_text_fires_background_embed():
    """ingest_text must fire exactly one background embed task."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro) as mock_create_task:
        await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="Test",
        )
    mock_create_task.assert_called_once()
    # The task coroutine passed to create_task should be for _embed_and_store.
    coro = mock_create_task.call_args[0][0]
    assert coro.__name__ == "_embed_and_store"
    coro.close()  # Clean up the un-awaited coroutine.


# ── Link ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_link_creates_row():
    """ingest_link creates a row with kind='link' and url set."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_link(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            url="https://example.com/article",
            title="Example Article",
        )

    assert isinstance(item_id, int)
    row = await get_item(item_id)
    assert row is not None
    assert row["kind"] == "link"
    assert row["url"] == "https://example.com/article"
    assert row["title"] == "Example Article"
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_ingest_link_no_title():
    """ingest_link works without a caller-supplied title."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_link(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            url="https://example.com/page",
        )
    row = await get_item(item_id)
    assert row is not None
    assert row["url"] == "https://example.com/page"
    assert row["title"] is None


@pytest.mark.asyncio
async def test_ingest_link_fires_background_task():
    """ingest_link must fire a background metadata-fetch task."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro) as mock_create_task:
        await ingest_link(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            url="https://example.com",
        )
    mock_create_task.assert_called_once()
    coro = mock_create_task.call_args[0][0]
    assert coro.__name__ == "_fetch_link_metadata"
    coro.close()


# ── Video ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_video_creates_row():
    """ingest_video creates a row with kind='video' and url set."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_video(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Rick Astley",
        )

    assert isinstance(item_id, int)
    row = await get_item(item_id)
    assert row is not None
    assert row["kind"] == "video"
    assert "youtube.com" in row["url"]
    assert row["title"] == "Rick Astley"
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_ingest_video_short_kind():
    """ingest_video accepts kind='short' for YouTube Shorts / TikTok."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_video(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            url="https://www.youtube.com/shorts/abc123",
            kind="short",
        )
    row = await get_item(item_id)
    assert row is not None
    assert row["kind"] == "short"


@pytest.mark.asyncio
async def test_ingest_video_fires_background_task():
    """ingest_video must fire a background metadata-fetch task."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro) as mock_create_task:
        await ingest_video(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            url="https://www.youtube.com/watch?v=test",
        )
    mock_create_task.assert_called_once()
    coro = mock_create_task.call_args[0][0]
    assert coro.__name__ == "_fetch_video_metadata"
    coro.close()


# ── Screenshot ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_screenshot_creates_row(tmp_path, monkeypatch):
    """ingest_screenshot writes the file and creates an item row."""
    # Redirect ITEMS_DIR to tmp_path so we don't need /data.
    import app.items.ingest as ingest_mod
    import app.storage.items as items_mod

    monkeypatch.setattr(items_mod, "ITEMS_DIR", tmp_path)

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20  # minimal PNG magic

    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_screenshot(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            image_bytes=fake_png,
            mime_type="image/png",
            title="My screenshot",
        )

    assert isinstance(item_id, int)
    row = await get_item(item_id)
    assert row is not None
    assert row["kind"] == "screenshot"
    assert row["title"] == "My screenshot"
    assert row["media_path"] == f"{item_id}.png"

    written = tmp_path / f"{item_id}.png"
    assert written.exists()
    assert written.read_bytes() == fake_png


@pytest.mark.asyncio
async def test_ingest_screenshot_jpeg_extension(tmp_path, monkeypatch):
    """JPEG mime_type produces a .jpg filename."""
    import app.storage.items as items_mod

    monkeypatch.setattr(items_mod, "ITEMS_DIR", tmp_path)

    fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 20

    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_screenshot(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            image_bytes=fake_jpeg,
            mime_type="image/jpeg",
        )

    row = await get_item(item_id)
    assert row is not None
    assert row["media_path"] == f"{item_id}.jpg"
    assert (tmp_path / f"{item_id}.jpg").exists()


# ── _embed_and_store ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_embed_and_store_updates_embedding():
    """_embed_and_store stores the encoded float32 blob in the DB."""
    import numpy as np

    fake_vec = np.array([0.1, 0.2, 0.3], dtype="float32")
    fake_blob = fake_vec.tobytes()

    # Create an item to attach the embedding to.
    from app.storage.items import create_item, get_item as _get_item

    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="embed me",
        )

    # Confirm no embedding yet.
    row = await _get_item(item_id)
    assert row["embedding"] is None

    # Mock memory module: embed_passage returns fake_vec, encode returns fake_blob.
    mock_memory = MagicMock()
    mock_memory.embed_passage = AsyncMock(return_value=fake_vec)
    mock_memory.encode = MagicMock(return_value=fake_blob)

    with patch("app.items.ingest.memory", mock_memory):
        await _embed_and_store(item_id, "embed me")

    row = await _get_item(item_id)
    assert row["embedding"] == fake_blob
    assert len(row["embedding"]) == len(fake_blob)


@pytest.mark.asyncio
async def test_embed_and_store_swallows_exceptions():
    """_embed_and_store never raises — it logs warnings on failure."""
    # Create a row to target.
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="crash test",
        )

    mock_memory = MagicMock()
    mock_memory.embed_passage = AsyncMock(side_effect=RuntimeError("model offline"))

    with patch("app.items.ingest.memory", mock_memory):
        # Should not raise.
        await _embed_and_store(item_id, "crash test")

    # Row should still exist and embedding remain None.
    row = await get_item(item_id)
    assert row is not None
    assert row["embedding"] is None


# ── _summarise_and_store ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarise_and_store_updates_summary():
    """_summarise_and_store writes the LLM-generated summary to the DB."""
    # Create a target row.
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="Long article text that needs summarising",
        )

    fake_choice = {"message": {"content": "A concise summary.", "role": "assistant"}}

    mock_llm_utils = MagicMock()
    mock_llm_utils.chat = AsyncMock(return_value=fake_choice)
    mock_llm_utils.extract_text = MagicMock(return_value="A concise summary.")

    with patch("app.items.ingest.llm_utils", mock_llm_utils):
        await _summarise_and_store(item_id, "Long article text that needs summarising")

    row = await get_item(item_id)
    assert row is not None
    assert row["summary"] == "A concise summary."


@pytest.mark.asyncio
async def test_summarise_and_store_swallows_exceptions():
    """_summarise_and_store never raises — logs warning on failure."""
    with patch("app.items.ingest.asyncio.create_task", side_effect=_close_coro):
        item_id = await ingest_text(
            owner_profile_id=_OWNER,
            created_by_profile_id=_CREATOR,
            category_id=_CAT,
            body="fail target",
        )

    mock_llm_utils = MagicMock()
    mock_llm_utils.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

    with patch("app.items.ingest.llm_utils", mock_llm_utils):
        # Should not raise.
        await _summarise_and_store(item_id, "fail target")

    row = await get_item(item_id)
    assert row is not None
    assert row["summary"] is None
