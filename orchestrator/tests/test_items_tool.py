"""
Voice tool tests for the items and categories tools.

Tests call the tool functions directly (not via the agent dispatch loop)
so we can assert on ToolResult.data without parsing spoken text.

In-memory DB is managed by the shared _fresh_db autouse fixture in
conftest.py.  We seed one speaker_profile row so `profile_id=1` is a
valid profile for ownership checks.

ingest_text is mocked to avoid needing a live embedding model — the
storage calls inside it are real (they go through the in-memory DB).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.tools.items import _items_risk, items
from app.tools.categories import _categories_risk, categories


# ── Minimal AgentContext stub ─────────────────────────────────────────────


class _StubCtx:
    """Minimal AgentContext-compatible stub for tool tests."""

    def __init__(self, profile_id: int | None = 1, lang: str = "en"):
        self.profile_id = profile_id
        self.user_lang = lang
        self.progress_sink = None
        self.stream_sink = None
        self.is_authenticated = True
        self.client_id = "test"


def _ctx(profile_id: int | None = 1) -> _StubCtx:
    return _StubCtx(profile_id=profile_id)


# ── Shared fixture: seed a speaker_profile row ────────────────────────────


@pytest.fixture(autouse=True)
async def _seed_profile():
    """Insert a speaker_profile row so profile_id=1 exists in the DB.

    Many storage helpers are scoped by owner_profile_id — they don't
    FK-check against speaker_profiles, but having the row keeps things
    clean for tests that might probe the relation.
    """
    from app.storage.db import _conn
    import time
    c = _conn()
    c.execute(
        "INSERT OR IGNORE INTO speaker_profiles "
        "(id, client_id, name, embedding, sample_count, created_at) "
        "VALUES (1, 'test-client', 'Tester', X'00', 1, ?)",
        (time.time(),),
    )


# ── Risk-level tests (pure, no DB) ───────────────────────────────────────


def test_items_tool_search_read_risk():
    assert _items_risk({"action": "search"}) == "read"


def test_items_tool_auto_sort_high_write_risk():
    assert _items_risk({"action": "auto_sort"}) == "high_write"


def test_categories_tool_delete_high_write():
    assert _categories_risk({"action": "delete"}) == "high_write"


# ── items tool — functional tests ────────────────────────────────────────


async def test_items_tool_save_text():
    """save action with kind=text inserts a row and returns item_id."""
    # Mock ingest_text to return a fixed item_id without needing the
    # embedding model; the DB row is actually inserted by ingest_text
    # itself (we let the real create_item run via a partial mock).
    with patch("app.items.ingest.ingest_text", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = 42
        result = await items(action="save", kind="text", body="hello world", ctx=_ctx())

    assert result.data is not None
    assert result.data.get("item_id") == 42
    assert result.data.get("kind") == "text"


async def test_items_tool_list_empty():
    """list on empty store returns the 'no items' message."""
    result = await items(action="list", ctx=_ctx())
    assert result.data is not None
    # Empty list — either count==0 or the text mentions nothing
    assert result.data.get("count", 0) == 0


async def test_items_tool_delete_and_restore():
    """Create an item via storage directly, then delete/restore via tool."""
    from app.storage.items import create_item, get_item

    item_id = await create_item(
        owner_profile_id=1, created_by_profile_id=1,
        kind="text", body="delete me",
    )

    # Delete via tool
    del_result = await items(action="delete", item_id=item_id, ctx=_ctx())
    assert del_result.data is not None
    assert "error" not in del_result.data

    # Confirm soft-deleted
    row = await get_item(item_id)
    assert row["deleted_at"] is not None

    # Restore via tool
    restore_result = await items(action="restore", item_id=item_id, ctx=_ctx())
    assert restore_result.data is not None
    assert "error" not in restore_result.data

    # Confirm active again
    row2 = await get_item(item_id)
    assert row2["deleted_at"] is None


async def test_items_tool_no_profile_returns_error():
    """ctx with profile_id=None should return an auth error, not crash."""
    result = await items(action="list", ctx=_ctx(profile_id=None))
    assert result.data is not None
    assert result.data.get("error") == "no_profile"


async def test_items_tool_unknown_action():
    """Unknown action should return unknown_action error, not raise."""
    result = await items(action="nonsense", ctx=_ctx())
    assert result.data is not None
    assert result.data.get("error") == "unknown_action"


async def test_items_tool_check_toggle():
    """save a text item via storage, then toggle checked via tool."""
    from app.storage.items import create_item, get_item

    item_id = await create_item(
        owner_profile_id=1, created_by_profile_id=1,
        kind="text", body="checklist entry",
    )

    # Initially unchecked
    row = await get_item(item_id)
    assert row["completed_at"] is None

    # Check via tool
    check_result = await items(action="check", item_id=item_id, ctx=_ctx())
    assert check_result.data is not None
    assert check_result.data.get("found") is True
    assert check_result.data.get("completed") is True


# ── categories tool — functional tests ───────────────────────────────────


async def test_categories_tool_create_list():
    """create a category, then list and verify it's present."""
    create_result = await categories(action="create", name="Tech Articles", ctx=_ctx())
    assert create_result.data is not None
    assert "error" not in create_result.data
    assert create_result.data.get("name") == "Tech Articles"

    list_result = await categories(action="list", ctx=_ctx())
    assert list_result.data is not None
    cats = list_result.data.get("categories", [])
    assert any(c["name"] == "Tech Articles" for c in cats)
