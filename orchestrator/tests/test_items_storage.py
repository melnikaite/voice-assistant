"""
Storage-layer tests for the personal item store.

Covers storage/items.py and storage/categories.py.  Every test runs
against an in-memory SQLite DB via the shared _fresh_db fixture in
conftest.py (which calls close_thread_conn() + init_schema() before each
test and tears down after).

Note: conftest.py uses a real /tmp file (not :memory:) because
thread-local sqlite3 connections don't work with :memory: — each thread
would see its own empty DB.  The fixture handles creation/deletion.
"""
from __future__ import annotations

import time

import pytest

from app.storage.items import (
    create_item,
    delete_item,
    fts_search,
    get_item,
    list_items,
    move_item,
    purge_expired_trash,
    restore_item,
    toggle_checked,
    update_item,
)
from app.storage.categories import (
    create_category,
    delete_category,
    list_categories,
    list_subtree,
    rename_category,
    resolve_category_by_name,
    restore_category,
    share_category,
)
from app.storage.items import _SENTINEL


# ── Items ─────────────────────────────────────────────────────────────────


async def test_create_and_get_item():
    item_id = await create_item(
        owner_profile_id=1,
        created_by_profile_id=1,
        kind="text",
        title="My Note",
        body="Hello world",
    )
    assert isinstance(item_id, int) and item_id > 0

    row = await get_item(item_id)
    assert row is not None
    assert row["id"] == item_id
    assert row["owner_profile_id"] == 1
    assert row["kind"] == "text"
    assert row["title"] == "My Note"
    assert row["body"] == "Hello world"
    assert row["deleted_at"] is None


async def test_list_items_by_category():
    cat_a = await create_category(owner_profile_id=1, name="Alpha")
    cat_b = await create_category(owner_profile_id=1, name="Beta")

    id_a1 = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="a1", category_id=cat_a)
    id_a2 = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="a2", category_id=cat_a)
    id_b1 = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="b1", category_id=cat_b)

    alpha_items = await list_items(1, category_id=cat_a)
    alpha_ids = {r["id"] for r in alpha_items}
    assert id_a1 in alpha_ids
    assert id_a2 in alpha_ids
    assert id_b1 not in alpha_ids

    beta_items = await list_items(1, category_id=cat_b)
    assert {r["id"] for r in beta_items} == {id_b1}


async def test_list_items_deleted_only():
    item_id = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="to delete")

    # Normal list should include it
    normal = await list_items(1)
    assert any(r["id"] == item_id for r in normal)

    # Soft-delete it
    ok = await delete_item(item_id, 1)
    assert ok is True

    # Deleted item should NOT appear in normal list
    normal_after = await list_items(1)
    assert not any(r["id"] == item_id for r in normal_after)

    # Deleted item SHOULD appear in trash view
    trash = await list_items(1, deleted_only=True)
    assert any(r["id"] == item_id for r in trash)


async def test_update_item_partial():
    item_id = await create_item(
        owner_profile_id=1, created_by_profile_id=1,
        kind="text", title="Original", body="Keep this body",
    )

    # Update only the title; body should remain unchanged
    ok = await update_item(item_id, 1, title="Updated Title")
    assert ok is True

    row = await get_item(item_id)
    assert row["title"] == "Updated Title"
    assert row["body"] == "Keep this body"


async def test_move_item():
    cat_a = await create_category(owner_profile_id=1, name="Cat A")
    cat_b = await create_category(owner_profile_id=1, name="Cat B")
    item_id = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="move me", category_id=cat_a)

    ok = await move_item(item_id, 1, cat_b)
    assert ok is True

    row = await get_item(item_id)
    assert row["category_id"] == cat_b

    in_b = await list_items(1, category_id=cat_b)
    assert any(r["id"] == item_id for r in in_b)

    in_a = await list_items(1, category_id=cat_a)
    assert not any(r["id"] == item_id for r in in_a)


async def test_toggle_checked():
    item_id = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="checklist item")

    # Initially unchecked
    row = await get_item(item_id)
    assert row["completed_at"] is None

    # Check it
    result = await toggle_checked(item_id, 1)
    assert result["found"] is True
    assert result["completed"] is True
    assert result["completed_at"] is not None

    # Uncheck it
    result2 = await toggle_checked(item_id, 1)
    assert result2["found"] is True
    assert result2["completed"] is False
    assert result2["completed_at"] is None


async def test_purge_expired_trash():
    item_id = await create_item(owner_profile_id=1, created_by_profile_id=1, kind="text", body="old trash")

    # Soft-delete
    await delete_item(item_id, 1)

    # Manually backdate deleted_at to 10 days ago so it's past the 7-day threshold
    from app.storage.db import _conn
    c = _conn()
    c.execute("UPDATE items SET deleted_at = ? WHERE id = ?", (time.time() - 10 * 86400, item_id))

    purged = await purge_expired_trash(max_age_days=7)
    assert purged >= 1

    # Row should be gone
    row = await get_item(item_id)
    assert row is None


async def test_purge_expired_trash_preserves_recent():
    """Items deleted within the TTL window survive a GC pass.

    Regression guard against the off-by-one case where the cutoff
    compared with ``>`` instead of ``<`` and wiped fresh trash too.
    """
    recent_id = await create_item(
        owner_profile_id=1, created_by_profile_id=1, kind="text", body="recent trash",
    )
    await delete_item(recent_id, 1)
    # deleted_at is already "now" — well inside the 7-day window.

    purged = await purge_expired_trash(max_age_days=7)
    assert purged == 0

    # Still in trash (soft-deleted but not GC'd).
    row = await get_item(recent_id)
    assert row is not None
    assert row["deleted_at"] is not None


async def test_fts_search():
    item_id = await create_item(
        owner_profile_id=1, created_by_profile_id=1,
        kind="text", title="Unique phrase test", body="antidisestablishmentarianism",
    )

    results = await fts_search(1, "antidisestablishmentarianism")
    assert any(r["id"] == item_id for r in results)


# ── Categories ────────────────────────────────────────────────────────────


async def test_create_category_and_list():
    id_a = await create_category(owner_profile_id=1, name="Recipes")
    id_b = await create_category(owner_profile_id=1, name="Travel")

    cats = await list_categories(1)
    names = {c["name"] for c in cats}
    assert "Recipes" in names
    assert "Travel" in names
    ids = {c["id"] for c in cats}
    assert id_a in ids and id_b in ids


async def test_subtree_query():
    parent_id = await create_category(owner_profile_id=1, name="Parent")
    child_id = await create_category(owner_profile_id=1, name="Child", parent_id=parent_id)

    subtree = await list_subtree(parent_id, 1)
    subtree_ids = {c["id"] for c in subtree}
    assert parent_id in subtree_ids
    assert child_id in subtree_ids


async def test_share_category():
    # Profile 1 creates a category, shares with profile 2
    cat_id = await create_category(owner_profile_id=1, name="Shared Folder")
    await share_category(cat_id, 2, permission="read")

    # Profile 2 can see it via include_shared=True
    cats_2 = await list_categories(2, include_shared=True)
    assert any(c["id"] == cat_id for c in cats_2)

    # Profile 2 cannot see it without sharing
    cats_2_no_share = await list_categories(2, include_shared=False)
    assert not any(c["id"] == cat_id for c in cats_2_no_share)


async def test_resolve_category_by_name():
    await create_category(owner_profile_id=1, name="My Recipes")

    # Exact case
    row = await resolve_category_by_name("My Recipes", 1)
    assert row is not None
    assert row["name"] == "My Recipes"

    # Case-insensitive
    row_lower = await resolve_category_by_name("my recipes", 1)
    assert row_lower is not None
    assert row_lower["name"] == "My Recipes"

    # Non-existent
    row_none = await resolve_category_by_name("Does Not Exist", 1)
    assert row_none is None


async def test_delete_and_restore_category():
    cat_id = await create_category(owner_profile_id=1, name="Temp Folder")

    # Visible before delete
    cats = await list_categories(1)
    assert any(c["id"] == cat_id for c in cats)

    # Soft-delete
    ok = await delete_category(cat_id, 1)
    assert ok is True

    # Not visible after soft-delete
    cats_after = await list_categories(1)
    assert not any(c["id"] == cat_id for c in cats_after)

    # Restore
    ok2 = await restore_category(cat_id, 1)
    assert ok2 is True

    # Visible again
    cats_restored = await list_categories(1)
    assert any(c["id"] == cat_id for c in cats_restored)


async def test_rename_category():
    cat_id = await create_category(owner_profile_id=1, name="Old Name")

    ok = await rename_category(cat_id, 1, "New Name")
    assert ok is True

    cats = await list_categories(1)
    matching = [c for c in cats if c["id"] == cat_id]
    assert len(matching) == 1
    assert matching[0]["name"] == "New Name"
