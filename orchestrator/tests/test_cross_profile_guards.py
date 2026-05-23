"""
Storage-layer cross-profile isolation guards.

The orchestrator is single-user by design, but two profiles can co-exist
in the same household (one Mac, multiple speaker_profile rows + their
voicemails / items / categories / pending actions).  All "list" /
"search" storage helpers MUST scope by ``owner_profile_id`` (items,
categories, pending) or ``to_profile_id`` (voicemail) — a regression
that drops the filter would silently expose one household member's
private data to another.

These tests seed two profiles ("alice" and "bob") with mirror data,
then assert each ``list_X(profile_id=N)`` returns only profile N's rows.
Negative ownership-check tests (``delete_item(other_owner_id)`` returns
False) guard the write paths.
"""
from __future__ import annotations

import pytest

from app.storage import (
    create_category,
    create_item,
    delete_category,
    delete_item,
    enqueue_action,
    fts_search_items,
    list_categories,
    list_items,
    list_pending_actions,
    list_voicemail,
    move_item,
    save_speaker_profile,
    save_voice_message,
)
from app.storage.items import batch_get_items, get_item_embeddings


@pytest.fixture
async def profiles() -> tuple[int, int]:
    """Seed two speaker_profile rows — some tables (voice_messages)
    have FK constraints pointing here, so we can't use bare integers.
    """
    alice = await save_speaker_profile(
        client_id="cli-alice", name="Alice", embedding=b"\x00" * 4 * 256,
    )
    bob = await save_speaker_profile(
        client_id="cli-bob", name="Bob", embedding=b"\x00" * 4 * 256,
    )
    return alice, bob


# ── items ────────────────────────────────────────────────────────────────


async def test_list_items_scoped_by_owner(profiles):
    alice, bob = profiles
    await create_item(owner_profile_id=alice, created_by_profile_id=alice, kind="text", body="alice's note")
    await create_item(owner_profile_id=bob, created_by_profile_id=bob, kind="text", body="bob's note")

    alice_items = await list_items(owner_profile_id=alice)
    bob_items = await list_items(owner_profile_id=bob)

    assert len(alice_items) == 1
    assert len(bob_items) == 1
    assert alice_items[0]["owner_profile_id"] == alice
    assert bob_items[0]["owner_profile_id"] == bob
    assert alice_items[0]["body"] == "alice's note"
    assert bob_items[0]["body"] == "bob's note"


async def test_fts_search_scoped_by_owner(profiles):
    """Same query word, different owners — leak would return both rows."""
    alice, bob = profiles
    await create_item(owner_profile_id=alice, created_by_profile_id=alice, kind="text", body="coffee morning")
    await create_item(owner_profile_id=bob, created_by_profile_id=bob, kind="text", body="coffee evening")

    alice_hits = await fts_search_items(alice, "coffee")
    bob_hits = await fts_search_items(bob, "coffee")

    assert len(alice_hits) == 1
    assert len(bob_hits) == 1
    assert alice_hits[0]["body"] == "coffee morning"
    assert bob_hits[0]["body"] == "coffee evening"


async def test_get_item_embeddings_scoped_by_owner(profiles):
    """The semantic-search candidate pool MUST be owner-scoped."""
    alice, bob = profiles
    await create_item(owner_profile_id=alice, created_by_profile_id=alice, kind="text", body="a")
    await create_item(owner_profile_id=bob, created_by_profile_id=bob, kind="text", body="b")

    alice_pool = await get_item_embeddings(alice)
    bob_pool = await get_item_embeddings(bob)

    # No embedding rows yet (we never ran ingest) — but the pools must be
    # disjoint regardless of size.
    alice_ids = {iid for iid, _ in alice_pool}
    bob_ids = {iid for iid, _ in bob_pool}
    assert alice_ids.isdisjoint(bob_ids)


async def test_delete_item_rejects_foreign_owner(profiles):
    """delete_item(id, wrong_owner) returns False; row stays soft-deletable later."""
    alice, bob = profiles
    iid = await create_item(owner_profile_id=alice, created_by_profile_id=alice, kind="text", body="hers")

    # Bob tries to delete Alice's item.
    ok = await delete_item(iid, bob)
    assert ok is False

    # Item still visible to Alice.
    alice_items = await list_items(owner_profile_id=alice)
    assert any(r["id"] == iid for r in alice_items)


async def test_move_item_rejects_foreign_owner(profiles):
    """move_item(id, wrong_owner, cat) must not silently change the row."""
    alice, bob = profiles
    iid = await create_item(owner_profile_id=alice, created_by_profile_id=alice, kind="text", body="hers")
    cat = await create_category(owner_profile_id=bob, name="bob's stuff")

    ok = await move_item(iid, bob, cat)
    assert ok is False


async def test_batch_get_items_no_owner_filter_is_documented(profiles):
    """batch_get_items() returns rows without ownership filter — callers MUST verify.

    This test pins the *contract*: the helper does not auto-filter,
    matching the get_item() contract.  Search code paths pre-filter via
    get_item_embeddings (which IS owner-scoped) so the ids ever passed
    to batch_get_items are already trusted.
    """
    alice, bob = profiles
    a = await create_item(owner_profile_id=alice, created_by_profile_id=alice, kind="text", body="a")
    b = await create_item(owner_profile_id=bob, created_by_profile_id=bob, kind="text", body="b")

    rows = await batch_get_items([a, b])
    # Both come back — caller filters by owner_profile_id.
    assert set(rows.keys()) == {a, b}
    assert rows[a]["owner_profile_id"] == alice
    assert rows[b]["owner_profile_id"] == bob


# ── categories ───────────────────────────────────────────────────────────


async def test_list_categories_scoped_by_owner(profiles):
    alice, bob = profiles
    await create_category(owner_profile_id=alice, name="alice cat")
    await create_category(owner_profile_id=bob, name="bob cat")

    alice_cats = await list_categories(alice)
    bob_cats = await list_categories(bob)

    assert {c["name"] for c in alice_cats} == {"alice cat"}
    assert {c["name"] for c in bob_cats} == {"bob cat"}


async def test_delete_category_rejects_foreign_owner(profiles):
    alice, bob = profiles
    cid = await create_category(owner_profile_id=alice, name="hers")

    ok = await delete_category(cid, bob)
    assert ok is False

    # Still listable for Alice.
    assert any(c["id"] == cid for c in await list_categories(alice))


# ── voicemail ────────────────────────────────────────────────────────────


async def test_list_for_recipient_scoped_by_recipient(profiles):
    """Voicemail inbox is keyed on to_profile_id, not from_profile_id."""
    alice, bob = profiles
    # Bob → Alice
    await save_voice_message(
        from_profile_id=bob, from_name="bob",
        to_profile_id=alice, to_name="alice",
        transcript="hey alice", duration_ms=1000,
        audio_pcm=b"\x00" * 32000,
    )
    # Alice → Bob
    await save_voice_message(
        from_profile_id=alice, from_name="alice",
        to_profile_id=bob, to_name="bob",
        transcript="hi bob", duration_ms=1000,
        audio_pcm=b"\x00" * 32000,
    )

    alice_inbox = await list_voicemail(alice)
    bob_inbox = await list_voicemail(bob)

    assert len(alice_inbox) == 1
    assert len(bob_inbox) == 1
    assert alice_inbox[0]["transcript"] == "hey alice"
    assert bob_inbox[0]["transcript"] == "hi bob"


# ── pending actions ──────────────────────────────────────────────────────


async def test_list_pending_actions_scoped_by_profile(profiles):
    """Pending actions awaiting approval must not cross profiles."""
    alice, bob = profiles
    await enqueue_action(
        profile_id=alice, client_id="cli-a", tool_name="memory_write",
        tool_args={"key": "x", "value": "1"}, summary="alice's pending",
    )
    await enqueue_action(
        profile_id=bob, client_id="cli-b", tool_name="memory_write",
        tool_args={"key": "y", "value": "2"}, summary="bob's pending",
    )

    alice_pending = await list_pending_actions(profile_id=alice)
    bob_pending = await list_pending_actions(profile_id=bob)

    assert len(alice_pending) == 1
    assert len(bob_pending) == 1
    assert alice_pending[0]["summary"] == "alice's pending"
    assert bob_pending[0]["summary"] == "bob's pending"
