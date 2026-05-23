"""
Pending action lifecycle — enqueue → approve → executor dispatch.

Guards the contract:
  pending → approved → executed (or execution_failed)

If any leg breaks, the user can queue an action that never runs (or
worse: a "rejected" row that still runs).  We assert the full happy
path plus the rejection short-circuit.
"""
from __future__ import annotations

import pytest

from app.storage import (
    enqueue_action,
    get_pending_action,
    list_approved_actions,
    list_pending_actions,
    list_recent_actions,
    mark_approved,
    mark_executed,
    mark_rejected,
)
from app.storage.pending_actions import _sweep_expired_sync


async def test_enqueue_list_round_trip():
    aid = await enqueue_action(
        profile_id=1, client_id="cli-1",
        tool_name="remember", tool_args={"action": "append", "content": "milk"},
        summary="remember: milk",
    )
    rows = await list_pending_actions(profile_id=1)
    assert len(rows) == 1
    assert rows[0]["id"] == aid
    assert rows[0]["tool_name"] == "remember"
    assert rows[0]["tool_args"] == {"action": "append", "content": "milk"}


async def test_mark_approved_moves_to_approved_queue():
    aid = await enqueue_action(
        profile_id=1, client_id="cli-1",
        tool_name="remember", tool_args={"a": 1}, summary="X",
    )
    ok = await mark_approved(aid, via="ui")
    assert ok is True
    # No longer in the pending list
    pending = await list_pending_actions(profile_id=1)
    assert pending == []
    # Now visible to the executor
    approved = await list_approved_actions()
    assert len(approved) == 1 and approved[0]["id"] == aid


async def test_mark_executed_terminal():
    aid = await enqueue_action(
        profile_id=1, client_id="cli-1",
        tool_name="remember", tool_args={"a": 1}, summary="X",
    )
    await mark_approved(aid)
    await mark_executed(aid, ok=True, note="OK")
    # Gone from approved queue
    assert await list_approved_actions() == []
    # In the recent list with terminal status
    recent = await list_recent_actions(profile_id=1, limit=10)
    assert len(recent) == 1
    assert recent[0]["status"] == "executed"
    assert "OK" in recent[0]["summary"]


async def test_mark_rejected_does_not_appear_in_approved():
    aid = await enqueue_action(
        profile_id=1, client_id="cli-1",
        tool_name="remember", tool_args={"a": 1}, summary="X",
    )
    assert await mark_rejected(aid, via="ui") is True
    assert await list_approved_actions() == []
    assert await list_pending_actions(profile_id=1) == []
    recent = await list_recent_actions(profile_id=1)
    assert recent and recent[0]["status"] == "rejected"


async def test_double_approve_is_noop():
    aid = await enqueue_action(
        profile_id=1, client_id="cli-1",
        tool_name="remember", tool_args={"a": 1}, summary="X",
    )
    assert await mark_approved(aid) is True
    # Second approve hits the WHERE status='pending' clause and noops.
    assert await mark_approved(aid) is False


async def test_expired_sweep_flips_status():
    """The TTL sweep — runs from gc, not the read path."""
    import time
    aid = await enqueue_action(
        profile_id=1, client_id="cli-1",
        tool_name="remember", tool_args={"a": 1}, summary="X",
        ttl_s=-1.0,  # expire immediately
    )
    # Read path does NOT sweep anymore; the row is hidden by the
    # `expires_at > now` clause but its status is still "pending"
    # until the GC tick.
    assert await list_pending_actions(profile_id=1) == []
    row_before = await get_pending_action(aid)
    assert row_before["status"] == "pending"
    # Run the sweep (gc.py calls this on a timer)
    n = _sweep_expired_sync()
    assert n == 1
    row_after = await get_pending_action(aid)
    assert row_after["status"] == "expired"
