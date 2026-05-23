"""
Tool risk gate — the single most important invariant in the agent loop.

If this regresses, an unauthenticated speaker could execute a
``high_write`` tool (memory edit, settings change, anything queued
behind the passphrase).  We assert two halves:

1. ``risk="high_write"`` + ``is_authenticated=False`` →
   - real handler is NOT called
   - row is enqueued into pending_actions
   - returned ToolInvocation is terminal with ``data["deferred"]=True``

2. ``risk="high_write"`` + ``is_authenticated=True`` →
   - real handler IS called
   - no pending_actions row created
"""
from __future__ import annotations

import pytest

from app.agent import _execute_one
from app.llm_utils import ParsedToolCall
from app.tools.base import ToolResult, tool


# Local tool fixtures.  These register into TOOL_REGISTRY at module
# import time — harmless: real production tools share the same dict
# but with different names.
_calls: list[str] = []


@tool(
    name="_test_high_write_tool",
    description="Test sink for risk gating.",
    params_schema={"type": "object", "properties": {}, "required": []},
    risk="high_write",
)
async def _high_write_handler() -> ToolResult:
    _calls.append("high_write")
    return ToolResult(text="ran", data={"ran": True})


@tool(
    name="_test_read_tool",
    description="Test sink for risk gating (read path).",
    params_schema={"type": "object", "properties": {}, "required": []},
    risk="read",
)
async def _read_handler() -> ToolResult:
    _calls.append("read")
    return ToolResult(text="read-ok", data={})


@pytest.fixture(autouse=True)
def _reset_calls():
    _calls.clear()
    yield


async def test_high_write_unauth_defers(make_agent_ctx):
    """Unauthenticated high_write tool → enqueued, not executed."""
    from app.storage import list_pending_actions

    ctx = make_agent_ctx(is_authenticated=False, profile_id=1, client_id="cli-A")
    tc = ParsedToolCall(id="x", name="_test_high_write_tool", args={})
    inv = await _execute_one(tc, ctx)

    assert _calls == [], "handler must NOT run when unauthenticated"
    assert inv.terminal is True
    assert inv.data and inv.data.get("deferred") is True
    assert inv.data.get("risk") == "high_write"

    queued = await list_pending_actions(profile_id=1, client_id="cli-A")
    assert len(queued) == 1, "exactly one row should be enqueued"
    assert queued[0]["tool_name"] == "_test_high_write_tool"


async def test_high_write_authed_runs(make_agent_ctx):
    """Authenticated high_write tool → real handler runs, no enqueue."""
    from app.storage import list_pending_actions

    ctx = make_agent_ctx(is_authenticated=True, profile_id=2, client_id="cli-B")
    tc = ParsedToolCall(id="y", name="_test_high_write_tool", args={})
    inv = await _execute_one(tc, ctx)

    assert _calls == ["high_write"], "handler must run when authenticated"
    assert inv.text == "ran"
    queued = await list_pending_actions(profile_id=2, client_id="cli-B")
    assert queued == [], "no enqueue on the auth path"


async def test_read_runs_regardless_of_auth(make_agent_ctx):
    """Read tools are never gated by the passphrase."""
    ctx = make_agent_ctx(is_authenticated=False, profile_id=3, client_id="cli-C")
    tc = ParsedToolCall(id="z", name="_test_read_tool", args={})
    inv = await _execute_one(tc, ctx)
    assert _calls == ["read"]
    assert inv.text == "read-ok"
    assert not (inv.data and inv.data.get("deferred"))
