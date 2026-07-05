"""
Tests for the step-up auth flow (#55).

Scope:
  * storage.step_up_grants — create / consume / expired
  * registry.broadcast_step_up_granted — session found / not found
  * agent._execute_one — private tool gating (step_up_auth False/True/no profile)
  * /api/step-up/approve GET — valid / expired token
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Storage: create / consume ─────────────────────────────────────────


async def test_create_and_consume_grant():
    """Round-trip: create a grant, consume it, get back the profile_id."""
    from app.storage.step_up_grants import create_grant, consume_grant

    profile_id = 42
    token = await create_grant(profile_id=profile_id, client_id="abc")
    assert isinstance(token, str) and len(token) > 16

    # Valid consumption returns (profile_id, client_id) so the caller can
    # elevate ONLY the originating session.
    result = await consume_grant(token)
    assert result == (profile_id, "abc")

    # Second consumption → None (already deleted, single-use).
    result2 = await consume_grant(token)
    assert result2 is None


async def test_consume_expired_grant():
    """An expired grant (inserted with expires_at in the past) returns None."""
    import asyncio
    import secrets
    from app.storage.db import _conn, _lock

    # Insert a grant row directly with an already-expired timestamp.
    token = secrets.token_urlsafe(24)
    now = time.time()
    expired_at = now - 10  # 10 seconds ago
    def _insert():
        with _lock:
            c = _conn()
            try:
                c.execute(
                    "INSERT INTO step_up_grants(token, profile_id, client_id, issued_at, expires_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (token, 7, None, now - 20, expired_at),
                )
            finally:
                c.close()
    await asyncio.to_thread(_insert)

    from app.storage.step_up_grants import consume_grant
    result = await consume_grant(token)
    assert result is None


async def test_consume_unknown_token():
    """A token that was never created returns None."""
    from app.storage.step_up_grants import consume_grant
    assert await consume_grant("nonexistent_token_xyz") is None


# ── Registry: broadcast_step_up_granted ──────────────────────────────


async def test_broadcast_step_up_granted_found():
    """Session exists + profile matches → on_step_up_granted called, returns 1."""
    session = MagicMock()
    session.on_step_up_granted = AsyncMock()
    session.auth_profile_id = 99

    with patch("app.registry._sessions", {"client_abc": session}):
        from app.registry import broadcast_step_up_granted
        n = await broadcast_step_up_granted("client_abc", profile_id=99, window_s=120)

    assert n == 1
    session.on_step_up_granted.assert_awaited_once_with(window_s=120)


async def test_broadcast_step_up_granted_not_found():
    """Session missing → returns 0, no error."""
    with patch("app.registry._sessions", {}):
        from app.registry import broadcast_step_up_granted
        n = await broadcast_step_up_granted("ghost_client", profile_id=1, window_s=60)
    assert n == 0


async def test_broadcast_step_up_granted_profile_mismatch_refused():
    """Session's profile differs from the grant's → refused (returns 0).

    Defence in depth: a leaked token whose client_id slot is now occupied
    by a DIFFERENT profile must not elevate that session.
    """
    session = MagicMock()
    session.on_step_up_granted = AsyncMock()
    session.auth_profile_id = 7  # different from the grant's profile

    with patch("app.registry._sessions", {"client_abc": session}):
        from app.registry import broadcast_step_up_granted
        n = await broadcast_step_up_granted("client_abc", profile_id=99, window_s=120)

    assert n == 0
    session.on_step_up_granted.assert_not_awaited()


# ── Agent loop: private tool gating ──────────────────────────────────


class _FakeCtxNoStepUp:
    client_id = "client_test"
    profile_id = 99
    user_lang = "en"
    is_authenticated = False
    step_up_auth = False
    device_kind = None


class _FakeCtxWithStepUp:
    client_id = "client_test"
    profile_id = 99
    user_lang = "en"
    is_authenticated = False
    step_up_auth = True
    device_kind = None


async def test_private_tool_blocked_without_step_up():
    """A private tool is blocked and push is sent when step_up_auth is False."""
    from app.tools.base import TOOL_REGISTRY, tool, ToolResult
    from app.llm_utils import ParsedToolCall

    # Register a throwaway private tool.
    @tool(
        "test_private_tool_a",
        "desc",
        {"type": "object", "properties": {}, "required": []},
        private=True,
    )
    async def _private_fn():
        return ToolResult(text="secret data")

    tc = ParsedToolCall(id="id1", name="test_private_tool_a", args={})
    ctx = _FakeCtxNoStepUp()

    with (
        patch("app.storage.step_up_grants.create_grant", AsyncMock(return_value="tok123")),
        patch("app.push.send_to_profile", AsyncMock(return_value=1)),
    ):
        from app.agent import _execute_one
        inv = await _execute_one(tc, ctx)

    assert inv.terminal is True
    assert inv.data.get("step_up_pending") is True
    # Clean up registry.
    TOOL_REGISTRY.pop("test_private_tool_a", None)


async def test_private_tool_allowed_with_step_up():
    """A private tool executes normally when step_up_auth is True."""
    from app.tools.base import TOOL_REGISTRY, tool, ToolResult
    from app.llm_utils import ParsedToolCall

    @tool(
        "test_private_tool_b",
        "desc",
        {"type": "object", "properties": {}, "required": []},
        private=True,
    )
    async def _private_fn_b():
        return ToolResult(text="secret data ok")

    tc = ParsedToolCall(id="id2", name="test_private_tool_b", args={})
    ctx = _FakeCtxWithStepUp()

    from app.agent import _execute_one
    inv = await _execute_one(tc, ctx)

    assert inv.text == "secret data ok"
    assert not inv.data or not inv.data.get("step_up_pending")
    TOOL_REGISTRY.pop("test_private_tool_b", None)


async def test_private_tool_no_profile_returns_error():
    """A private tool with no profile_id returns a clear error."""
    from app.tools.base import TOOL_REGISTRY, tool, ToolResult
    from app.llm_utils import ParsedToolCall

    @tool(
        "test_private_tool_c",
        "desc",
        {"type": "object", "properties": {}, "required": []},
        private=True,
    )
    async def _private_fn_c():
        return ToolResult(text="private")

    class _NoProfileCtx:
        client_id = "anon"
        profile_id = None
        user_lang = "en"
        is_authenticated = False
        step_up_auth = False
        device_kind = None

    tc = ParsedToolCall(id="id3", name="test_private_tool_c", args={})
    from app.agent import _execute_one
    inv = await _execute_one(tc, _NoProfileCtx())

    assert inv.data.get("error") == "step_up_no_profile"
    TOOL_REGISTRY.pop("test_private_tool_c", None)


# ── HTTP route: POST /api/step-up/approve ────────────────────────────


async def test_approve_route_valid_token():
    """POST /api/step-up/approve with a valid token elevates the grant's session."""
    from fastapi.testclient import TestClient
    from app.main import app

    broadcast_mock = AsyncMock(return_value=1)
    with (
        # consume now returns (profile_id, client_id) — the route must use
        # the STORED client_id, never one from the request body.
        patch("app.routes.step_up.consume_step_up_grant",
              AsyncMock(return_value=(5, "client_x"))),
        patch("app.routes.step_up.registry.broadcast_step_up_granted", broadcast_mock),
    ):
        with TestClient(app) as client:
            resp = client.post("/api/step-up/approve", json={"token": "valid_tok"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["elevated"] is True
    # The route must target the grant's stored client_id + profile_id.
    broadcast_mock.assert_awaited_once()
    _args, kwargs = broadcast_mock.call_args
    assert _args[0] == "client_x"
    assert kwargs["profile_id"] == 5


async def test_approve_route_expired_token():
    """POST /api/step-up/approve with an expired / invalid token → 400."""
    from fastapi.testclient import TestClient
    from app.main import app

    with patch("app.routes.step_up.consume_step_up_grant", AsyncMock(return_value=None)):
        with TestClient(app) as client:
            resp = client.post("/api/step-up/approve", json={"token": "bad_tok"})

    assert resp.status_code == 400
