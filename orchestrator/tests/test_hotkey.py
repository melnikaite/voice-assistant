"""
Tests for the global dictation hotkey (#41 — Osaurus pattern).

Scope:
  * registry.broadcast_ptt_trigger — sends ptt_trigger to all sessions
  * /api/hotkey/ptt route — 401 on missing/bad token, 200 on valid token
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── registry.broadcast_ptt_trigger ───────────────────────────────────


async def test_broadcast_ptt_trigger_empty():
    """No sessions → 0 sent, no errors."""
    with patch("app.registry._sessions", {}):
        from app.registry import broadcast_ptt_trigger
        n = await broadcast_ptt_trigger()
    assert n == 0


async def test_broadcast_ptt_trigger_one_session():
    """One connected session → receives ptt_trigger payload."""
    session = MagicMock()
    session._send = AsyncMock()
    session.client_id = "abc123"

    with patch("app.registry._sessions", {"abc123": session}):
        from app.registry import broadcast_ptt_trigger
        n = await broadcast_ptt_trigger(hold_ms=8000)

    assert n == 1
    session._send.assert_awaited_once_with({"type": "ptt_trigger", "hold_ms": 8000})


async def test_broadcast_ptt_trigger_two_sessions():
    """Two sessions → both receive the event."""
    s1, s2 = MagicMock(), MagicMock()
    s1._send = AsyncMock()
    s2._send = AsyncMock()
    s1.client_id = "s1"
    s2.client_id = "s2"

    with patch("app.registry._sessions", {"s1": s1, "s2": s2}):
        from app.registry import broadcast_ptt_trigger
        n = await broadcast_ptt_trigger()

    assert n == 2
    s1._send.assert_awaited_once()
    s2._send.assert_awaited_once()


async def test_broadcast_ptt_trigger_failed_session():
    """A session whose _send raises → swallowed; other sessions still notified."""
    s_bad = MagicMock()
    s_bad._send = AsyncMock(side_effect=RuntimeError("socket closed"))
    s_bad.client_id = "bad"

    s_good = MagicMock()
    s_good._send = AsyncMock()
    s_good.client_id = "good"

    with patch("app.registry._sessions", {"bad": s_bad, "good": s_good}):
        from app.registry import broadcast_ptt_trigger
        # Should not raise even though s_bad fails
        n = await broadcast_ptt_trigger()

    # Only the good session counts
    assert n == 1
    s_good._send.assert_awaited_once()


# ── desktop_client.is_valid_token ────────────────────────────────────


def test_is_valid_token_match():
    """Token matching any agent's token → True."""
    from app.desktop_client import AgentInfo, _AGENTS
    agent = AgentInfo(agent_id="test", url="http://localhost:9877", token="secret42")
    with patch.dict(_AGENTS, {"test": agent}):
        from app.desktop_client import is_valid_token
        assert is_valid_token("secret42") is True


def test_is_valid_token_mismatch():
    """Wrong token → False."""
    from app.desktop_client import AgentInfo, _AGENTS
    agent = AgentInfo(agent_id="test", url="http://localhost:9877", token="secret42")
    with patch.dict(_AGENTS, {"test": agent}):
        from app.desktop_client import is_valid_token
        assert is_valid_token("wrongtoken") is False


def test_is_valid_token_empty():
    """Empty string → False."""
    from app.desktop_client import is_valid_token
    assert is_valid_token("") is False
