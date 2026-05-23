"""
agent_proxy — Future correlation for the reverse-WSS RPC layer.

We can't easily spin up a full WSS server + agent subprocess in unit
tests; that's a brittle integration story.  What we CAN unit-test is
the in-process glue:

  • ``AgentConnection.call()`` posts a frame and awaits a Future keyed
    by the generated call_id.
  • ``handle_result()`` resolves the matching Future with ``data`` on
    success or :class:`DesktopUnavailable` on ``ok=False``.
  • A timeout cleans up the pending entry.
  • ``close()`` rejects every still-pending Future with a transport
    error.

The WSS itself is replaced with a tiny ``FakeWS`` that records sent
frames and lets the test inject incoming ones.
"""
from __future__ import annotations

import asyncio
import json

import pytest


class FakeWS:
    """Stand-in for a FastAPI WebSocket.

    Records ``send_text`` calls so the test can read the frame the
    connection sent; exposes ``inject(frame)`` to simulate an incoming
    message (the real WSS would deliver these via the recv loop).
    """

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send_text(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        self.closed = True


async def test_agent_connection_call_correlates_result():
    """call() posts a frame, awaits the matching result Future."""
    from app import agent_proxy

    ws = FakeWS()
    conn = agent_proxy.AgentConnection(
        ws,  # type: ignore[arg-type] — FakeWS is structurally compatible
        agent_id="testagent",
        session_id="sess",
        capabilities={"capabilities": {"screenshot": True}},
    )

    # Run call() in the background; meanwhile inject the matching
    # ``result`` frame and assert call() returns the data.
    async def driver():
        return await conn.call("screenshot", {}, timeout=5.0)

    task = asyncio.create_task(driver())
    # Yield once so call() runs to the point of awaiting its Future.
    await asyncio.sleep(0)

    # The frame call() posted must have a fresh call_id we can echo.
    assert len(ws.sent) == 1
    frame = ws.sent[0]
    assert frame["type"] == "call"
    assert frame["method"] == "screenshot"
    call_id = frame["call_id"]

    # Inject the result.
    conn.handle_result({
        "type": "result", "call_id": call_id,
        "ok": True, "data": {"png_b64": "abc"},
    })
    out = await task
    assert out == {"png_b64": "abc"}


async def test_agent_connection_handles_error_result():
    """An ``ok=False`` result raises DesktopUnavailable in the caller."""
    from app import agent_proxy
    from app.desktop_client import DesktopUnavailable

    ws = FakeWS()
    conn = agent_proxy.AgentConnection(
        ws, agent_id="t", session_id="s", capabilities={},  # type: ignore[arg-type]
    )

    async def driver():
        return await conn.call("applescript", {}, timeout=5.0)

    task = asyncio.create_task(driver())
    await asyncio.sleep(0)
    call_id = ws.sent[0]["call_id"]
    conn.handle_result({
        "type": "result", "call_id": call_id,
        "ok": False, "error": "osascript not available",
    })
    with pytest.raises(DesktopUnavailable) as exc:
        await task
    assert "osascript not available" in str(exc.value)


async def test_agent_connection_call_times_out():
    """call() raises DesktopUnavailable on timeout + cleans up state."""
    from app import agent_proxy
    from app.desktop_client import DesktopUnavailable

    ws = FakeWS()
    conn = agent_proxy.AgentConnection(
        ws, agent_id="t", session_id="s", capabilities={},  # type: ignore[arg-type]
    )

    with pytest.raises(DesktopUnavailable) as exc:
        await conn.call("screenshot", {}, timeout=0.05)
    assert "timeout" in str(exc.value).lower()
    # Pending dict must be empty post-timeout — no leaked Futures.
    assert len(conn._pending) == 0


async def test_agent_connection_close_rejects_pending():
    """close() resolves every in-flight Future with a transport error.

    Without this, a caller awaiting a result when the WSS drops would
    hang forever.  We pin this contract because it's the difference
    between «graceful reconnect» and «orchestrator dies».
    """
    from app import agent_proxy
    from app.desktop_client import DesktopUnavailable

    ws = FakeWS()
    conn = agent_proxy.AgentConnection(
        ws, agent_id="t", session_id="s", capabilities={},  # type: ignore[arg-type]
    )

    # Start a call that will never get its result.
    async def driver():
        return await conn.call("screenshot", {}, timeout=5.0)

    task = asyncio.create_task(driver())
    await asyncio.sleep(0)
    assert len(conn._pending) == 1

    # Drop the connection.
    await conn.close(reason="test")
    with pytest.raises(DesktopUnavailable) as exc:
        await task
    assert "connection lost" in str(exc.value)
    assert ws.closed is True
