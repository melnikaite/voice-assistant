"""
Reverse-WSS agent proxy — Phase 3c.

Default orchestrator deployment lives behind localhost+host-networking
and talks to the desktop-agent via HTTP on 127.0.0.1.  When the agent
sits behind NAT (laptop at a coffee shop, work-pc on another LAN) the
orchestrator can't reach it that way.  This module flips the polarity:
the agent dials the orchestrator over WSS and holds the connection;
the orchestrator pushes RPC calls down the same socket.

Protocol (v1 — newline-delimited JSON over WSS, no schema versioning
yet; bump the ``version`` field on a breaking change)
─────────────────────────────────────────────────────────────────────

Agent → Orchestrator on connect (``/v1/agent/connect``):
    {"type": "hello", "agent_id": "...", "token": "<DESKTOP_TOKEN>",
     "capabilities": {...}, "version": "1.1.0"}

Orchestrator → Agent:
    {"type": "hello_ack", "session_id": "..."}
    {"type": "reject",    "reason": "auth"|"version"|"protocol"}

Orchestrator → Agent (RPC):
    {"type": "call", "call_id": "uuid", "method": "screenshot",
     "params": {...}}

Agent → Orchestrator (RPC response):
    {"type": "result", "call_id": "uuid", "ok": true,  "data": {...}}
    {"type": "result", "call_id": "uuid", "ok": false, "error": "..."}

Keepalive:
    {"type": "ping"}  ↔  {"type": "pong"}     (every 30 s)

Threading model: one :class:`AgentConnection` per live WSS.  The
connection runs a ``recv_loop`` task that dispatches incoming
``result`` frames by resolving the per-call ``asyncio.Future`` kept in
``_pending``.  Callers go through ``conn.call(method, params)`` which
posts the call frame and waits on the Future.  No locks — every
mutation happens on the event loop thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from . import desktop_client

log = logging.getLogger(__name__)


# Per-call default timeout (seconds).  Callers (desktop_client) pass
# their own based on the operation (screenshot ~ 35 s, default_app ~
# 10 s, cursor_activity ~ 5 s); this is the absolute fallback for
# callers that omit it.
_DEFAULT_CALL_TIMEOUT_S = 30.0


# ── Connection registry ────────────────────────────────────────────────


# One entry per live WSS.  Keyed by agent_id — if the same agent_id
# reconnects we evict the prior one (and cancel its pending futures)
# so a flapping link doesn't accumulate zombie connections.
_CONNS: dict[str, "AgentConnection"] = {}


def get_connection(agent_id: str) -> "AgentConnection | None":
    """Return the live WSS connection for an agent_id, or None.

    Used by :mod:`.desktop_client` to dispatch RPC calls when an agent
    is registered via reverse mode.
    """
    conn = _CONNS.get(agent_id)
    if conn is None or conn.closed:
        return None
    return conn


def list_connections() -> list["AgentConnection"]:
    """All currently live reverse-mode connections."""
    return [c for c in _CONNS.values() if not c.closed]


# ── AgentConnection ────────────────────────────────────────────────────


class AgentConnection:
    """One live WSS link to a reverse-mode agent.

    Owns:
      • The :class:`WebSocket` itself (FastAPI's accepted instance).
      • A ``recv_loop`` task that consumes incoming JSON frames.
      • A ``_pending`` dict mapping ``call_id`` → ``asyncio.Future``;
        the recv loop fulfils each future when the matching ``result``
        frame arrives.

    Lifecycle:
      • Constructed by the WS endpoint after a successful hello.
      • ``call(method, params, timeout)`` posts a call frame and awaits
        the result future.
      • ``close()`` cancels every pending future with a transport
        error and removes the entry from the registry.

    Design notes:
      * We don't queue calls — the underlying WSS is full-duplex and
        each call carries its own ``call_id`` for correlation, so
        concurrent ``call()`` invocations interleave freely.
      * Futures live in an in-process dict; a process crash loses
        every in-flight call.  That's acceptable for voice-assistant
        use cases (re-issue from the LLM if needed).
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        agent_id: str,
        session_id: str,
        capabilities: dict[str, Any],
    ) -> None:
        self.ws = websocket
        self.agent_id = agent_id
        self.session_id = session_id
        self.capabilities = capabilities or {}
        self._pending: dict[str, asyncio.Future] = {}
        self._closed = False
        self.connected_at = time.time()

    @property
    def closed(self) -> bool:
        return self._closed

    # ── outgoing ───────────────────────────────────────────────────

    async def _send(self, payload: dict) -> None:
        if self._closed:
            raise desktop_client.DesktopUnavailable("connection closed")
        await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def call(
        self,
        method: str,
        params: dict,
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT_S,
    ) -> Any:
        """Post one RPC call, await the matching result frame.

        Returns the agent's ``data`` payload on success; raises
        :class:`DesktopUnavailable` on any failure (timeout, transport
        loss, agent-reported error).
        """
        if self._closed:
            raise desktop_client.DesktopUnavailable("connection closed")
        call_id = secrets.token_urlsafe(12)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[call_id] = fut
        try:
            await self._send({
                "type": "call",
                "call_id": call_id,
                "method": method,
                "params": params,
            })
        except Exception as exc:
            self._pending.pop(call_id, None)
            raise desktop_client.DesktopUnavailable(
                f"send failed: {exc.__class__.__name__}"
            ) from exc
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(call_id, None)
            raise desktop_client.DesktopUnavailable(
                f"reverse: timeout after {timeout:.1f}s on {method}"
            )
        finally:
            # Belt-and-braces: the future MAY have been popped already
            # by ``handle_result``; this is the idempotent cleanup for
            # the timeout path.
            self._pending.pop(call_id, None)

    # ── incoming ───────────────────────────────────────────────────

    def handle_result(self, frame: dict) -> None:
        """Dispatch one incoming ``result`` frame to its Future."""
        call_id = frame.get("call_id")
        if not call_id:
            log.warning("agent_proxy[%s]: result without call_id: %r", self.agent_id, frame)
            return
        fut = self._pending.pop(call_id, None)
        if fut is None or fut.done():
            log.debug(
                "agent_proxy[%s]: stale result for %s (already timed out?)",
                self.agent_id, call_id,
            )
            return
        if frame.get("ok"):
            fut.set_result(frame.get("data"))
        else:
            fut.set_exception(desktop_client.DesktopUnavailable(
                f"agent error: {frame.get('error', 'unknown')}"
            ))

    # ── teardown ───────────────────────────────────────────────────

    async def close(self, *, reason: str = "closed") -> None:
        """Drop the connection + cancel pending futures.

        Idempotent — safe to call from both the recv-loop exit and the
        registry-eviction paths.
        """
        if self._closed:
            return
        self._closed = True
        # Resolve every pending future with a transport error so callers
        # don't hang forever on a lost socket.
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(desktop_client.DesktopUnavailable(
                    f"connection lost: {reason}"
                ))
        self._pending.clear()
        try:
            await self.ws.close()
        except Exception:
            pass
        # De-register from the registry only if we're still the
        # current connection for that agent_id (a reconnect may have
        # already evicted us).
        if _CONNS.get(self.agent_id) is self:
            _CONNS.pop(self.agent_id, None)
        desktop_client.unregister_reverse_agent(self.agent_id)


# ── WS endpoint entry point ────────────────────────────────────────────


async def handle_agent_connect(websocket: WebSocket) -> None:
    """Server-side handler for ``/v1/agent/connect``.

    Flow:
      1. Accept the socket.
      2. Read the agent's ``hello`` frame; validate token against the
         agent's expected token (we don't know which agent it is until
         the hello, so we match the ``agent_id`` against the registry
         and compare tokens there).
      3. Send back ``hello_ack`` with a session_id (purely informational
         today; a future revision could use it for resume-after-blip).
      4. Loop: read frames, dispatch ``result`` to the connection's
         futures, respond to ``ping`` with ``pong``, log everything
         else.
      5. On any error: close + cleanup.
    """
    await websocket.accept()
    conn: AgentConnection | None = None
    agent_id = "?"
    try:
        # 1. Hello.
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        try:
            hello = json.loads(raw)
        except json.JSONDecodeError:
            await _reject(websocket, "protocol")
            return
        if hello.get("type") != "hello":
            await _reject(websocket, "protocol")
            return
        agent_id = str(hello.get("agent_id") or "").strip()
        token = str(hello.get("token") or "")
        capabilities_payload = hello.get("capabilities") or {}
        if not agent_id:
            await _reject(websocket, "protocol")
            return

        # 2. Auth check.  We accept either:
        #    a) an agent_id pre-listed in DESKTOP_AGENTS with a token
        #       matching the hello's token (operator-vetted), OR
        #    b) any agent_id whose hello carries the env DESKTOP_TOKEN
        #       (single-agent install — token is the only secret).
        expected = desktop_client.get_agent(agent_id)
        env_token = desktop_client.DESKTOP_TOKEN
        ok = False
        if expected and expected.token and token and expected.token == token:
            ok = True
        elif env_token and token and env_token == token:
            ok = True
        elif not env_token and not (expected and expected.token):
            # No token configured at all — accept any hello.  This is
            # the "operator deliberately disabled auth" case; the same
            # default applies on the HTTP path.
            ok = True
        if not ok:
            log.warning("agent_proxy: rejecting %r (auth)", agent_id)
            await _reject(websocket, "auth")
            return

        # 3. Hello-ack + registry insertion.  Evict any prior connection
        # under the same agent_id (a re-dial after a blip).
        existing = _CONNS.get(agent_id)
        if existing is not None and not existing.closed:
            log.info("agent_proxy[%s]: evicting prior connection", agent_id)
            await existing.close(reason="superseded")

        session_id = secrets.token_urlsafe(8)
        conn = AgentConnection(
            websocket,
            agent_id=agent_id,
            session_id=session_id,
            capabilities={
                "agent_id": agent_id,
                "platform": (hello.get("platform")
                             or capabilities_payload.get("platform")
                             or "unknown"),
                "capabilities": (capabilities_payload.get("capabilities")
                                 if isinstance(capabilities_payload, dict)
                                 and "capabilities" in capabilities_payload
                                 else capabilities_payload),
                "version": hello.get("version") or "unknown",
            },
        )
        _CONNS[agent_id] = conn
        # Plug the new connection into the global registry so
        # desktop_client routes RPC through it.
        info = desktop_client.register_reverse_agent(agent_id, token=token)
        info.capabilities_cache = conn.capabilities
        info.capabilities_cached_at = time.time()
        info.reachable = True
        info.last_seen = time.time()

        await websocket.send_text(json.dumps({
            "type": "hello_ack",
            "session_id": session_id,
        }))
        log.info(
            "agent_proxy[%s]: connected (session=%s, caps=%s)",
            agent_id, session_id,
            ",".join(k for k, v in (conn.capabilities.get("capabilities") or {}).items() if v),
        )

        # 4. Recv loop.
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("agent_proxy[%s]: bad JSON frame", agent_id)
                continue
            ftype = frame.get("type")
            if ftype == "result":
                conn.handle_result(frame)
            elif ftype == "ping":
                # Keepalive — answer with pong.  The agent's heartbeat
                # is the proof-of-life we use to keep ``reachable=True``.
                await websocket.send_text(json.dumps({"type": "pong"}))
                info.last_seen = time.time()
            elif ftype == "pong":
                # We don't actively send pings today, but accept them
                # silently for protocol forward-compat.
                info.last_seen = time.time()
            elif ftype == "heartbeat":
                # Extended heartbeat from the agent — carries lock state
                # and optionally other telemetry.  Replaces / supplements
                # the bare ``ping`` for agents that want richer reporting.
                info.last_seen = time.time()
                info.reachable = True
                locked = frame.get("locked")
                if locked is not None:
                    desktop_client.update_agent_lock_state(agent_id, bool(locked))
                # Update last_seen_at in profile_devices if this agent is
                # paired with a profile.  Async fire-and-forget — we don't
                # await because the recv loop must stay responsive.
                asyncio.create_task(_touch_paired_device(agent_id))
                log.debug(
                    "agent_proxy[%s]: heartbeat (locked=%s)",
                    agent_id, locked,
                )
            elif ftype == "pair":
                # Desktop-agent claims a pairing code so it can be linked
                # to a profile without a Settings UI.
                # Frame: {"type": "pair", "code": "123456",
                #         "friendly_name": "Mom's MacBook"}
                code = str(frame.get("code") or "").strip()
                friendly_name = str(frame.get("friendly_name") or agent_id)
                if code:
                    asyncio.create_task(
                        _handle_pair_frame(agent_id, code, friendly_name, websocket)
                    )
                else:
                    log.warning("agent_proxy[%s]: pair frame missing code", agent_id)
            elif ftype == "hello":
                # Duplicate hello — ignore.
                log.debug("agent_proxy[%s]: duplicate hello, ignoring", agent_id)
            else:
                log.warning("agent_proxy[%s]: unknown frame type %r", agent_id, ftype)
    except WebSocketDisconnect:
        log.info("agent_proxy[%s]: disconnected", agent_id)
    except asyncio.TimeoutError:
        log.warning("agent_proxy[%s]: hello timeout", agent_id)
    except Exception:
        log.exception("agent_proxy[%s]: handler crashed", agent_id)
    finally:
        if conn is not None:
            await conn.close(reason="recv_loop_exit")


async def _touch_paired_device(agent_id: str) -> None:
    """Update profile_devices.last_seen_at for the agent, best-effort."""
    try:
        from .storage.profile_devices import touch_device
        await touch_device(agent_id)
    except Exception:
        pass


async def _handle_pair_frame(
    agent_id: str, code: str, friendly_name: str, websocket: WebSocket
) -> None:
    """Consume a pairing code and link agent_id to the profile.

    Sends ``pair_ack`` with ``ok=true`` on success or
    ``pair_ack`` with ``ok=false, reason=...`` on failure.
    """
    try:
        from .storage.profile_devices import consume_pairing_code
        result = await consume_pairing_code(code, agent_id, friendly_name)
        if result is None:
            log.warning(
                "agent_proxy[%s]: pair code %r invalid/expired", agent_id, code
            )
            await websocket.send_text(json.dumps({
                "type": "pair_ack",
                "ok": False,
                "reason": "invalid_or_expired_code",
            }))
        else:
            profile_id, device_kind = result
            log.info(
                "agent_proxy[%s]: paired to profile=%d device_kind=%s",
                agent_id, profile_id, device_kind,
            )
            await websocket.send_text(json.dumps({
                "type": "pair_ack",
                "ok": True,
                "profile_id": profile_id,
                "device_kind": device_kind,
            }))
    except Exception:
        log.exception("agent_proxy[%s]: pair frame handling failed", agent_id)


async def _reject(websocket: WebSocket, reason: str) -> None:
    """Send a reject frame + close the WSS.  Best-effort."""
    try:
        await websocket.send_text(json.dumps({"type": "reject", "reason": reason}))
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass
