"""Global dictation hotkey webhook (#41 — Osaurus pattern).

The desktop-agent's pynput global hotkey listener POSTs here when the
user presses the configured combo (default Ctrl+Shift+Space).  The
orchestrator fans the event out to every active browser session as a
``ptt_trigger`` message, which the frontend handles by toggling PTT:

  First trigger  → pttStart() + 10-second auto-release timer
  Second trigger → pttEnd() immediately (toggle off)
  Timeout        → pttEnd() automatically after hold_ms

Auth: ``X-Desktop-Token`` header (same shared secret as all other
agent endpoints).  This endpoint is intentionally NOT behind
``va_session`` so the desktop-agent can call it without a browser
session cookie.

Endpoint
────────
POST /api/hotkey/ptt
  Request body: none
  Response:     {"ok": true, "sessions": <int>}
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from .. import desktop_client, registry

router = APIRouter()
log = logging.getLogger(__name__)

# Auto-release timeout given to the browser.  The browser releases PTT
# after this many milliseconds if the user hasn't done so themselves.
_HOLD_MS = 10_000


@router.post("/api/hotkey/ptt")
async def api_hotkey_ptt(
    x_desktop_token: str | None = Header(default=None),
) -> JSONResponse:
    """Receive a hotkey-fired PTT trigger from the desktop-agent.

    The endpoint validates the shared desktop token and broadcasts
    ``{type: "ptt_trigger", hold_ms: 10000}`` to every active
    WebSocket session.  The browser toggles PTT state on receipt.

    Returns ``{"ok": true, "sessions": N}`` where N is the number of
    sessions that received the event.  N == 0 is not an error — it
    means no browser tabs have an active session.
    """
    # Validate token — accept any registered agent's token.
    if not x_desktop_token:
        raise HTTPException(401, "X-Desktop-Token required")

    # Check against all configured agents (multi-agent setups may have
    # different tokens per agent).  Also check the module-level
    # DESKTOP_TOKEN env var as a fallback (single-agent default).
    if not desktop_client.is_valid_token(x_desktop_token):
        raise HTTPException(401, "invalid desktop token")

    n = await registry.broadcast_ptt_trigger(hold_ms=_HOLD_MS)
    log.info("hotkey/ptt: triggered, %d session(s) notified", n)
    return JSONResponse({"ok": True, "sessions": n})
