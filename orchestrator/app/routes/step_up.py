"""Step-up auth approval endpoint (#55).

``POST /api/step-up/approve`` is called by the service worker's
``notificationclick`` handler (a background fetch) after the user taps the
step-up push notification.  No browser session cookie is required — the
opaque grant token in the JSON body IS the proof of approval (it was
delivered to the user's device by encrypted Web Push).

The token is single-use and short-lived.  On a valid token the endpoint:
  1. Consumes (deletes) the grant, recovering its stored
     ``(profile_id, client_id)``.
  2. Elevates ONLY the session named by the grant's stored ``client_id``
     (verified against the live session's profile) via the registry —
     the request never gets to choose which session is elevated, so a
     leaked token can't elevate an attacker-chosen session.
  3. Returns 200 JSON so the SW can dismiss cleanly.

Security notes:
  * The token travels ONLY in the POST body (never the URL/query string),
    so it doesn't leak via access logs, ``Referer``, or browser history.
  * Invalid / expired tokens return 400 (not 401) — 401 would make
    browsers surface a credentials prompt, confusing for a background
    SW fetch.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import registry
from ..storage import consume_step_up_grant

log = logging.getLogger(__name__)

router = APIRouter()

# How long the approved step-up window lasts inside the WS session.
# Mirrors AUTH_WINDOW_S from pipeline.py — five minutes gives the user
# time to re-ask their question after tapping the notification.
_STEP_UP_WINDOW_S = 5 * 60


class ApproveRequest(BaseModel):
    token: str


@router.post("/api/step-up/approve")
async def step_up_approve(body: ApproveRequest) -> JSONResponse:
    """Consume a step-up grant token and elevate the requesting session.

    Returns ``{ok: true, elevated: bool}``.  ``elevated`` is False when the
    token was valid but the originating session is no longer connected
    (the user closed the tab) — the approval is still consumed.
    Raises 400 on a missing / invalid / expired token.
    """
    if not body.token:
        raise HTTPException(400, "token required")

    grant = await consume_step_up_grant(body.token)
    if grant is None:
        log.info("step-up/approve: invalid or expired token %.8s…", body.token)
        raise HTTPException(400, "token invalid or expired")

    profile_id, client_id = grant
    elevated = 0
    if client_id:
        elevated = await registry.broadcast_step_up_granted(
            client_id,
            profile_id=profile_id,
            window_s=_STEP_UP_WINDOW_S,
        )
    else:
        # Legacy grant with no client_id recorded — can't safely target a
        # single session, so we don't elevate (fail closed).  Shouldn't
        # happen for grants created by the current agent loop.
        log.warning("step-up/approve: grant for profile=%d has no client_id", profile_id)

    log.info(
        "step-up/approve: profile=%d → elevated=%d session(s)",
        profile_id, elevated,
    )
    return JSONResponse({"ok": True, "elevated": bool(elevated)})
