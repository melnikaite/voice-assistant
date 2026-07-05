"""Instance-level settings API (#43).

Owner-accessible endpoints for reading and toggling instance settings.
``_current_user`` is used as the auth guard — any logged-in profile can
read/toggle these for now; a future ``is_owner`` flag on speaker_profiles
can narrow this to the primary profile.

Security note on one-way toggle
────────────────────────────────
``registration_open`` can only be turned **OFF** via the API.  Turning it
back ON requires manual editing of ``/data/settings.json`` + restart.
This is defence-in-depth: an attacker who captures a logged-in session
cannot reopen registration (and add their own profile with high-write
permissions) without host-level access.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import instance_settings
from ._deps import _current_user

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

@router.get("/api/instance/settings")
async def api_instance_settings(
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Return the current instance settings (non-sensitive fields only).

    Response:
        {
          "registration_open": true,
          "allow_guest_voice": false,
          "basic_auth_configured": false
        }

    The ``basic_auth_password_hash`` is never returned; only a boolean
    indicating whether Basic Auth is active.
    """
    cfg = await instance_settings.read()
    return JSONResponse({
        "registration_open": cfg.registration_open,
        "allow_guest_voice": cfg.allow_guest_voice,
        "basic_auth_configured": bool(cfg.basic_auth_user and cfg.basic_auth_password_hash),
        "basic_auth_user": cfg.basic_auth_user,
    })


# ---------------------------------------------------------------------------
# Registration toggle (one-way: open → closed only)
# ---------------------------------------------------------------------------

@router.post("/api/instance/close-registration")
async def api_close_registration(
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Close new-profile registration permanently (one-way via API).

    Sets ``registration_open = false``.  To re-open, edit
    ``/data/settings.json`` manually and restart the orchestrator.

    Returns: {"ok": true, "registration_open": false}
    """
    cfg = await instance_settings.read()
    if not cfg.registration_open:
        return JSONResponse({"ok": True, "registration_open": False, "already_closed": True})
    await instance_settings.patch(registration_open=False)
    log.info("instance: registration closed by profile=%d", user["profile_id"])
    return JSONResponse({"ok": True, "registration_open": False})


# ---------------------------------------------------------------------------
# Guest-voice toggle (two-way)
# ---------------------------------------------------------------------------

class GuestVoiceRequest(BaseModel):
    allow: bool


@router.post("/api/instance/guest-voice")
async def api_set_guest_voice(
    req: GuestVoiceRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Enable or disable guest-voice mode.

    When enabled, unauthenticated visitors can open a WS voice session
    in read-only mode.  When disabled, the WS rejects connections that
    don't carry a valid ``va_session`` cookie.

    Returns: {"ok": true, "allow_guest_voice": <bool>}
    """
    await instance_settings.patch(allow_guest_voice=req.allow)
    log.info(
        "instance: allow_guest_voice=%s by profile=%d",
        req.allow, user["profile_id"],
    )
    return JSONResponse({"ok": True, "allow_guest_voice": req.allow})


# ---------------------------------------------------------------------------
# Basic Auth setup
# ---------------------------------------------------------------------------

class BasicAuthRequest(BaseModel):
    username: str
    password: str   # plaintext — will be hashed before storage


@router.post("/api/instance/basic-auth")
async def api_set_basic_auth(
    req: BasicAuthRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Configure HTTP Basic Auth for the instance.

    Hashes the password with bcrypt and stores both in
    ``/data/settings.json``.  Takes effect after orchestrator restart
    (the middleware is installed at startup).

    To disable Basic Auth: set ``basic_auth_user`` and
    ``basic_auth_password_hash`` to null in ``/data/settings.json`` and
    restart.

    Returns: {"ok": true, "basic_auth_user": "...", "note": "restart required"}
    """
    if not req.username.strip():
        raise HTTPException(400, "username must not be empty")
    if not req.password:
        raise HTTPException(400, "password must not be empty")

    import bcrypt
    hashed = bcrypt.hashpw(
        req.password.encode("utf-8"),
        bcrypt.gensalt(rounds=12),
    ).decode("ascii")

    await instance_settings.patch(
        basic_auth_user=req.username.strip(),
        basic_auth_password_hash=hashed,
    )
    log.info(
        "instance: basic-auth configured for user %r by profile=%d",
        req.username, user["profile_id"],
    )
    return JSONResponse({
        "ok": True,
        "basic_auth_user": req.username.strip(),
        "note": "restart orchestrator for Basic Auth to take effect",
    })
