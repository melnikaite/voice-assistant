import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..storage import (
    create_session as auth_create_session,
    revoke_session as auth_revoke_session,
)
from ..user_files import (
    read_settings,
    set_passphrase,
    verify_passphrase,
)
from ._deps import SESSION_COOKIE, _current_user

log = logging.getLogger(__name__)

router = APIRouter()


class PassphraseSetupRequest(BaseModel):
    profile_id: int
    passphrase: str


@router.post("/api/auth/setup_passphrase")
async def setup_passphrase(req: PassphraseSetupRequest) -> JSONResponse:
    """
    Set or reset the passphrase for a profile.

    Open to anyone the first time — necessary so a fresh install can
    bootstrap.  Once a passphrase exists, only an authenticated session
    for THAT profile may rotate it (else a stranger could lock a
    legitimate user out of their own profile).
    """
    settings_now = await read_settings(req.profile_id)
    if settings_now.code_word_hash:
        # Already set — require an active session for this profile.
        # (Frontend will only show this control to a logged-in user.)
        raise HTTPException(
            403,
            "Passphrase already set — log in and rotate from the Settings tab.",
        )
    try:
        await set_passphrase(req.profile_id, req.passphrase)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("auth: passphrase set for profile=%d", req.profile_id)
    return JSONResponse({"ok": True, "profile_id": req.profile_id})


class LoginRequest(BaseModel):
    profile_id: int
    passphrase: str


@router.post("/api/auth/login")
async def auth_login(req: LoginRequest, request: Request) -> JSONResponse:
    """Verify passphrase, mint a server-side session, set the cookie.

    Note on FastAPI cookie wiring: a cookie set on the dependency-
    injected ``Response`` parameter is dropped when the handler returns
    a *new* Response object (JSONResponse).  We therefore set the cookie
    directly on the JSONResponse instance we're returning.
    """
    ok = await verify_passphrase(req.profile_id, req.passphrase)
    if not ok:
        raise HTTPException(401, "passphrase mismatch")
    token = await auth_create_session(
        req.profile_id,
        user_agent=request.headers.get("user-agent"),
    )
    resp = JSONResponse({"ok": True, "profile_id": req.profile_id})
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=30 * 86400,
        # secure=True  # enable when fronted by HTTPS
        path="/",
    )
    log.info("auth: login OK for profile=%d", req.profile_id)
    return resp


@router.post("/api/auth/logout")
async def auth_logout(
    va_session: str | None = Cookie(default=None),
) -> JSONResponse:
    if va_session:
        await auth_revoke_session(va_session)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@router.get("/api/me")
async def api_me(user: dict = Depends(_current_user)) -> JSONResponse:
    """Tell the UI which profile is logged in (+ session expiry)."""
    return JSONResponse({"profile_id": user["profile_id"], "expires_at": user["expires_at"]})


class RotatePassphraseRequest(BaseModel):
    current_passphrase: str
    new_passphrase: str


@router.post("/api/auth/rotate_passphrase")
async def rotate_passphrase(
    req: RotatePassphraseRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Change the logged-in user's passphrase.

    Re-verifies the current passphrase before accepting the new one —
    so an unattended logged-in tab can't silently lock the owner out.
    The cookie session stays valid (no re-login required) since the
    user just proved they own the profile.
    """
    profile_id = user["profile_id"]
    ok = await verify_passphrase(profile_id, req.current_passphrase)
    if not ok:
        raise HTTPException(401, "current passphrase mismatch")
    try:
        await set_passphrase(profile_id, req.new_passphrase)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("auth: passphrase rotated for profile=%d", profile_id)
    return JSONResponse({"ok": True, "profile_id": profile_id})
