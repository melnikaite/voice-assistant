"""wwyd connection for the Settings panel (one per profile).

GET    /api/wwyd                 — connection state, audiences, asker mapping
POST   /api/wwyd/connect         — start OAuth (DCR + PKCE); returns the authorize URL
GET    /api/wwyd/callback        — OAuth redirect target; stores the tokens
POST   /api/wwyd/api-key         — connect with a pasted API key instead
PUT    /api/wwyd/audiences       — asker → wwyd audience mapping
DELETE /api/wwyd                 — revoke at wwyd (best effort) and forget locally

All endpoints act on the cookie session's profile.  Tokens never leave
the server: responses carry the URL, auth kind and mapping only.
"""
from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .. import wwyd
from ..storage import (
    connection_public,
    delete_wwyd_connection,
    get_all_speaker_profiles,
    get_wwyd_connection,
    pop_wwyd_pending,
    put_wwyd_pending,
    save_wwyd_connection,
    set_wwyd_audience_map,
)
from ._deps import _current_user

log = logging.getLogger(__name__)
router = APIRouter()

GUEST_KEY = "guest"


class ConnectBody(BaseModel):
    url: str = Field(max_length=500)


class ApiKeyBody(BaseModel):
    url: str = Field(max_length=500)
    key: str = Field(min_length=10, max_length=200)


class AudienceMapBody(BaseModel):
    audience_map: dict[str, str]


async def _askers(profile_id: int) -> list[dict]:
    """Who can ask about this profile: other household profiles (voice or
    their linked Telegram) and unrecognised guests."""
    rows = await get_all_speaker_profiles()
    out = [{"key": f"profile:{r[0]}", "name": r[1]} for r in rows if r[0] != profile_id]
    out.append({"key": GUEST_KEY, "name": None})
    return out


@router.get("/api/wwyd")
async def api_wwyd_status(user: dict = Depends(_current_user)) -> JSONResponse:
    pid = user["profile_id"]
    conn = await get_wwyd_connection(pid)
    body: dict = {"connection": connection_public(conn), "askers": await _askers(pid)}
    if conn is not None:
        try:
            body["audiences"] = await wwyd.list_audiences(pid)
        except wwyd.WwydError as exc:
            body["audiences"] = []
            body["error"] = exc.code
    return JSONResponse(body)


@router.post("/api/wwyd/connect")
async def api_wwyd_connect(
    body: ConnectBody, request: Request, user: dict = Depends(_current_user)
) -> JSONResponse:
    try:
        base = wwyd.normalize_base_url(body.url)
        meta = await wwyd.discover(base)
        redirect_uri = str(request.url_for("api_wwyd_callback"))
        client_id = await wwyd.register_client(meta["registration_endpoint"], redirect_uri)
    except wwyd.WwydError as exc:
        raise HTTPException(400 if exc.code == "invalid" else 502, str(exc)) from None
    verifier, challenge = wwyd.pkce_pair()
    state = secrets.token_urlsafe(24)
    await put_wwyd_pending(
        state, user["profile_id"], base_url=base, resource=meta["resource"],
        client_id=client_id, code_verifier=verifier, redirect_uri=redirect_uri,
        token_endpoint=meta["token_endpoint"], revocation_endpoint=meta["revocation_endpoint"],
    )
    return JSONResponse({
        "authorize_url": wwyd.authorize_url(meta, client_id, redirect_uri, state, challenge),
    })


@router.get("/api/wwyd/callback", name="api_wwyd_callback")
async def api_wwyd_callback(
    state: str = "", code: str = "", error: str = "", user: dict = Depends(_current_user)
) -> RedirectResponse:
    pending = await pop_wwyd_pending(state) if state else None
    # The state binds the redirect to the profile that started it.
    if pending is None or pending["profile_id"] != user["profile_id"]:
        return RedirectResponse("/?wwyd=invalid", status_code=303)
    if error or not code:
        return RedirectResponse("/?wwyd=denied", status_code=303)
    try:
        tokens = await wwyd.exchange_code(pending, code)
    except wwyd.WwydError as exc:
        log.warning("wwyd: code exchange failed for profile %d: %s", user["profile_id"], exc.code)
        return RedirectResponse("/?wwyd=failed", status_code=303)
    await save_wwyd_connection(
        user["profile_id"], base_url=pending["base_url"], resource=pending["resource"],
        auth_kind="oauth", client_id=pending["client_id"],
        token_endpoint=pending["token_endpoint"],
        revocation_endpoint=pending["revocation_endpoint"], **tokens,
    )
    log.info("wwyd: profile %d connected via OAuth", user["profile_id"])
    return RedirectResponse("/?wwyd=connected", status_code=303)


@router.post("/api/wwyd/api-key")
async def api_wwyd_api_key(body: ApiKeyBody, user: dict = Depends(_current_user)) -> JSONResponse:
    try:
        base = wwyd.normalize_base_url(body.url)
        resource = f"{base}/mcp"
        tools = await wwyd.probe_key(resource, body.key.strip())
    except wwyd.WwydError as exc:
        status = 400 if exc.code in ("invalid", "auth") else 502
        raise HTTPException(status, "wwyd refused the key" if exc.code == "auth" else str(exc)) from None
    # Ingest and add_claims need the owner scope.
    if not {"ingest_raw", "add_claims", "predict"} <= set(tools):
        raise HTTPException(400, "the key needs the owner scope")
    await save_wwyd_connection(
        user["profile_id"], base_url=base, resource=resource, auth_kind="api_key",
        access_token=body.key.strip(),
    )
    log.info("wwyd: profile %d connected via API key", user["profile_id"])
    return JSONResponse({"ok": True})


@router.put("/api/wwyd/audiences")
async def api_wwyd_audiences(
    body: AudienceMapBody, user: dict = Depends(_current_user)
) -> JSONResponse:
    pid = user["profile_id"]
    allowed = {a["key"] for a in await _askers(pid)}
    mapping = {k: v for k, v in body.audience_map.items() if k in allowed and v}
    if not await set_wwyd_audience_map(pid, mapping):
        raise HTTPException(404, "wwyd is not connected")
    return JSONResponse({"ok": True, "audience_map": mapping})


@router.delete("/api/wwyd")
async def api_wwyd_disconnect(user: dict = Depends(_current_user)) -> JSONResponse:
    pid = user["profile_id"]
    conn = await get_wwyd_connection(pid)
    revoked = await wwyd.revoke(conn) if conn else False
    removed = await delete_wwyd_connection(pid)
    return JSONResponse({"ok": True, "removed": removed, "revoked": revoked})
