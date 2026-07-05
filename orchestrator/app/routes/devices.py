"""Profile-device pairing endpoints (#50).

REST API for the Settings → My Devices panel and for the 6-digit
pairing code flow that lets a desktop-agent link itself to a profile
without a full Settings UI.

Endpoints
─────────
GET  /api/devices              — list all devices for the authed profile
POST /api/devices              — add / update a device pairing directly
DELETE /api/devices/{id}       — remove a device pairing

POST /api/devices/pair-code    — generate a 6-digit pairing code
                                 body: {"device_kind": "macos_agent"}
                                 Returns: {"code": "...", "expires_in": 300}

All endpoints require the ``va_session`` cookie (same auth as /api/me).
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import desktop_client
from ..storage.profile_devices import (
    create_pairing_code,
    delete_device,
    list_devices,
    upsert_device,
)
from ._deps import _current_user

router = APIRouter()


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("/api/devices")
async def api_list_devices(user: dict = Depends(_current_user)) -> JSONResponse:
    """Return all devices paired with the current profile.

    Response shape:
        {
          "devices": [
            {
              "id": 1,
              "device_kind": "macos_agent",
              "device_uid": "macbook-home",
              "friendly_name": "Mom's MacBook",
              "is_default": true,
              "wol_configured": false,
              "last_seen_at": <unix ts or null>,
              "online": true,    // last_seen within 90 s
              "locked": false    // null if unknown
            },
            ...
          ]
        }
    """
    profile_id: int = user["profile_id"]
    rows = await list_devices(profile_id)

    now = time.time()
    out = []
    for row in rows:
        agent_uid = row["device_uid"]
        agent = desktop_client.get_agent(agent_uid)
        last_seen = row.get("last_seen_at")
        # An agent is "online" when its desktop_client entry has a recent
        # last_seen timestamp (heartbeat / poll keeps this warm).
        if agent is not None:
            online = agent.reachable or (
                agent.last_seen > 0 and (now - agent.last_seen) < 90
            )
            locked = agent.locked
        else:
            online = False
            locked = None
        out.append({
            "id": row["id"],
            "device_kind": row["device_kind"],
            "device_uid": agent_uid,
            "friendly_name": row["friendly_name"],
            "is_default": bool(row["is_default"]),
            "wol_configured": bool(row.get("wol_mac")),
            "last_seen_at": last_seen,
            "online": online,
            "locked": locked,
        })
    return JSONResponse({"devices": out})


# ---------------------------------------------------------------------------
# Add / update
# ---------------------------------------------------------------------------

class AddDeviceRequest(BaseModel):
    device_uid: str          # = AgentInfo.agent_id
    device_kind: str         # "macos_agent", "linux_agent", etc.
    friendly_name: str = ""
    is_default: bool = False
    wol_mac: str | None = None
    wol_target_ip: str | None = None


@router.post("/api/devices")
async def api_add_device(
    req: AddDeviceRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Directly pair a device with the current profile.

    ``device_uid`` must match the ``agent_id`` the desktop-agent uses
    when connecting via reverse-WSS (set via the ``AGENT_ID`` env on the
    agent side).  Use this endpoint for admin/scripted setups; for
    user-friendly pairing use the ``pair-code`` flow instead.
    """
    profile_id: int = user["profile_id"]
    if not req.device_uid.strip():
        raise HTTPException(status_code=400, detail="device_uid must not be empty")
    if not req.device_kind.strip():
        raise HTTPException(status_code=400, detail="device_kind must not be empty")

    row_id = await upsert_device(
        profile_id=profile_id,
        device_uid=req.device_uid.strip(),
        device_kind=req.device_kind.strip(),
        friendly_name=req.friendly_name.strip(),
        is_default=req.is_default,
        wol_mac=req.wol_mac or None,
        wol_target_ip=req.wol_target_ip or None,
    )
    return JSONResponse({"ok": True, "id": row_id})


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/api/devices/{device_id}")
async def api_delete_device(
    device_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Un-pair a device from the current profile."""
    profile_id: int = user["profile_id"]
    deleted = await delete_device(device_id, profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Device not found")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Pairing code
# ---------------------------------------------------------------------------

class PairCodeRequest(BaseModel):
    device_kind: str = "macos_agent"


@router.post("/api/devices/pair-code")
async def api_pair_code(
    req: PairCodeRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Generate a 6-digit pairing code for the current profile.

    The code expires in 5 minutes and is single-use.

    Flow:
      1. Call this endpoint — note the 6-digit code.
      2. Set ``PAIRING_CODE=<code>`` in the desktop-agent's environment.
      3. Start (or restart) the desktop-agent — it will send a ``pair``
         frame over reverse-WSS which the orchestrator resolves to this
         profile.

    Alternatively, the voice assistant can speak the code aloud so the
    user can configure the agent without opening a browser.

    Response: {"code": "123456", "expires_in": 300, "device_kind": "..."}
    """
    profile_id: int = user["profile_id"]
    device_kind = req.device_kind.strip() or "macos_agent"
    code = await create_pairing_code(profile_id, device_kind)
    return JSONResponse({
        "code": code,
        "expires_in": 300,  # seconds
        "device_kind": device_kind,
    })
