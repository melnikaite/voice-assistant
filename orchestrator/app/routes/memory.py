from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..user_files import (
    UserSettings,
    read_memory,
    read_settings,
    write_memory,
)
from ._deps import _current_user

router = APIRouter()


@router.get("/api/users/{profile_id}/memory")
async def api_read_memory(
    profile_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    body = await read_memory(profile_id)
    return JSONResponse({"profile_id": profile_id, "memory": body})


class MemoryWriteRequest(BaseModel):
    content: str  # full replacement body


@router.put("/api/users/{profile_id}/memory")
async def api_write_memory(
    profile_id: int,
    body: MemoryWriteRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    await write_memory(profile_id, body.content)
    return JSONResponse({"ok": True, "profile_id": profile_id})


@router.get("/api/users/{profile_id}/settings")
async def api_read_settings(
    profile_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    s = await read_settings(profile_id)
    return JSONResponse({"profile_id": profile_id, "settings": s.model_dump()})


@router.put("/api/users/{profile_id}/settings")
async def api_write_settings(
    profile_id: int,
    body: UserSettings,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Replace the typed settings with the supplied UserSettings.

    Note: ``code_word_hash`` is intentionally excluded from this flow
    — a UI that ever sends it would let a logged-in user overwrite
    their own hash with garbage and lock themselves out.  Passphrase
    rotation goes through the dedicated ``/api/auth/rotate_passphrase``
    endpoint below.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    safe = body.model_copy(update={"code_word_hash": (await read_settings(profile_id)).code_word_hash})
    from ..user_files import write_settings as _ws
    await _ws(profile_id, safe)
    return JSONResponse({"ok": True, "settings": safe.model_dump()})
