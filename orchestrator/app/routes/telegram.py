"""Telegram account linking for the Settings panel.

GET    /api/telegram             — feature state + the current profile's link
POST   /api/telegram/link-token  — mint a 5-min single-use token + t.me deep link
DELETE /api/telegram/link        — unlink the current profile's Telegram account

All endpoints require the ``va_session`` cookie and act on that profile only.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .. import telegram
from ..storage import (
    create_telegram_link_token,
    delete_telegram_link_for_profile,
    get_telegram_link_for_profile,
)
from ..storage.telegram_links import LINK_TOKEN_TTL_S
from ._deps import _current_user

router = APIRouter()


@router.get("/api/telegram")
async def api_telegram_status(user: dict = Depends(_current_user)) -> JSONResponse:
    link = await get_telegram_link_for_profile(user["profile_id"])
    return JSONResponse({
        "enabled": telegram.enabled(),
        "bot_username": telegram.bot_username(),
        "link": (
            {"telegram_name": link["telegram_name"], "linked_at": link["linked_at"]}
            if link else None
        ),
    })


@router.post("/api/telegram/link-token")
async def api_telegram_link_token(user: dict = Depends(_current_user)) -> JSONResponse:
    username = telegram.bot_username()
    if not telegram.enabled() or not username:
        raise HTTPException(503, "telegram bot not configured or not reachable")
    token = await create_telegram_link_token(user["profile_id"])
    return JSONResponse({
        "token": token,
        "url": f"https://t.me/{username}?start={token}",
        "bot_username": username,
        "expires_in": LINK_TOKEN_TTL_S,
    })


@router.delete("/api/telegram/link")
async def api_telegram_unlink(user: dict = Depends(_current_user)) -> JSONResponse:
    removed = await delete_telegram_link_for_profile(user["profile_id"])
    return JSONResponse({"ok": True, "removed": removed})
