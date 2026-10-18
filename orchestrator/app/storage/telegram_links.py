"""Telegram account ↔ speaker_profile links + single-use link tokens.

Flow (mirrors ``device_pairing_codes`` in ``profile_devices.py``):

1.  The logged-in web UI asks for a token → :func:`create_telegram_link_token`
    stores a CSPRNG token bound to that profile with a 5-minute TTL.
2.  The user opens ``t.me/<bot>?start=<token>``; the bot receives
    ``/start <token>`` and calls :func:`consume_telegram_link_token`,
    which marks the token used and writes the ``telegram_links`` row.

One Telegram account maps to exactly one profile and a profile has at
most one linked account: re-linking from either side replaces the old row.
"""
from __future__ import annotations

import asyncio
import secrets
import time

from .db import _conn

LINK_TOKEN_TTL_S = 5 * 60


def _profile_row(c, profile_id: int) -> tuple | None:
    return c.execute(
        "SELECT id, name, client_id FROM speaker_profiles WHERE id=?",
        (profile_id,),
    ).fetchone()


def _link_dict(row: tuple | None) -> dict | None:
    if row is None:
        return None
    return {
        "telegram_user_id": int(row[0]),
        "chat_id": int(row[1]),
        "profile_id": int(row[2]),
        "telegram_name": row[3] or "",
        "linked_at": float(row[4]),
        "profile_name": row[5],
        "profile_client_id": row[6],
    }


_LINK_SELECT = (
    "SELECT l.telegram_user_id, l.chat_id, l.profile_id, l.telegram_name,"
    " l.linked_at, p.name, p.client_id"
    " FROM telegram_links l JOIN speaker_profiles p ON p.id = l.profile_id"
)


# ── Tokens ──────────────────────────────────────────────────────────────


def _create_token_sync(profile_id: int) -> str:
    c = _conn()
    now = time.time()
    c.execute("DELETE FROM telegram_link_tokens WHERE expires_at < ?", (now,))
    # A fresh token invalidates any earlier unused one for this profile so
    # only the link currently on screen works.
    c.execute(
        "DELETE FROM telegram_link_tokens WHERE profile_id=? AND used=0",
        (profile_id,),
    )
    # 128 bits from the CSPRNG; token_urlsafe's alphabet (A-Za-z0-9_-)
    # is exactly what Telegram allows in a ``start`` deep-link parameter.
    token = secrets.token_urlsafe(16)
    c.execute(
        "INSERT INTO telegram_link_tokens (token, profile_id, expires_at, used)"
        " VALUES (?, ?, ?, 0)",
        (token, profile_id, now + LINK_TOKEN_TTL_S),
    )
    return token


async def create_telegram_link_token(profile_id: int) -> str:
    """Mint a single-use link token for ``profile_id`` (5-min TTL)."""
    return await asyncio.to_thread(_create_token_sync, profile_id)


def _consume_token_sync(
    token: str, telegram_user_id: int, chat_id: int, telegram_name: str
) -> dict | None:
    c = _conn()
    row = c.execute(
        "SELECT profile_id FROM telegram_link_tokens"
        " WHERE token=? AND used=0 AND expires_at>?",
        (token, time.time()),
    ).fetchone()
    if row is None:
        return None
    profile_id = int(row[0])
    # The conditional UPDATE is the single-use guard: of two concurrent
    # consumers only one sees rowcount == 1.
    cur = c.execute(
        "UPDATE telegram_link_tokens SET used=1 WHERE token=? AND used=0",
        (token,),
    )
    if cur.rowcount != 1:
        return None
    if _profile_row(c, profile_id) is None:
        return None  # profile deleted after the token was minted
    c.execute(
        "DELETE FROM telegram_links WHERE telegram_user_id=? OR profile_id=?",
        (telegram_user_id, profile_id),
    )
    c.execute(
        "INSERT INTO telegram_links"
        " (telegram_user_id, chat_id, profile_id, telegram_name, linked_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (telegram_user_id, chat_id, profile_id, telegram_name or "", time.time()),
    )
    return _link_dict(
        c.execute(_LINK_SELECT + " WHERE l.telegram_user_id=?", (telegram_user_id,)).fetchone()
    )


async def consume_telegram_link_token(
    token: str, telegram_user_id: int, chat_id: int, telegram_name: str = ""
) -> dict | None:
    """Consume ``token`` and link the Telegram account to its profile.

    Returns the new link (see :func:`get_telegram_link`) or ``None`` when
    the token is unknown, expired or already used.
    """
    return await asyncio.to_thread(
        _consume_token_sync, token, telegram_user_id, chat_id, telegram_name
    )


# ── Links ───────────────────────────────────────────────────────────────


def _get_by_user_sync(telegram_user_id: int) -> dict | None:
    return _link_dict(
        _conn().execute(_LINK_SELECT + " WHERE l.telegram_user_id=?", (telegram_user_id,)).fetchone()
    )


async def get_telegram_link(telegram_user_id: int) -> dict | None:
    """Link row for a Telegram user, joined with the profile name/client_id."""
    return await asyncio.to_thread(_get_by_user_sync, telegram_user_id)


def _get_by_chat_sync(chat_id: int) -> dict | None:
    return _link_dict(
        _conn().execute(_LINK_SELECT + " WHERE l.chat_id=?", (chat_id,)).fetchone()
    )


async def get_telegram_link_by_chat(chat_id: int) -> dict | None:
    return await asyncio.to_thread(_get_by_chat_sync, chat_id)


def _get_by_profile_sync(profile_id: int) -> dict | None:
    return _link_dict(
        _conn().execute(_LINK_SELECT + " WHERE l.profile_id=?", (profile_id,)).fetchone()
    )


async def get_telegram_link_for_profile(profile_id: int) -> dict | None:
    return await asyncio.to_thread(_get_by_profile_sync, profile_id)


def _list_sync() -> list[dict]:
    return [_link_dict(r) for r in _conn().execute(_LINK_SELECT).fetchall()]  # type: ignore[misc]


async def list_telegram_links() -> list[dict]:
    return await asyncio.to_thread(_list_sync)


def _delete_for_profile_sync(profile_id: int) -> bool:
    cur = _conn().execute("DELETE FROM telegram_links WHERE profile_id=?", (profile_id,))
    return cur.rowcount > 0


async def delete_telegram_link_for_profile(profile_id: int) -> bool:
    """Unlink whatever Telegram account is linked to ``profile_id``."""
    return await asyncio.to_thread(_delete_for_profile_sync, profile_id)
