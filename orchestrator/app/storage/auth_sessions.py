"""
auth_sessions — server-side cookie session store for the UI.

Why server-side instead of signed JWT-style cookies:
  • Revocation is trivial (DELETE WHERE token=?).
  • The DB is on the same machine as the orchestrator, so the round
    trip is negligible.
  • The table stays small — one row per active browser tab.  Read
    paths filter on ``expires_at > now`` so stale rows never surface;
    ``sweep_expired_sessions()`` exists for ops/manual cleanup but no
    periodic sweeper runs (one user, table never grows).

Token shape: 32-byte ``secrets.token_urlsafe(24)`` — ~96 bits of
entropy, URL-safe, no padding chars.
"""
from __future__ import annotations

import asyncio
import secrets
import time

from .db import _conn, _lock

# How long an auth-session cookie is valid before re-login is required.
# Sliding renewal is intentionally NOT done — that lets a stolen cookie
# live forever as long as the thief keeps using it.  Hard cap = re-login.
DEFAULT_SESSION_TTL_S = 30 * 86400  # 30 days


def _new_token() -> str:
    return secrets.token_urlsafe(24)


def _create_sync(profile_id: int, ttl_s: float, user_agent: str | None) -> str:
    token = _new_token()
    now = time.time()
    with _lock:
        c = _conn()
        try:
            c.execute(
                "INSERT INTO auth_sessions(token, profile_id, created_at, expires_at, user_agent)"
                " VALUES (?, ?, ?, ?, ?)",
                (token, profile_id, now, now + ttl_s, user_agent),
            )
            return token
        finally:
            c.close()


async def create_session(
    profile_id: int,
    *,
    ttl_s: float = DEFAULT_SESSION_TTL_S,
    user_agent: str | None = None,
) -> str:
    """Mint a fresh session token for ``profile_id`` and return it."""
    return await asyncio.to_thread(_create_sync, profile_id, ttl_s, user_agent)


def _get_sync(token: str) -> tuple[int, float] | None:
    now = time.time()
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT profile_id, expires_at FROM auth_sessions"
                " WHERE token=? AND expires_at > ?",
                (token, now),
            ).fetchone()
            return row  # type: ignore[return-value]
        finally:
            c.close()


async def get_session(token: str) -> dict | None:
    """Return ``{profile_id, expires_at}`` for a live session, else None."""
    row = await asyncio.to_thread(_get_sync, token)
    if row is None:
        return None
    return {"profile_id": row[0], "expires_at": row[1]}


def _revoke_sync(token: str) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute("DELETE FROM auth_sessions WHERE token=?", (token,))
        finally:
            c.close()


async def revoke_session(token: str) -> None:
    await asyncio.to_thread(_revoke_sync, token)


def _sweep_sync() -> int:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?", (time.time(),)
            )
            return cur.rowcount
        finally:
            c.close()


async def sweep_expired_sessions() -> int:
    """Background-safe GC.  Returns number of rows deleted."""
    return await asyncio.to_thread(_sweep_sync)
