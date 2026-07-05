"""Step-up auth grants — push-to-device approval for private tools.

When the LLM requests a ``private=True`` tool and the session has no
active step-up grant, the pipeline:
  1. Creates a grant row here (token + 90-second TTL).
  2. Sends a Web Push to the speaker's registered devices with that token.
  3. Returns a voice reply "tap your device to confirm".

The user taps the push notification → the SW does a background fetch to
``POST /api/step-up/approve`` → orchestrator marks the grant consumed and
broadcasts a ``step_up_granted`` WS event → the open browser tab sets a
session-level flag so the next voice turn sees ``step_up_auth=True`` and
executes the tool.

This module owns only the DB side.  The HTTP route + WS broadcast live in
``routes/step_up.py`` and ``registry.py``.
"""
from __future__ import annotations

import asyncio
import secrets
import time

from .db import _conn, _lock

# Default grant lifetime — long enough for the user to see the
# notification, tap it, and re-ask; short enough that a stale approval
# can't be replayed hours later.  90 seconds is the sweet spot.
GRANT_TTL_S: int = 90


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _create_grant_sync(profile_id: int, client_id: str | None) -> str:
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _lock:
        c = _conn()
        try:
            c.execute(
                "INSERT INTO step_up_grants(token, profile_id, client_id, issued_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (token, profile_id, client_id, now, now + GRANT_TTL_S),
            )
        finally:
            c.close()
    return token


async def create_grant(profile_id: int, client_id: str | None = None) -> str:
    """Create a fresh step-up token and return it.

    The token is a 32-char URL-safe base64 string.  It expires in
    :data:`GRANT_TTL_S` seconds.  A single profile may have multiple
    live grants (e.g. two concurrent sessions), which is fine — any
    valid token approves the requesting session.
    """
    return await asyncio.to_thread(_create_grant_sync, profile_id, client_id)


def _consume_grant_sync(token: str) -> tuple[int, str | None] | None:
    """Mark a grant consumed; return ``(profile_id, client_id)`` or None.

    A grant is invalid when:
      * the token doesn't exist, OR
      * it has expired (``expires_at`` in the past).

    Consumed grants are deleted immediately — we don't keep a trail
    here because the WS session's ``_step_up_auth_until`` field records
    the approval on the in-memory side for the session's lifetime.

    Returns the stored ``(profile_id, client_id)`` so the caller can
    elevate ONLY the session that requested the grant — a leaked token
    must not be replayable against an arbitrary attacker-named session.
    """
    now = time.time()
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT profile_id, client_id FROM step_up_grants"
                " WHERE token=? AND expires_at>?",
                (token, now),
            ).fetchone()
            if not row:
                return None
            c.execute("DELETE FROM step_up_grants WHERE token=?", (token,))
            return (int(row[0]), row[1])
        finally:
            c.close()


async def consume_grant(token: str) -> tuple[int, str | None] | None:
    """Validate + delete a grant.

    Returns ``(profile_id, client_id)`` on success, None on invalid/expired.
    The ``client_id`` is the session that originally requested the grant;
    only that session should be elevated.
    """
    return await asyncio.to_thread(_consume_grant_sync, token)


def _purge_expired_sync() -> int:
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM step_up_grants WHERE expires_at <= ?", (now,)
            )
            return cur.rowcount
        finally:
            c.close()


async def purge_expired_grants() -> int:
    """GC expired grants.  Called from the periodic scheduler."""
    return await asyncio.to_thread(_purge_expired_sync)
