"""
push_subscriptions — Web Push subscription store.

One row per browser/device that opted in to receive Web Push notifications
for a profile.  The ``endpoint`` URL is unique per subscription (the push
service mints it on subscribe), so we UPSERT on it: re-subscribing from
the same tab refreshes the keys and the timestamps without creating a
duplicate row.

Why server-side instead of stuffing the subscription into the browser's
local storage:
  • A user may log in from multiple devices — each needs its own row so
    we can fan out a single voicemail to all open tabs / phones.
  • On 410 Gone / 404 from the push service the orchestrator auto-deletes
    the row (see :func:`app.push.send_to_profile`).  That GC requires the
    subscription to be addressable server-side.

Thread-safety: same shape as ``voice_messages`` / ``auth_sessions`` — the
``_conn()`` proxy is per-thread, every sync helper opens it lazily, the
async wrapper offloads to ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import time

from .db import _conn, _lock


# ── Insert / upsert ────────────────────────────────────────────────────


def _upsert_sync(
    *,
    profile_id: int,
    endpoint: str,
    p256dh_key: str,
    auth_key: str,
    user_agent: str | None,
) -> int:
    """Insert a fresh subscription, or refresh the keys + timestamps if
    a row with the same ``endpoint`` already exists.

    Returns the row id.  The ON CONFLICT clause covers two scenarios:
      • Same browser re-subscribes after permission was revoked and
        granted again — endpoint stays stable; keys may rotate.
      • A second profile on the same device subscribes — the endpoint
        is browser-scoped, not profile-scoped, so we move the row's
        ``profile_id`` over to the new owner.  The previous owner just
        stops getting pushes on this device, which is the correct
        outcome (the device now belongs to someone else).
    """
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                """
                INSERT INTO push_subscriptions
                  (profile_id, endpoint, p256dh_key, auth_key, user_agent,
                   created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(endpoint) DO UPDATE SET
                  profile_id   = excluded.profile_id,
                  p256dh_key   = excluded.p256dh_key,
                  auth_key     = excluded.auth_key,
                  user_agent   = excluded.user_agent,
                  created_at   = excluded.created_at
                RETURNING id
                """,
                (profile_id, endpoint, p256dh_key, auth_key, user_agent, now),
            )
            row = cur.fetchone()
            return int(row[0])
        finally:
            c.close()


async def upsert_subscription(
    *,
    profile_id: int,
    endpoint: str,
    p256dh_key: str,
    auth_key: str,
    user_agent: str | None = None,
) -> int:
    """Persist a subscription, returning the row id.

    Idempotent — repeated calls with the same ``endpoint`` refresh the
    keys + timestamps without creating duplicates.
    """
    return await asyncio.to_thread(
        _upsert_sync,
        profile_id=profile_id,
        endpoint=endpoint,
        p256dh_key=p256dh_key,
        auth_key=auth_key,
        user_agent=user_agent,
    )


# ── Read ───────────────────────────────────────────────────────────────


def _row_to_dict(r: tuple) -> dict:
    return {
        "id": r[0],
        "profile_id": r[1],
        "endpoint": r[2],
        "p256dh_key": r[3],
        "auth_key": r[4],
        "user_agent": r[5],
        "created_at": r[6],
        "last_used_at": r[7],
    }


_SELECT_COLS = (
    "id, profile_id, endpoint, p256dh_key, auth_key, "
    "user_agent, created_at, last_used_at"
)


def _list_for_profile_sync(profile_id: int) -> list[tuple]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                f"SELECT {_SELECT_COLS} FROM push_subscriptions"
                "  WHERE profile_id=?"
                "  ORDER BY created_at DESC",
                (profile_id,),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def list_for_profile(profile_id: int) -> list[dict]:
    """Return every subscription registered for ``profile_id``, newest first."""
    rows = await asyncio.to_thread(_list_for_profile_sync, profile_id)
    return [_row_to_dict(r) for r in rows]


def _get_by_endpoint_sync(endpoint: str) -> tuple | None:
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                f"SELECT {_SELECT_COLS} FROM push_subscriptions WHERE endpoint=?",
                (endpoint,),
            ).fetchone()
            return row  # type: ignore[return-value]
        finally:
            c.close()


async def get_by_endpoint(endpoint: str) -> dict | None:
    """Fetch one subscription by its push-service endpoint URL.

    Returns ``None`` if the endpoint was never registered or has been
    auto-deleted after a 410 Gone from the push service.
    """
    row = await asyncio.to_thread(_get_by_endpoint_sync, endpoint)
    return _row_to_dict(row) if row else None


# ── Mutate ─────────────────────────────────────────────────────────────


def _delete_by_endpoint_sync(endpoint: str) -> bool:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM push_subscriptions WHERE endpoint=?",
                (endpoint,),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def delete_by_endpoint(endpoint: str) -> bool:
    """Drop a subscription by endpoint.  Returns True on first deletion.

    Called from two paths:
      • User-initiated unsubscribe (e.g. logout) — DELETE /api/push/subscribe.
      • Server-side GC when the push service returns 410 Gone / 404 (the
        subscription has been revoked by the browser or push provider).
    """
    return await asyncio.to_thread(_delete_by_endpoint_sync, endpoint)


def _touch_last_used_sync(endpoint: str) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE push_subscriptions SET last_used_at=? WHERE endpoint=?",
                (time.time(), endpoint),
            )
        finally:
            c.close()


async def touch_last_used(endpoint: str) -> None:
    """Stamp ``last_used_at`` after a successful send.  Best-effort."""
    await asyncio.to_thread(_touch_last_used_sync, endpoint)
