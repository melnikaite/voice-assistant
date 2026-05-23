"""Reminders persistence — create, read, cancel, mark fired."""
from __future__ import annotations

import asyncio
import time

from .db import _conn, _lock


# ---------------------------------------------------------------------------
# Write / create
# ---------------------------------------------------------------------------

def _add_reminder_sync(client_id: str, fire_at: float, push_text: str) -> int:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "INSERT INTO reminders(client_id, fire_at, push_text, created_at)"
                " VALUES (?, ?, ?, ?)",
                (client_id, fire_at, push_text, time.time()),
            )
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            c.close()


async def add_reminder(client_id: str, fire_at: float, push_text: str) -> int:
    return await asyncio.to_thread(_add_reminder_sync, client_id, fire_at, push_text)


# ---------------------------------------------------------------------------
# Mark fired / delivered
# ---------------------------------------------------------------------------

def _mark_fired_sync(reminder_id: int, delivered: bool) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE reminders SET fired=1, delivered=? WHERE id=?",
                (1 if delivered else 0, reminder_id),
            )
        finally:
            c.close()


async def mark_reminder_fired(reminder_id: int, delivered: bool) -> None:
    await asyncio.to_thread(_mark_fired_sync, reminder_id, delivered)


async def mark_reminder_delivered(reminder_id: int) -> None:
    """Mark a missed reminder as successfully delivered after reconnect."""
    await asyncio.to_thread(_mark_fired_sync, reminder_id, True)


# ---------------------------------------------------------------------------
# Cancel (user-initiated)
# ---------------------------------------------------------------------------

def _cancel_reminder_sync(reminder_id: int, client_id: str) -> bool:
    """Delete a pending reminder.  Returns True if it existed and was deleted."""
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM reminders WHERE id=? AND client_id=? AND fired=0",
                (reminder_id, client_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def cancel_reminder_db(reminder_id: int, client_id: str) -> bool:
    """Remove a pending reminder from the DB.  Returns True if it was found."""
    return await asyncio.to_thread(_cancel_reminder_sync, reminder_id, client_id)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _get_pending_sync(client_id: str | None) -> list[tuple]:
    with _lock:
        c = _conn()
        try:
            if client_id:
                rows = c.execute(
                    "SELECT id, client_id, fire_at, push_text FROM reminders"
                    " WHERE fired=0 AND client_id=?"
                    " ORDER BY fire_at",
                    (client_id,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id, client_id, fire_at, push_text FROM reminders"
                    " WHERE fired=0 ORDER BY fire_at"
                ).fetchall()
            return rows
        finally:
            c.close()


async def get_pending_reminders(
    client_id: str | None = None,
) -> list[tuple[int, str, float, str]]:
    """Return (id, client_id, fire_at, push_text) for all un-fired reminders."""
    return await asyncio.to_thread(_get_pending_sync, client_id)  # type: ignore[return-value]


def _list_upcoming_sync(client_id: str) -> list[dict]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, fire_at, push_text FROM reminders"
                " WHERE client_id=? AND fired=0"
                " ORDER BY fire_at"
                " LIMIT 20",
                (client_id,),
            ).fetchall()
            return [
                {"id": row[0], "fire_at": row[1], "push_text": row[2]}
                for row in rows
            ]
        finally:
            c.close()


async def list_upcoming_reminders(client_id: str) -> list[dict]:
    """Return upcoming un-fired reminders for ``client_id`` as a list of dicts."""
    return await asyncio.to_thread(_list_upcoming_sync, client_id)


def _get_missed_sync(client_id: str) -> list[tuple[int, str]]:
    """Reminders that fired while the client was offline (fired=1, delivered=0)."""
    with _lock:
        c = _conn()
        try:
            since = time.time() - 86400  # last 24 h only
            rows = c.execute(
                "SELECT id, push_text FROM reminders"
                " WHERE client_id=? AND fired=1 AND delivered=0 AND fire_at > ?"
                " ORDER BY fire_at",
                (client_id, since),
            ).fetchall()
            return rows
        finally:
            c.close()


async def get_missed_reminders(client_id: str) -> list[tuple[int, str]]:
    """Return (id, push_text) for reminders that missed delivery while offline."""
    return await asyncio.to_thread(_get_missed_sync, client_id)
