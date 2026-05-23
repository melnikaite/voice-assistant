"""
Pending-action queue — deferred invasive tool calls.

A ``high_write`` tool can be "armed" by the LLM even when the speaker
hasn't said the passphrase in this turn: instead of executing, the
agent loop calls ``enqueue_action()`` and tells the speaker the action
was deferred.  When the speaker later says the passphrase (or clicks
"Approve" in the UI), the agent replays the queue.

Status transitions:
  pending  → approved   (user/voice/auto)
  pending  → rejected   (user explicitly rejected)
  pending  → expired    (TTL hit; swept periodically)

Default TTL is 24 hours — short enough to bound the queue, long enough
that you can deal with it after the dinner guests leave.
"""
from __future__ import annotations

import asyncio
import json
import time

from .db import _conn, _lock

# Default time-to-live for a queued action.  Override per-action with
# the ``ttl_s`` arg to ``enqueue_action``.
DEFAULT_TTL_S = 24 * 3600


# ── Insert ──────────────────────────────────────────────────────────────


def _enqueue_sync(
    profile_id: int | None,
    client_id: str | None,
    tool_name: str,
    tool_args: dict,
    summary: str,
    ttl_s: float,
) -> int:
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                """
                INSERT INTO pending_actions
                  (profile_id, client_id, tool_name, tool_args, summary,
                   requested_at, expires_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    profile_id,
                    client_id,
                    tool_name,
                    json.dumps(tool_args, ensure_ascii=False),
                    summary,
                    now,
                    now + ttl_s,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            c.close()


async def enqueue_action(
    *,
    profile_id: int | None,
    client_id: str | None,
    tool_name: str,
    tool_args: dict,
    summary: str,
    ttl_s: float = DEFAULT_TTL_S,
) -> int:
    """Queue an invasive tool call for later approval. Returns its id."""
    return await asyncio.to_thread(
        _enqueue_sync, profile_id, client_id, tool_name, tool_args, summary, ttl_s
    )


# ── Read ────────────────────────────────────────────────────────────────


def _list_pending_sync(
    profile_id: int | None, client_id: str | None
) -> list[tuple]:
    """Return all still-pending actions for the (profile, client) pair.

    ``profile_id`` is the strong identity (resemblyzer-matched speaker);
    ``client_id`` is the per-browser stable ID.  We OR them so an
    unidentified speaker can still see their own queued actions tied
    to this browser, and a speaker who moved devices can still see
    actions tied to their profile.

    Stale-row safety: we filter by ``expires_at > now`` here so that
    a row whose TTL just expired won't appear even if the periodic
    sweep (``gc._gc_pending_expired``) hasn't run yet — keeps the UI
    honest between GC ticks.  The status UPDATE itself was moved off
    the read path to ``_sweep_expired_sync`` so authenticated turns
    no longer write on every read.
    """
    now = time.time()
    with _lock:
        c = _conn()
        try:
            where = ["status='pending'", "expires_at > ?"]
            params: list = [now]
            id_clauses = []
            if profile_id is not None:
                id_clauses.append("profile_id=?")
                params.append(profile_id)
            if client_id is not None:
                id_clauses.append("client_id=?")
                params.append(client_id)
            if id_clauses:
                where.append("(" + " OR ".join(id_clauses) + ")")
            rows = c.execute(
                "SELECT id, profile_id, client_id, tool_name, tool_args,"
                " summary, requested_at, expires_at"
                f" FROM pending_actions WHERE {' AND '.join(where)}"
                " ORDER BY requested_at",
                tuple(params),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


# ── GC helpers (called from gc.py on a timer) ──────────────────────────


def _sweep_expired_sync() -> int:
    """Flip ``pending`` rows past their TTL to ``expired``.

    Used to run inside ``_list_pending_sync`` on every authenticated
    turn — that meant a write+commit through the global lock per
    voice turn even when nothing had expired.  Now runs on the
    periodic GC tick in ``gc.py``.  Returns rowcount for log output.
    """
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "UPDATE pending_actions SET status='expired'"
                " WHERE status='pending' AND expires_at <= ?",
                (now,),
            )
            return cur.rowcount
        finally:
            c.close()


def _purge_terminal_sync(cutoff_ts: float) -> int:
    """Delete pending_actions rows in terminal state older than ``cutoff_ts``.

    Terminal = `executed | execution_failed | rejected | expired`.
    The UI's "recent" panel caps at 20, so older rows are pure
    history with no surface for the user.  Retention window comes
    from ``gc.PENDING_TERMINAL_RETENTION_DAYS``.
    """
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM pending_actions"
                " WHERE status IN ('executed','execution_failed','rejected','expired')"
                "   AND COALESCE(approved_at, requested_at) < ?",
                (cutoff_ts,),
            )
            return cur.rowcount
        finally:
            c.close()


async def list_pending_actions(
    *,
    profile_id: int | None = None,
    client_id: str | None = None,
) -> list[dict]:
    """Return still-pending actions as dicts (UI- and LLM-friendly)."""
    rows = await asyncio.to_thread(_list_pending_sync, profile_id, client_id)
    out: list[dict] = []
    for r in rows:
        out.append({
            "id": r[0],
            "profile_id": r[1],
            "client_id": r[2],
            "tool_name": r[3],
            "tool_args": json.loads(r[4]) if r[4] else {},
            "summary": r[5],
            "requested_at": r[6],
            "expires_at": r[7],
        })
    return out


def _get_one_sync(action_id: int) -> tuple | None:
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT id, profile_id, client_id, tool_name, tool_args,"
                " summary, requested_at, expires_at, status"
                " FROM pending_actions WHERE id=?",
                (action_id,),
            ).fetchone()
            return row  # type: ignore[return-value]
        finally:
            c.close()


async def get_pending_action(action_id: int) -> dict | None:
    row = await asyncio.to_thread(_get_one_sync, action_id)
    if row is None:
        return None
    return {
        "id": row[0],
        "profile_id": row[1],
        "client_id": row[2],
        "tool_name": row[3],
        "tool_args": json.loads(row[4]) if row[4] else {},
        "summary": row[5],
        "requested_at": row[6],
        "expires_at": row[7],
        "status": row[8],
    }


# ── Mutate ──────────────────────────────────────────────────────────────


def _set_status_sync(action_id: int, status: str, via: str) -> bool:
    """Move an action out of pending. Returns True if the row was eligible."""
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "UPDATE pending_actions SET status=?, approved_via=?, approved_at=?"
                " WHERE id=? AND status='pending'",
                (status, via, now, action_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def mark_approved(action_id: int, via: str = "voice") -> bool:
    """Approve a pending action. Returns False if it wasn't pending anymore."""
    return await asyncio.to_thread(_set_status_sync, action_id, "approved", via)


async def mark_rejected(action_id: int, via: str = "voice") -> bool:
    """Reject a pending action."""
    return await asyncio.to_thread(_set_status_sync, action_id, "rejected", via)


# ── Executor surface (pending_executor.py) ─────────────────────────────


def _list_approved_sync() -> list[tuple]:
    """Return ``approved`` rows ready to be dispatched.  Caller is
    responsible for advancing them to ``executed`` (or
    ``execution_failed``) afterwards via mark_executed()."""
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, profile_id, client_id, tool_name, tool_args, summary"
                " FROM pending_actions"
                " WHERE status='approved'"
                " ORDER BY approved_at"
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def list_approved_actions() -> list[dict]:
    rows = await asyncio.to_thread(_list_approved_sync)
    return [
        {
            "id": r[0],
            "profile_id": r[1],
            "client_id": r[2],
            "tool_name": r[3],
            "tool_args": json.loads(r[4]) if r[4] else {},
            "summary": r[5],
        }
        for r in rows
    ]


def _mark_executed_sync(action_id: int, ok: bool, summary_suffix: str | None) -> None:
    new_status = "executed" if ok else "execution_failed"
    now = time.time()
    with _lock:
        c = _conn()
        try:
            if summary_suffix:
                c.execute(
                    "UPDATE pending_actions SET status=?, approved_at=?, summary=summary || ' · ' || ?"
                    " WHERE id=?",
                    (new_status, now, summary_suffix, action_id),
                )
            else:
                c.execute(
                    "UPDATE pending_actions SET status=?, approved_at=?"
                    " WHERE id=?",
                    (new_status, now, action_id),
                )
        finally:
            c.close()


async def mark_executed(action_id: int, ok: bool, note: str | None = None) -> None:
    """Move an action from ``approved`` to ``executed`` / ``execution_failed``.

    ``note`` is appended to the row's ``summary`` so the UI can later
    show a one-line outcome line (e.g. "Created event · OK" or
    "Deleted reminder · failure: ...").
    """
    await asyncio.to_thread(_mark_executed_sync, action_id, ok, note)


# ── Recent (terminal-status) read — UI surface ─────────────────────────


_TERMINAL_STATUSES = ("executed", "execution_failed", "rejected", "expired")


def _list_recent_sync(
    profile_id: int | None,
    client_id: str | None,
    limit: int,
) -> list[tuple]:
    """Return the most-recently-finalised actions across all terminal states.

    Ordered by ``approved_at DESC`` (which we also stamp on
    ``execution_failed``/``expired`` since the column is overloaded as
    "finalisation time" — keeping schema flat).  Limit defaults to 20
    so the UI can show a small "Recent" panel without paginating.
    """
    with _lock:
        c = _conn()
        try:
            where = [f"status IN ({','.join(['?']*len(_TERMINAL_STATUSES))})"]
            params: list = list(_TERMINAL_STATUSES)
            id_clauses = []
            if profile_id is not None:
                id_clauses.append("profile_id=?")
                params.append(profile_id)
            if client_id is not None:
                id_clauses.append("client_id=?")
                params.append(client_id)
            if id_clauses:
                where.append("(" + " OR ".join(id_clauses) + ")")
            rows = c.execute(
                "SELECT id, profile_id, client_id, tool_name, tool_args, summary, "
                "       requested_at, expires_at, status, approved_via, approved_at "
                f"  FROM pending_actions WHERE {' AND '.join(where)}"
                "  ORDER BY COALESCE(approved_at, requested_at) DESC"
                "  LIMIT ?",
                tuple(params) + (limit,),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def list_recent_actions(
    *,
    profile_id: int | None = None,
    client_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return recently-finalised actions (executed / failed / rejected / expired)."""
    rows = await asyncio.to_thread(_list_recent_sync, profile_id, client_id, limit)
    return [
        {
            "id": r[0],
            "profile_id": r[1],
            "client_id": r[2],
            "tool_name": r[3],
            "tool_args": json.loads(r[4]) if r[4] else {},
            "summary": r[5],
            "requested_at": r[6],
            "expires_at": r[7],
            "status": r[8],
            "approved_via": r[9],
            "approved_at": r[10],
        }
        for r in rows
    ]
