"""Session CRUD and conversation-history retrieval."""
from __future__ import annotations

import asyncio
import time

from .db import HISTORY_RESUME_MAX_AGE_S, _conn, _lock


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _start_session_sync(
    client: str | None,
    client_id: str | None,
    device_kind: str | None,
) -> int:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "INSERT INTO sessions(started_at, client, client_id, device_kind)"
                " VALUES (?, ?, ?, ?)",
                (time.time(), client, client_id, device_kind),
            )
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            c.close()


async def start_session(
    client: str | None = None,
    client_id: str | None = None,
    device_kind: str | None = None,
) -> int:
    return await asyncio.to_thread(_start_session_sync, client, client_id, device_kind)


# ---------------------------------------------------------------------------
# Read — conversation history
# ---------------------------------------------------------------------------

def _get_recent_history_sync(
    client_id: str, max_turns: int, since_ts: float
) -> list[tuple[str, str]]:
    """
    Return up to ``max_turns`` (transcript, response_text) pairs for
    ``client_id`` from the given time window, newest first.
    """
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                """
                SELECT u.transcript, u.response_text
                FROM   utterances u
                JOIN   sessions s ON u.session_id = s.id
                WHERE  s.client_id = ?
                  AND  u.ts > ?
                  AND  u.error IS NULL
                  AND  u.transcript IS NOT NULL
                  AND  u.response_text IS NOT NULL
                ORDER  BY u.ts DESC
                LIMIT  ?
                """,
                (client_id, since_ts, max_turns),
            ).fetchall()
            return rows
        finally:
            c.close()


async def get_recent_history(
    client_id: str, max_turns: int
) -> list[dict]:
    """
    Return last ``max_turns`` conversation turns for ``client_id`` within
    HISTORY_RESUME_MAX_AGE_S seconds, as a flat [{role, content}, …] list
    in chronological order (ready to splice into the LLM messages array).
    """
    since_ts = time.time() - HISTORY_RESUME_MAX_AGE_S
    rows = await asyncio.to_thread(
        _get_recent_history_sync, client_id, max_turns, since_ts
    )
    history: list[dict] = []
    for transcript, response_text in reversed(rows):  # oldest first
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": response_text})
    return history
