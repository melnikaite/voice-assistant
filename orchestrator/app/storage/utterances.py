"""Utterance write/read — transcript, embedding, memory candidates."""
from __future__ import annotations

import asyncio
import time

from .db import _conn, _lock


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _save_utterance_sync(fields: dict) -> int:
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                f"INSERT INTO utterances({cols}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            c.close()


async def save_utterance(**fields) -> int:
    return await asyncio.to_thread(_save_utterance_sync, fields)


def _update_embedding_sync(utterance_id: int, blob: bytes) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE utterances SET embedding=? WHERE id=?",
                (blob, utterance_id),
            )
        finally:
            c.close()


async def update_utterance_embedding(utterance_id: int, blob: bytes) -> None:
    await asyncio.to_thread(_update_embedding_sync, utterance_id, blob)


# ---------------------------------------------------------------------------
# Read — semantic memory candidates
# ---------------------------------------------------------------------------

def _get_candidate_utterances_sync(
    client_id: str,
    since_ts: float,
    limit: int,
    speaker_name: str | None,
) -> list[tuple[str, str, bytes]]:
    """
    Return up to ``limit`` (transcript, response_text, embedding) tuples
    for semantic similarity ranking.  Only rows that already have an
    embedding are returned.

    Speaker isolation:
    - If ``speaker_name`` is given: return rows for that speaker OR rows
      marked ``is_shared=1`` (explicitly shared across speakers).
    - If ``speaker_name`` is None (no ID in this session): return all rows
      (no speaker filter) so the assistant still has full context.
    """
    with _lock:
        c = _conn()
        try:
            if speaker_name:
                rows = c.execute(
                    """
                    SELECT u.transcript, u.response_text, u.embedding
                    FROM   utterances u
                    JOIN   sessions s ON u.session_id = s.id
                    WHERE  s.client_id = ?
                      AND  u.ts > ?
                      AND  u.embedding IS NOT NULL
                      AND  u.error IS NULL
                      AND  u.transcript IS NOT NULL
                      AND  (u.is_shared = 1
                            OR u.speaker_name IS NULL
                            OR u.speaker_name = ?)
                    ORDER  BY u.ts DESC
                    LIMIT  ?
                    """,
                    (client_id, since_ts, speaker_name, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    """
                    SELECT u.transcript, u.response_text, u.embedding
                    FROM   utterances u
                    JOIN   sessions s ON u.session_id = s.id
                    WHERE  s.client_id = ?
                      AND  u.ts > ?
                      AND  u.embedding IS NOT NULL
                      AND  u.error IS NULL
                      AND  u.transcript IS NOT NULL
                    ORDER  BY u.ts DESC
                    LIMIT  ?
                    """,
                    (client_id, since_ts, limit),
                ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def get_candidate_utterances(
    client_id: str,
    since_ts: float,
    limit: int = 200,
    speaker_name: str | None = None,
) -> list[tuple[str, str, bytes]]:
    """
    Fetch embedding candidates for semantic memory retrieval.
    Pass ``speaker_name`` to restrict results to that speaker (+ shared rows).
    """
    return await asyncio.to_thread(
        _get_candidate_utterances_sync, client_id, since_ts, limit, speaker_name
    )


# ---------------------------------------------------------------------------
# Read — observability: tool performance + voice turn counts
# ---------------------------------------------------------------------------


def _get_tool_perf_sync(days: int) -> list[dict]:
    """Per-tool stats from utterances: call count, avg latency, error rate.

    Only rows with a non-null tool_name are included (pure LLM turns
    where the model responded directly without a tool call have
    tool_name=NULL and are not counted here).  The error_count covers
    any non-null ``error`` column value — network failures, unknown
    tool, crash, etc.
    """
    since = time.time() - days * 86400
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                """
                SELECT tool_name,
                       COUNT(*)                                       AS calls,
                       AVG(llm_ms)                                    AS avg_ms,
                       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
                FROM   utterances
                WHERE  ts > ?
                  AND  tool_name IS NOT NULL
                GROUP  BY tool_name
                ORDER  BY calls DESC
                """,
                (since,),
            ).fetchall()
            return [
                {
                    "tool_name": row[0],
                    "calls": int(row[1] or 0),
                    "avg_ms": round(float(row[2] or 0)),
                    "errors": int(row[3] or 0),
                    "error_rate": round(int(row[3] or 0) / max(1, int(row[1] or 1)), 3),
                }
                for row in rows
            ]
        finally:
            c.close()


async def get_tool_perf(days: int = 7) -> list[dict]:
    """Per-tool call count / avg latency / error rate for the last N days."""
    return await asyncio.to_thread(_get_tool_perf_sync, days)


def _get_voice_turns_today_sync() -> int:
    """Count utterances with a transcript recorded today (local-day boundary)."""
    with _lock:
        c = _conn()
        try:
            # Single query: compare each row's local-day to SQLite's
            # current local-day.  The container TZ is set to the
            # deployment's zone, so this matches what a human means by
            # "today".
            row = c.execute(
                """
                SELECT COUNT(*)
                FROM   utterances
                WHERE  date(ts, 'unixepoch', 'localtime') = date('now', 'localtime')
                  AND  transcript IS NOT NULL
                """
            ).fetchone()
            return int(row[0] or 0)
        finally:
            c.close()


async def get_voice_turns_today() -> int:
    """Number of voice turns completed today (local time)."""
    return await asyncio.to_thread(_get_voice_turns_today_sync)
