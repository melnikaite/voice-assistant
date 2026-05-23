"""Utterance write/read — transcript, embedding, memory candidates."""
from __future__ import annotations

import asyncio

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
