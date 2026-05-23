"""
Voicemail storage — messages addressed to a household member.

Audio bytes are written as 16 kHz mono int16 WAV files under
``$DATA_DIR_CONTAINER/voice_messages/<id>.wav`` (default
``/data/voice_messages``).  The DB row stores only the relative
filename so the on-disk layout can move without a migration.

All read queries are scoped by ``to_profile_id`` — the caller is
responsible for verifying the requesting profile matches.  Auth lives
one layer up (the inbox tool checks ``ctx.profile_id``; the HTTP
endpoints check the cookie-session profile).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import wave
from pathlib import Path

from .db import _conn, _lock

log = logging.getLogger(__name__)


# On-disk audio storage.  Lives under the same mount as user-files and
# custom_voices so a single volume mount in docker-compose covers
# everything that's user data.
_DATA_DIR = Path(os.environ.get("DATA_DIR_CONTAINER", "/data"))
VOICE_MESSAGES_DIR = _DATA_DIR / "voice_messages"


def _ensure_dir() -> None:
    """Create the on-disk voice-messages folder if it doesn't exist yet.

    Cheap (single ``os.stat``) so we call it on every write — avoids a
    startup-order dependency with init_schema.
    """
    VOICE_MESSAGES_DIR.mkdir(parents=True, exist_ok=True)


def _write_wav(audio_pcm: bytes, dest: Path) -> None:
    """Wrap raw 16 kHz mono int16 PCM in a WAV container.

    The pipeline-level audio buffer is already in that exact shape (see
    :data:`app.pipeline.SAMPLE_BYTES_PER_SECOND`), so we just slap a
    header on it — no resampling, no compression.
    """
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16 = 2 bytes
        w.setframerate(16_000)
        w.writeframes(audio_pcm)


# ── Insert ──────────────────────────────────────────────────────────────


def _save_sync(
    *,
    from_profile_id: int | None,
    from_name: str | None,
    to_profile_id: int,
    to_name: str,
    transcript: str,
    duration_ms: int,
    audio_pcm: bytes,
) -> int:
    """Insert the row first (to mint an id), then write the WAV at <id>.wav.

    Two-step because we need the autoincremented id to name the file.
    If the wav write fails we delete the row so the table doesn't carry
    a dangling reference.
    """
    now = time.time()
    _ensure_dir()
    with _lock:
        c = _conn()
        try:
            # Placeholder path — we'll UPDATE it below once we know the id.
            cur = c.execute(
                """
                INSERT INTO voice_messages
                  (from_profile_id, from_name, to_profile_id, to_name,
                   audio_path, transcript, duration_ms, created_at)
                VALUES (?, ?, ?, ?, '', ?, ?, ?)
                """,
                (
                    from_profile_id,
                    from_name,
                    to_profile_id,
                    to_name,
                    transcript,
                    duration_ms,
                    now,
                ),
            )
            new_id = cur.lastrowid
            if not new_id:
                raise RuntimeError("INSERT returned no lastrowid")
            audio_name = f"{new_id}.wav"
            wav_path = VOICE_MESSAGES_DIR / audio_name
            try:
                _write_wav(audio_pcm, wav_path)
            except Exception:
                # Roll back the row — we don't want a row that points
                # at a wav that isn't there.
                c.execute("DELETE FROM voice_messages WHERE id=?", (new_id,))
                raise
            c.execute(
                "UPDATE voice_messages SET audio_path=? WHERE id=?",
                (audio_name, new_id),
            )
            return new_id
        finally:
            c.close()


async def save_voice_message(
    *,
    from_profile_id: int | None,
    from_name: str | None,
    to_profile_id: int,
    to_name: str,
    transcript: str,
    duration_ms: int,
    audio_pcm: bytes,
) -> int:
    """Persist a voicemail row + WAV file.  Returns the new row id.

    The audio is the FULL recorded utterance (including the "tell <X>
    that …" preamble) so the recipient hears the whole thing in the
    sender's voice — preserving tone and emphasis that a re-recorded
    "transcript-only" playback would lose.
    """
    return await asyncio.to_thread(
        _save_sync,
        from_profile_id=from_profile_id,
        from_name=from_name,
        to_profile_id=to_profile_id,
        to_name=to_name,
        transcript=transcript,
        duration_ms=duration_ms,
        audio_pcm=audio_pcm,
    )


# ── Read ────────────────────────────────────────────────────────────────


def _row_to_dict(r: tuple) -> dict:
    return {
        "id": r[0],
        "from_profile_id": r[1],
        "from_name": r[2],
        "to_profile_id": r[3],
        "to_name": r[4],
        "audio_path": r[5],
        "transcript": r[6],
        "summary": r[7],
        "duration_ms": r[8],
        "created_at": r[9],
        "listened_at": r[10],
        "replied_at": r[11],
        "reply_text": r[12],
        "reply_delivered_to_sender_at": r[13],
    }


_SELECT_COLS = (
    "id, from_profile_id, from_name, to_profile_id, to_name, "
    "audio_path, transcript, summary, duration_ms, created_at, "
    "listened_at, replied_at, reply_text, reply_delivered_to_sender_at"
)


def _list_for_recipient_sync(
    to_profile_id: int, unread_only: bool, limit: int
) -> list[tuple]:
    with _lock:
        c = _conn()
        try:
            where = "to_profile_id=?"
            params: list = [to_profile_id]
            if unread_only:
                where += " AND listened_at IS NULL"
            rows = c.execute(
                f"SELECT {_SELECT_COLS} FROM voice_messages"
                f" WHERE {where}"
                "  ORDER BY created_at DESC"
                "  LIMIT ?",
                tuple(params) + (limit,),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def list_for_recipient(
    to_profile_id: int, *, unread_only: bool = False, limit: int = 50
) -> list[dict]:
    """Return voicemail rows for one recipient, newest first.

    ``unread_only=True`` filters to rows that haven't been listened to
    yet (``listened_at IS NULL``).  The default limit of 50 is plenty
    for the inbox panel; the LLM caller is encouraged to pass
    ``limit=5-10`` for spoken summaries.
    """
    rows = await asyncio.to_thread(
        _list_for_recipient_sync, to_profile_id, unread_only, limit
    )
    return [_row_to_dict(r) for r in rows]


def _count_unread_sync(to_profile_id: int) -> int:
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT COUNT(*) FROM voice_messages"
                " WHERE to_profile_id=? AND listened_at IS NULL",
                (to_profile_id,),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            c.close()


async def count_unread(to_profile_id: int) -> int:
    """How many unlistened messages does this profile have."""
    return await asyncio.to_thread(_count_unread_sync, to_profile_id)


def _get_one_sync(message_id: int) -> tuple | None:
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                f"SELECT {_SELECT_COLS} FROM voice_messages WHERE id=?",
                (message_id,),
            ).fetchone()
            return row  # type: ignore[return-value]
        finally:
            c.close()


async def get_voice_message(message_id: int) -> dict | None:
    """Fetch one row by id, regardless of recipient.

    Callers MUST verify ownership (``row['to_profile_id'] ==
    requesting_profile_id``) before exposing the result — this read
    is unfiltered on purpose so the executor / HTTP layer can decide
    the right 401/403 response.
    """
    row = await asyncio.to_thread(_get_one_sync, message_id)
    return _row_to_dict(row) if row else None


# ── Mutate ──────────────────────────────────────────────────────────────


def _mark_listened_sync(message_id: int, to_profile_id: int) -> bool:
    """Stamp ``listened_at`` if not already set.  Returns True on the first hit."""
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "UPDATE voice_messages SET listened_at=?"
                " WHERE id=? AND to_profile_id=? AND listened_at IS NULL",
                (now, message_id, to_profile_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def mark_listened(message_id: int, to_profile_id: int) -> bool:
    """Record that the recipient has now heard this message.

    Idempotent — repeated calls after the first one return False (the
    UPDATE WHERE clause filters out already-listened rows).  Scoped by
    ``to_profile_id`` so a stolen message id can't be marked-listened
    by the wrong profile.
    """
    return await asyncio.to_thread(_mark_listened_sync, message_id, to_profile_id)


def _save_reply_sync(message_id: int, to_profile_id: int, reply: str) -> bool:
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "UPDATE voice_messages"
                "    SET reply_text=?, replied_at=?,"
                "        listened_at = COALESCE(listened_at, ?)"
                "  WHERE id=? AND to_profile_id=?",
                (reply, now, now, message_id, to_profile_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def save_reply(message_id: int, to_profile_id: int, reply: str) -> bool:
    """Record the recipient's textual reply.

    Also marks the message as listened (a reply implies it was heard),
    using ``COALESCE`` so an existing ``listened_at`` isn't overwritten.
    Returns False if the row doesn't belong to ``to_profile_id`` — the
    caller surfaces that as a 404 in the HTTP layer.
    """
    return await asyncio.to_thread(
        _save_reply_sync, message_id, to_profile_id, reply
    )


def _set_summary_sync(message_id: int, summary: str) -> bool:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "UPDATE voice_messages SET summary=? WHERE id=?",
                (summary, message_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def set_summary(message_id: int, summary: str) -> bool:
    """Cache the LLM-generated summary so re-asks don't hit the model.

    No ``to_profile_id`` guard here because the summary is derived
    content; the auth check on the way IN to ``inbox_summary`` is
    sufficient.
    """
    return await asyncio.to_thread(_set_summary_sync, message_id, summary)


# ── Reply-replay to the original sender ────────────────────────────────
#
# Wiring:  recipient replies via inbox_reply / UI POST → reply_text is
# set on the row.  The next time the ORIGINAL SENDER speaks (any
# utterance), the pipeline injects a context line telling the LLM that
# the recipient replied; the LLM then folds it into its response.  We
# stamp ``reply_delivered_to_sender_at`` at injection time so subsequent
# turns don't re-surface the same reply.


def _list_unseen_replies_for_sender_sync(from_profile_id: int) -> list[tuple]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                f"SELECT {_SELECT_COLS} FROM voice_messages"
                "  WHERE from_profile_id=?"
                "    AND reply_text IS NOT NULL"
                "    AND reply_delivered_to_sender_at IS NULL"
                "  ORDER BY replied_at DESC"
                "  LIMIT 3",
                (from_profile_id,),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def list_unseen_replies_for_sender(from_profile_id: int) -> list[dict]:
    """Return replies the sender hasn't been told about yet, newest first.

    Capped at 3 — voice-replay needs a sentence the LLM can naturally
    fold into its answer ("by the way, X replied…") without overwhelming
    the user.  Older replies are not "lost"; they just stop auto-
    surfacing.  The recipient's inbox row still shows them.
    """
    rows = await asyncio.to_thread(
        _list_unseen_replies_for_sender_sync, from_profile_id
    )
    return [_row_to_dict(r) for r in rows]


def _mark_reply_delivered_sync(message_id: int) -> bool:
    """Stamp ``reply_delivered_to_sender_at`` so we don't re-surface this row.

    Idempotent — the WHERE clause filters out already-stamped rows so a
    crash-loop between two pipeline turns can't double-mark anything
    (the field is set or it isn't).
    """
    now = time.time()
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "UPDATE voice_messages"
                "   SET reply_delivered_to_sender_at=?"
                " WHERE id=? AND reply_delivered_to_sender_at IS NULL",
                (now, message_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def mark_reply_delivered(message_id: int) -> bool:
    """Record that the sender has now been told the recipient replied.

    Doesn't take a ``from_profile_id`` because the list helper already
    scoped to the sender — the message_id round-trips through the same
    pipeline turn and never crosses a profile boundary.
    """
    return await asyncio.to_thread(_mark_reply_delivered_sync, message_id)


# ── Outgoing (sender's view) ───────────────────────────────────────────


def _list_outgoing_sync(from_profile_id: int, limit: int) -> list[tuple]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                f"SELECT {_SELECT_COLS} FROM voice_messages"
                "  WHERE from_profile_id=?"
                "  ORDER BY created_at DESC"
                "  LIMIT ?",
                (from_profile_id, limit),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def list_outgoing_voicemail(
    from_profile_id: int, *, limit: int = 50
) -> list[dict]:
    """Return voicemail rows the given profile *sent*, newest first.

    Mirrors :func:`list_for_recipient` but filters on the sender column.
    The reply (if any) is included in the returned dict — the Sent view
    in the UI shows it as the main payload.
    """
    rows = await asyncio.to_thread(
        _list_outgoing_sync, from_profile_id, max(1, min(int(limit), 200))
    )
    return [_row_to_dict(r) for r in rows]


def _delete_sync(message_id: int, to_profile_id: int) -> bool:
    """Drop one row + its on-disk audio file.

    The file unlink errors are swallowed — a missing wav is harmless
    (better than a half-deletion that leaves the row orphaned).
    """
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT audio_path FROM voice_messages"
                " WHERE id=? AND to_profile_id=?",
                (message_id, to_profile_id),
            ).fetchone()
            if not row:
                return False
            audio_name = row[0]
            cur = c.execute(
                "DELETE FROM voice_messages WHERE id=? AND to_profile_id=?",
                (message_id, to_profile_id),
            )
            if cur.rowcount > 0 and audio_name:
                try:
                    (VOICE_MESSAGES_DIR / audio_name).unlink(missing_ok=True)
                except Exception:
                    log.warning(
                        "voice_messages: failed to unlink %r", audio_name,
                        exc_info=True,
                    )
            return cur.rowcount > 0
        finally:
            c.close()


async def delete_voice_message(message_id: int, to_profile_id: int) -> bool:
    """Remove a message + its wav.  Scoped by recipient like every mutate path."""
    return await asyncio.to_thread(_delete_sync, message_id, to_profile_id)


def audio_path(message_id_or_filename: int | str) -> Path:
    """Build the absolute on-disk path for a voicemail audio file.

    Accepts either the row id (``1`` → ``1.wav``) or a stored
    ``audio_path`` value (which is already ``<id>.wav``).
    """
    if isinstance(message_id_or_filename, int):
        return VOICE_MESSAGES_DIR / f"{message_id_or_filename}.wav"
    return VOICE_MESSAGES_DIR / message_id_or_filename
