"""Speaker profile persistence — enroll, identify, delete, average."""
from __future__ import annotations

import asyncio
import time

from .db import _conn, _lock


# ---------------------------------------------------------------------------
# Write / upsert
# ---------------------------------------------------------------------------

def _save_sync(client_id: str, name: str, embedding: bytes) -> int:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "INSERT INTO speaker_profiles(client_id, name, embedding, sample_count, created_at)"
                " VALUES (?, ?, ?, 1, ?)",
                (client_id, name, embedding, time.time()),
            )
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            c.close()


async def save_speaker_profile(client_id: str, name: str, embedding: bytes) -> int:
    """Insert a brand-new speaker profile (sample_count = 1)."""
    return await asyncio.to_thread(_save_sync, client_id, name, embedding)


def _update_sync(profile_id: int, embedding: bytes, sample_count: int) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE speaker_profiles SET embedding=?, sample_count=? WHERE id=?",
                (embedding, sample_count, profile_id),
            )
        finally:
            c.close()


async def update_speaker_profile(
    profile_id: int, embedding: bytes, sample_count: int
) -> None:
    """Replace the embedding and sample count for an existing profile."""
    await asyncio.to_thread(_update_sync, profile_id, embedding, sample_count)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _get_all_sync(
    client_id: str,
) -> list[tuple[int, str, bytes, int, str | None]]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, name, embedding, sample_count, tts_voice"
                " FROM speaker_profiles"
                " WHERE client_id=?"
                " ORDER BY created_at",
                (client_id,),
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def get_speaker_profiles(
    client_id: str,
) -> list[tuple[int, str, bytes, int, str | None]]:
    """
    Return (id, name, embedding_bytes, sample_count, tts_voice) for all
    enrolled speakers of this client.  ``sample_count`` is how many
    enrollment recordings have been averaged into this centroid via
    running-mean; ``tts_voice`` is the XTTS speaker name the household
    member wants to be answered with (NULL = server default).
    """
    return await asyncio.to_thread(_get_all_sync, client_id)


def _get_by_name_sync(
    client_id: str, name: str
) -> tuple[int, bytes, int] | None:
    """Return (id, embedding, sample_count) for the first profile matching (client_id, name)."""
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT id, embedding, sample_count"
                " FROM speaker_profiles"
                " WHERE client_id=? AND name=?"
                " ORDER BY created_at"
                " LIMIT 1",
                (client_id, name),
            ).fetchone()
            return row  # type: ignore[return-value]
        finally:
            c.close()


async def get_speaker_profile_by_name(
    client_id: str, name: str
) -> tuple[int, bytes, int] | None:
    """
    Return (profile_id, embedding_bytes, sample_count) for the named speaker,
    or None if not enrolled.  Used by the enrollment endpoint for running-mean
    averaging across multiple audio samples.
    """
    return await asyncio.to_thread(_get_by_name_sync, client_id, name)


# ---------------------------------------------------------------------------
# tts_voice (per-speaker TTS voice override)
# ---------------------------------------------------------------------------

def _set_tts_voice_sync(profile_id: int, voice: str | None) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE speaker_profiles SET tts_voice=? WHERE id=?",
                (voice, profile_id),
            )
        finally:
            c.close()


async def set_speaker_tts_voice(profile_id: int, voice: str | None) -> None:
    """
    Pin (or clear) the XTTS voice for a specific speaker profile.

    Pass ``None`` to clear — the speaker then falls back to the global
    xtts-server default the next time they're identified.
    """
    await asyncio.to_thread(_set_tts_voice_sync, profile_id, voice)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def _delete_sync(profile_id: int) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute("DELETE FROM speaker_profiles WHERE id=?", (profile_id,))
        finally:
            c.close()


async def delete_speaker_profile(profile_id: int) -> None:
    """Remove a speaker profile by id."""
    await asyncio.to_thread(_delete_sync, profile_id)
