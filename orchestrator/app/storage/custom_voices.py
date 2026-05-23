"""Custom output voices — persistence for cloned-voice references.

Each row is a user-recorded 6-12 s sample that xtts-server reads at
inference time to clone a voice on the fly.  Independent from speaker
profiles (which identify who's TALKING) — these describe who the
assistant SOUNDS like.

The WAV file lives on disk under ``/data/custom_voices/<id>.wav``
(container path) which maps to ``<repo>/data/custom_voices/<id>.wav``
on the host so xtts-server can read it directly.
"""
from __future__ import annotations

import asyncio
import time

from .db import _conn, _lock


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _save_sync(name: str, wav_path: str) -> int:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "INSERT INTO custom_voices(name, wav_path, created_at)"
                " VALUES (?, ?, ?)",
                (name, wav_path, time.time()),
            )
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            c.close()


async def save_custom_voice(name: str, wav_path: str) -> int:
    """Insert a new custom voice row; returns its id."""
    return await asyncio.to_thread(_save_sync, name, wav_path)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _get_all_sync() -> list[tuple[int, str, str]]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, name, wav_path FROM custom_voices ORDER BY created_at"
            ).fetchall()
            return rows  # type: ignore[return-value]
        finally:
            c.close()


async def get_custom_voices() -> list[tuple[int, str, str]]:
    """Return [(id, name, wav_path), ...] for every saved custom voice."""
    return await asyncio.to_thread(_get_all_sync)


def _get_by_id_sync(voice_id: int) -> tuple[int, str, str] | None:
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT id, name, wav_path FROM custom_voices WHERE id=?",
                (voice_id,),
            ).fetchone()
            return row  # type: ignore[return-value]
        finally:
            c.close()


async def get_custom_voice_by_id(voice_id: int) -> tuple[int, str, str] | None:
    """Return (id, name, wav_path) for the row or None."""
    return await asyncio.to_thread(_get_by_id_sync, voice_id)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def _delete_sync(voice_id: int) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute("DELETE FROM custom_voices WHERE id=?", (voice_id,))
        finally:
            c.close()


async def delete_custom_voice(voice_id: int) -> None:
    """Remove a custom voice row by id (file deletion happens in the endpoint)."""
    await asyncio.to_thread(_delete_sync, voice_id)
