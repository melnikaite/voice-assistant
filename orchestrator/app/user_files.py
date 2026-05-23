"""
Per-user file storage — settings.json + memory.md per enrolled profile.

This is the *non-SQLite* half of user identity.  Where the
``speaker_profiles`` table stores the resemblyzer d-vector + name + ID,
``user_files`` stores everything the assistant should *remember* and
*honour* about that person:

  /data/users/<profile_id>/
    ├── settings.json    # typed, schema-validated (Pydantic).  Read by
    │                    # the app (language, tts_voice, formality) AND
    │                    # injected into the LLM context (style_prompt,
    │                    # custom freeform keys).
    └── memory.md        # freeform Markdown.  Written and read by the
                         # `memory` LLM tool.  No schema, no parsing —
                         # the LLM reads the raw bytes and decides.

Why files instead of SQLite columns:

* settings.json is hand-editable (UI shows a JSON editor; power-users
  can `cat`/`vim` it from the host).
* memory.md grows freely and is reasonable to inspect, version, or
  back up as plain text.
* Splitting them keeps the "always-on" structured surface separate
  from the "occasionally-read by LLM" free text — no risk of one
  bloating into the other.

Passphrase storage:
  ``settings.code_word_hash`` carries a bcrypt hash of the user's
  spoken passphrase (the "tier-2" auth secret for invasive actions).
  ``set_passphrase`` / ``verify_passphrase`` are async wrappers around
  bcrypt so the (slow) hash runs in a thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


# ── Path layout ─────────────────────────────────────────────────────────

# Inside the container the orchestrator sees /data via the compose
# bind-mount.  Override via env if you ever change the mount point;
# storage.db, custom_voices/, users/ all share this root.
_DATA_ROOT = Path(os.environ.get("DATA_DIR_CONTAINER", "/data"))
_USERS_ROOT = _DATA_ROOT / "users"


def user_dir(profile_id: int) -> Path:
    """The directory holding one profile's settings + memory files.

    Creates it on first use so the rest of this module can assume it
    exists.  Idempotent — safe to call from any read or write path.
    """
    d = _USERS_ROOT / str(profile_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings_path(profile_id: int) -> Path:
    return user_dir(profile_id) / "settings.json"


def memory_path(profile_id: int) -> Path:
    return user_dir(profile_id) / "memory.md"


# ── Settings schema ─────────────────────────────────────────────────────

# Locale codes the orchestrator honours end-to-end (Whisper hint, TTS
# voice mapping, generated-text language).  ``auto`` defers to Whisper's
# per-utterance auto-detect (today's default behaviour).
Language = Literal["ru", "de", "en", "auto"]

# Stylistic register, injected into the LLM system prompt.  ``casual``
# is the project's default tone; ``formal`` for polite/honorific
# speech; ``kid`` for simpler vocabulary when answering children.
Formality = Literal["formal", "casual", "kid"]

# Tier name.  ``tier1`` covers every read + low_write action; ``tier2``
# unlocks high_write (memory edits, settings changes, calendar writes,
# anything destructive).  See docs/permissions.md for the matrix.
Permission = Literal["tier1", "tier2"]


class UserSettings(BaseModel):
    """
    Reserved keys live as typed fields; anything else the LLM or user
    wants to remember goes under ``custom``.  Validation runs every
    time we read or write — a malformed settings.json yields a polite
    default instead of crashing the orchestrator.
    """

    language: Language = "auto"
    # XTTS speaker name (built-in like "Claribel Dervla", or "clone:<id>"
    # for a recorded reference voice).  None falls back to server default.
    tts_voice: str | None = None
    formality: Formality = "casual"
    # Freeform addition to the system prompt — the user's preferred style
    # (e.g. "keep answers short", "address me formally", "include jokes").
    style_prompt: str | None = None
    # Bcrypt hash of the spoken passphrase that unlocks tier-2 actions.
    # None means "no passphrase set yet — tier-2 not available for this
    # profile".  Plain text never persisted.
    code_word_hash: str | None = None
    # What this profile is allowed to do.  Default is tier1 only.
    permissions: list[Permission] = Field(default_factory=lambda: ["tier1"])
    # Everything else.  LLM tools can write here freely; app code does
    # not interpret these fields.
    custom: dict = Field(default_factory=dict)

    class Config:
        extra = "ignore"  # silently drop legacy / unknown top-level keys


_DEFAULT_SETTINGS = UserSettings()


# ── Settings I/O ────────────────────────────────────────────────────────


def _read_settings_sync(profile_id: int) -> UserSettings:
    p = settings_path(profile_id)
    if not p.exists():
        return _DEFAULT_SETTINGS.model_copy()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("settings: bad JSON for profile=%d (%s) — using defaults", profile_id, exc)
        return _DEFAULT_SETTINGS.model_copy()
    try:
        return UserSettings.model_validate(raw)
    except ValidationError as exc:
        log.warning("settings: schema violation for profile=%d (%s) — using defaults", profile_id, exc)
        return _DEFAULT_SETTINGS.model_copy()


async def read_settings(profile_id: int) -> UserSettings:
    """Return this profile's settings, or a defaults instance if missing."""
    return await asyncio.to_thread(_read_settings_sync, profile_id)


def _write_settings_sync(profile_id: int, settings: UserSettings) -> None:
    p = settings_path(profile_id)
    # Atomic write via tmp + rename so a crash mid-write doesn't leave a
    # half-flushed JSON file on disk.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        settings.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    tmp.replace(p)


async def write_settings(profile_id: int, settings: UserSettings) -> None:
    await asyncio.to_thread(_write_settings_sync, profile_id, settings)


async def patch_settings(profile_id: int, **fields) -> UserSettings:
    """
    Read current settings, merge ``fields`` on top, write back.

    Used by the /api/users/<id>/settings PATCH endpoint and by the
    ``settings`` LLM tool.  Returns the resulting settings so the
    caller can echo them.
    """
    current = await read_settings(profile_id)
    updated = current.model_copy(update=fields)
    await write_settings(profile_id, updated)
    return updated


# ── Memory.md I/O ───────────────────────────────────────────────────────


def _read_memory_sync(profile_id: int) -> str:
    p = memory_path(profile_id)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("memory: read error for profile=%d (%s)", profile_id, exc)
        return ""


async def read_memory(profile_id: int) -> str:
    return await asyncio.to_thread(_read_memory_sync, profile_id)


def _write_memory_sync(profile_id: int, content: str) -> None:
    p = memory_path(profile_id)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)


async def write_memory(profile_id: int, content: str) -> None:
    """Replace the whole memory.md for this profile."""
    await asyncio.to_thread(_write_memory_sync, profile_id, content)


async def append_memory(profile_id: int, snippet: str) -> str:
    """
    Add a Markdown bullet to the end of memory.md.

    The LLM's `memory` tool calls this for "remember that …" — easier
    than asking the model to do diff-style edits.  Returns the new full
    contents so the caller can echo them.
    """
    existing = await read_memory(profile_id)
    sep = "" if not existing or existing.endswith("\n") else "\n"
    snippet = snippet.strip()
    if not snippet:
        return existing
    updated = f"{existing}{sep}- {snippet}\n"
    await write_memory(profile_id, updated)
    return updated


# ── Passphrase (bcrypt) ─────────────────────────────────────────────────


def _hash_passphrase_sync(plaintext: str) -> str:
    # Local import keeps the bcrypt dependency lazy — if you start the
    # orchestrator without bcrypt installed, settings reads and memory
    # still work; only passphrase ops fail.
    import bcrypt

    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def _verify_passphrase_sync(plaintext: str, hashed: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# Trailing punctuation Whisper likes to tack on at end-of-utterance.
# We strip these from both set + verify so the user's spoken phrase
# matches regardless of how Whisper punctuated it that turn.
_PUNCTUATION_STRIP = ".,!?;: \t\n"


def _normalise_passphrase(plaintext: str) -> str:
    """Casefold + strip whitespace and trailing punctuation.

    Spoken passphrases drift across Whisper transcriptions: "Amber.",
    "amber", " AMBER " all refer to the same word.  Normalise
    aggressively so the human just says the word.
    """
    return plaintext.strip(_PUNCTUATION_STRIP).lower()


async def set_passphrase(profile_id: int, plaintext: str) -> None:
    """Bcrypt-hash the spoken passphrase and store it in settings.json."""
    normalised = _normalise_passphrase(plaintext)
    if not normalised:
        raise ValueError("passphrase cannot be empty")
    hashed = await asyncio.to_thread(_hash_passphrase_sync, normalised)
    current = await read_settings(profile_id)
    updated = current.model_copy(update={"code_word_hash": hashed})
    await write_settings(profile_id, updated)


async def verify_passphrase(profile_id: int, plaintext: str) -> bool:
    """True iff ``plaintext`` matches the stored bcrypt hash."""
    normalised = _normalise_passphrase(plaintext)
    if not normalised:
        return False
    settings = await read_settings(profile_id)
    if not settings.code_word_hash:
        return False
    return await asyncio.to_thread(
        _verify_passphrase_sync, normalised, settings.code_word_hash
    )
