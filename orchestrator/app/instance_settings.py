"""
Instance-level settings (#43).

These settings apply to the whole orchestrator instance, not to a single
profile.  Stored at ``/data/settings.json`` (or ``$DATA_DIR_CONTAINER/``).

Contrast with ``user_files.UserSettings`` which is per-profile at
``/data/users/<id>/settings.json``.

Security model
──────────────
``registration_open``
    Can only be **turned OFF** via the API/UI (defence in depth).
    Re-enabling requires editing ``settings.json`` by hand and
    restarting — an attacker who captures an owner session cannot
    reopen registration to add more profiles.

``allow_guest_voice``
    Two-way toggle from the UI (lower-risk: worst case a guest can
    speak to the assistant in read-only mode).

``basic_auth_user`` / ``basic_auth_password_hash``
    Anti-scanner / anti-DDoS outer layer.  Not a replacement for
    profile authentication — it's the "door before the house" so web
    crawlers can't trivially trigger LLM / ASR calls.  Set via
    ``POST /api/instance/basic-auth``; clear by editing settings.json.
    Password is bcrypt-hashed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

# Instance settings live alongside the SQLite DB in the data root.
_DATA_ROOT = Path(os.environ.get("DATA_DIR_CONTAINER", "/data"))
_INSTANCE_SETTINGS_PATH = _DATA_ROOT / "settings.json"


class InstanceSettings(BaseModel):
    """Instance-wide configuration knobs.

    All fields are optional with safe defaults so a fresh install starts
    with registration open, no guest voice, and no Basic Auth.
    """

    # ---------- Registration ----------
    # When False, POST /api/speakers/enroll returns 403 and the "Create
    # profile" button is hidden in the UI.  Only settable to False via
    # the API — re-enabling requires manual settings.json edit + restart.
    registration_open: bool = True

    # ---------- Guest voice ----------
    # When True, unauthenticated visitors can open a WS voice session in
    # read-only (guest) mode.  When False, the WS is rejected unless
    # the va_session cookie is valid.
    allow_guest_voice: bool = False

    # ---------- Outer Basic Auth ----------
    # If both are set, all HTTP routes (except /health and /ws) require
    # HTTP Basic Auth.  password_hash is a bcrypt hash.
    basic_auth_user: str | None = None
    basic_auth_password_hash: str | None = None

    class Config:
        extra = "ignore"


_DEFAULT = InstanceSettings()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_sync() -> InstanceSettings:
    path = _INSTANCE_SETTINGS_PATH
    if not path.exists():
        return _DEFAULT.model_copy()
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("instance_settings: read error (%s) — using defaults", exc)
        return _DEFAULT.model_copy()
    try:
        return InstanceSettings.model_validate(raw)
    except ValidationError as exc:
        log.warning("instance_settings: validation error (%s) — using defaults", exc)
        return _DEFAULT.model_copy()


async def read() -> InstanceSettings:
    """Return the current instance settings (or defaults if not configured)."""
    return await asyncio.to_thread(_read_sync)


def _write_sync(settings: InstanceSettings) -> None:
    path = _INSTANCE_SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        settings.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    tmp.replace(path)


async def write(settings: InstanceSettings) -> None:
    """Atomically write instance settings to disk."""
    await asyncio.to_thread(_write_sync, settings)


async def patch(**fields: Any) -> InstanceSettings:
    """Read → merge ``fields`` → write.  Returns the new settings."""
    current = await read()
    updated = current.model_copy(update=fields)
    await write(updated)
    return updated
