"""Profile-device pairing storage (#50).

Maps a speaker_profile to the desktop-agent(s) it should use for
device-tier tool calls (computer_use, look_at_screen, desktop).
``device_uid`` equals ``AgentInfo.agent_id`` in desktop_client so the
router can resolve directly without a secondary lookup.

Pairing flow
────────────
1.  Admin/user calls ``POST /api/devices/pair-code`` → 6-digit code.
2.  Code is stored in ``device_pairing_codes`` with a 5-minute TTL.
3.  Desktop-agent is configured with the code (env var PAIRING_CODE).
4.  On reverse-WSS connect the agent sends a ``pair`` frame with the
    code + its agent_id.  ``consume_pairing_code()`` validates the code
    and calls ``upsert_device()`` to create the pairing row.
5.  The code is marked ``used = 1`` so a replay attack can't re-pair.

Direct registration (no code) is also supported for admin setups via
``upsert_device()`` directly.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any

from .db import _conn, _lock


# ---------------------------------------------------------------------------
# Profile-device CRUD
# ---------------------------------------------------------------------------

def _list_devices_sync(profile_id: int) -> list[dict[str, Any]]:
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                """
                SELECT id, profile_id, device_kind, device_uid,
                       friendly_name, is_default, wol_mac, wol_target_ip,
                       last_seen_at, created_at
                FROM   profile_devices
                WHERE  profile_id = ?
                ORDER  BY is_default DESC, created_at ASC
                """,
                (profile_id,),
            ).fetchall()
            cols = [
                "id", "profile_id", "device_kind", "device_uid",
                "friendly_name", "is_default", "wol_mac", "wol_target_ip",
                "last_seen_at", "created_at",
            ]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            c.close()


async def list_devices(profile_id: int) -> list[dict[str, Any]]:
    """Return all devices paired with this profile, default first."""
    return await asyncio.to_thread(_list_devices_sync, profile_id)


def _get_default_device_sync(
    profile_id: int, device_kind: str | None
) -> dict[str, Any] | None:
    """Return the preferred device for (profile, device_kind).

    Selection order:
      1. is_default = 1 (for the given device_kind)
      2. Any device of the given device_kind (oldest created)
      3. If device_kind is None: first device of any kind
    """
    with _lock:
        c = _conn()
        try:
            cols = (
                "id, profile_id, device_kind, device_uid, "
                "friendly_name, is_default, wol_mac, wol_target_ip, "
                "last_seen_at, created_at"
            )
            col_names = [
                "id", "profile_id", "device_kind", "device_uid",
                "friendly_name", "is_default", "wol_mac", "wol_target_ip",
                "last_seen_at", "created_at",
            ]
            if device_kind:
                row = c.execute(
                    f"SELECT {cols} FROM profile_devices"
                    " WHERE profile_id=? AND device_kind=?"
                    " ORDER BY is_default DESC, created_at ASC LIMIT 1",
                    (profile_id, device_kind),
                ).fetchone()
            else:
                row = c.execute(
                    f"SELECT {cols} FROM profile_devices"
                    " WHERE profile_id=?"
                    " ORDER BY is_default DESC, created_at ASC LIMIT 1",
                    (profile_id,),
                ).fetchone()
            if row is None:
                return None
            return dict(zip(col_names, row))
        finally:
            c.close()


async def get_default_device(
    profile_id: int, device_kind: str | None = None
) -> dict[str, Any] | None:
    """Return the preferred device for this profile (and optional device_kind)."""
    return await asyncio.to_thread(_get_default_device_sync, profile_id, device_kind)


def _get_device_by_uid_sync(device_uid: str) -> dict[str, Any] | None:
    with _lock:
        c = _conn()
        try:
            cols = (
                "id, profile_id, device_kind, device_uid, "
                "friendly_name, is_default, wol_mac, wol_target_ip, "
                "last_seen_at, created_at"
            )
            col_names = [
                "id", "profile_id", "device_kind", "device_uid",
                "friendly_name", "is_default", "wol_mac", "wol_target_ip",
                "last_seen_at", "created_at",
            ]
            row = c.execute(
                f"SELECT {cols} FROM profile_devices WHERE device_uid=? LIMIT 1",
                (device_uid,),
            ).fetchone()
            return dict(zip(col_names, row)) if row else None
        finally:
            c.close()


async def get_device_by_uid(device_uid: str) -> dict[str, Any] | None:
    """Look up a pairing row by the agent's device_uid."""
    return await asyncio.to_thread(_get_device_by_uid_sync, device_uid)


def _upsert_device_sync(
    profile_id: int,
    device_uid: str,
    device_kind: str,
    friendly_name: str,
    *,
    is_default: bool = False,
    wol_mac: str | None = None,
    wol_target_ip: str | None = None,
) -> int:
    with _lock:
        c = _conn()
        try:
            now = time.time()
            # Use INSERT OR REPLACE so a re-pair updates the row.
            cur = c.execute(
                """
                INSERT INTO profile_devices
                    (profile_id, device_kind, device_uid, friendly_name,
                     is_default, wol_mac, wol_target_ip, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, device_uid) DO UPDATE SET
                    device_kind   = excluded.device_kind,
                    friendly_name = excluded.friendly_name,
                    is_default    = excluded.is_default,
                    wol_mac       = excluded.wol_mac,
                    wol_target_ip = excluded.wol_target_ip
                """,
                (
                    profile_id, device_kind, device_uid, friendly_name,
                    int(is_default), wol_mac, wol_target_ip, now,
                ),
            )
            # If this device is default, demote others for the same
            # (profile, device_kind) so there's at most one default.
            if is_default:
                c.execute(
                    """
                    UPDATE profile_devices
                    SET is_default = 0
                    WHERE profile_id = ? AND device_kind = ?
                      AND device_uid != ?
                    """,
                    (profile_id, device_kind, device_uid),
                )
            return cur.lastrowid or 0  # type: ignore[return-value]
        finally:
            c.close()


async def upsert_device(
    profile_id: int,
    device_uid: str,
    device_kind: str,
    friendly_name: str,
    *,
    is_default: bool = False,
    wol_mac: str | None = None,
    wol_target_ip: str | None = None,
) -> int:
    """Create or update a profile-device pairing. Returns the row ID."""
    return await asyncio.to_thread(
        _upsert_device_sync,
        profile_id, device_uid, device_kind, friendly_name,
        is_default=is_default, wol_mac=wol_mac, wol_target_ip=wol_target_ip,
    )


def _delete_device_sync(device_id: int, profile_id: int) -> bool:
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM profile_devices WHERE id=? AND profile_id=?",
                (device_id, profile_id),
            )
            return cur.rowcount > 0
        finally:
            c.close()


async def delete_device(device_id: int, profile_id: int) -> bool:
    """Delete a device pairing. Returns True if a row was deleted."""
    return await asyncio.to_thread(_delete_device_sync, device_id, profile_id)


def _touch_device_sync(device_uid: str) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE profile_devices SET last_seen_at=? WHERE device_uid=?",
                (time.time(), device_uid),
            )
        finally:
            c.close()


async def touch_device(device_uid: str) -> None:
    """Update last_seen_at for the device (called on heartbeat)."""
    await asyncio.to_thread(_touch_device_sync, device_uid)


# ---------------------------------------------------------------------------
# Pairing codes
# ---------------------------------------------------------------------------

_PAIRING_TTL_S = 5 * 60  # 5 minutes


def _create_pairing_code_sync(profile_id: int, device_kind: str) -> str:
    """Generate a unique 6-digit code and store it.  Returns the code."""
    with _lock:
        c = _conn()
        try:
            # Purge expired codes first (housekeeping).
            c.execute("DELETE FROM device_pairing_codes WHERE expires_at < ?", (time.time(),))
            # Generate a code that isn't currently active.  Use the CSPRNG
            # (secrets), not Mersenne-Twister random — a predictable pairing
            # code is guessable within the 5-min window, and pairing binds a
            # device as the routing target for a profile's device-tier tools.
            for _ in range(20):
                code = f"{secrets.randbelow(1000000):06d}"
                exists = c.execute(
                    "SELECT 1 FROM device_pairing_codes WHERE code=? AND used=0 AND expires_at>?",
                    (code, time.time()),
                ).fetchone()
                if not exists:
                    break
            c.execute(
                "INSERT OR REPLACE INTO device_pairing_codes"
                " (code, profile_id, device_kind, expires_at, used)"
                " VALUES (?, ?, ?, ?, 0)",
                (code, profile_id, device_kind, time.time() + _PAIRING_TTL_S),
            )
            return code
        finally:
            c.close()


async def create_pairing_code(profile_id: int, device_kind: str) -> str:
    """Generate a 6-digit pairing code for this (profile, device_kind)."""
    return await asyncio.to_thread(_create_pairing_code_sync, profile_id, device_kind)


def _consume_pairing_code_sync(
    code: str, device_uid: str, friendly_name: str
) -> tuple[int, str] | None:
    """Validate + consume a pairing code.

    Returns ``(profile_id, device_kind)`` on success, ``None`` on failure
    (wrong/expired/already-used code).  On success also calls
    ``upsert_device`` inline so the pairing is committed atomically.
    """
    with _lock:
        c = _conn()
        try:
            row = c.execute(
                "SELECT profile_id, device_kind FROM device_pairing_codes"
                " WHERE code=? AND used=0 AND expires_at>?",
                (code, time.time()),
            ).fetchone()
            if row is None:
                return None
            profile_id, device_kind = int(row[0]), str(row[1])
            # Mark used.
            c.execute(
                "UPDATE device_pairing_codes SET used=1 WHERE code=?", (code,)
            )
            # Upsert the pairing directly (same lock, same transaction).
            now = time.time()
            c.execute(
                """
                INSERT INTO profile_devices
                    (profile_id, device_kind, device_uid, friendly_name,
                     is_default, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(profile_id, device_uid) DO UPDATE SET
                    device_kind   = excluded.device_kind,
                    friendly_name = excluded.friendly_name
                """,
                (profile_id, device_kind, device_uid, friendly_name, now),
            )
            return profile_id, device_kind
        finally:
            c.close()


async def consume_pairing_code(
    code: str, device_uid: str, friendly_name: str = ""
) -> tuple[int, str] | None:
    """Consume a pairing code and create the device row.

    Returns ``(profile_id, device_kind)`` on success, ``None`` if the
    code is invalid, expired, or already used.
    """
    return await asyncio.to_thread(
        _consume_pairing_code_sync, code, device_uid, friendly_name
    )
