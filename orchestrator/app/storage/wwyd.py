"""Per-profile wwyd connections and in-flight OAuth authorizations.

Tokens are bearer secrets: helpers that return them are used only by
``app.wwyd`` to build requests.  ``connection_public`` is the shape the
HTTP API may expose.
"""
from __future__ import annotations

import asyncio
import json
import time

from .db import _conn

PENDING_TTL_S = 10 * 60

_COLS = (
    "profile_id", "base_url", "resource", "auth_kind", "client_id",
    "access_token", "refresh_token", "expires_at", "token_endpoint",
    "revocation_endpoint", "audience_map", "connected_at",
)


def _row_dict(row: tuple | None) -> dict | None:
    if row is None:
        return None
    d = dict(zip(_COLS, row))
    try:
        d["audience_map"] = json.loads(d["audience_map"] or "{}")
    except ValueError:
        d["audience_map"] = {}
    return d


def _get_sync(profile_id: int) -> dict | None:
    row = _conn().execute(
        f"SELECT {', '.join(_COLS)} FROM wwyd_connections WHERE profile_id=?",
        (profile_id,),
    ).fetchone()
    return _row_dict(row)


async def get_wwyd_connection(profile_id: int) -> dict | None:
    return await asyncio.to_thread(_get_sync, profile_id)


def connection_public(conn: dict | None) -> dict | None:
    """The connection without its secrets."""
    if conn is None:
        return None
    return {
        "url": conn["base_url"],
        "auth_kind": conn["auth_kind"],
        "audience_map": conn["audience_map"],
        "connected_at": conn["connected_at"],
    }


def _save_sync(profile_id: int, fields: dict) -> None:
    now = time.time()
    _conn().execute(
        "INSERT OR REPLACE INTO wwyd_connections (profile_id, base_url, resource,"
        " auth_kind, client_id, access_token, refresh_token, expires_at,"
        " token_endpoint, revocation_endpoint, audience_map, connected_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)",
        (
            profile_id, fields["base_url"], fields["resource"], fields["auth_kind"],
            fields.get("client_id"), fields["access_token"], fields.get("refresh_token"),
            fields.get("expires_at"), fields.get("token_endpoint"),
            fields.get("revocation_endpoint"), now,
        ),
    )


async def save_wwyd_connection(profile_id: int, **fields) -> None:
    """Create or replace the profile's connection (the audience map resets)."""
    await asyncio.to_thread(_save_sync, profile_id, fields)


def _update_tokens_sync(profile_id: int, access: str, refresh: str | None, expires_at: float | None) -> None:
    _conn().execute(
        "UPDATE wwyd_connections SET access_token=?, refresh_token=?, expires_at=?"
        " WHERE profile_id=?",
        (access, refresh, expires_at, profile_id),
    )


async def update_wwyd_tokens(
    profile_id: int, access_token: str, refresh_token: str | None, expires_at: float | None
) -> None:
    await asyncio.to_thread(_update_tokens_sync, profile_id, access_token, refresh_token, expires_at)


def _set_map_sync(profile_id: int, mapping: dict) -> bool:
    cur = _conn().execute(
        "UPDATE wwyd_connections SET audience_map=? WHERE profile_id=?",
        (json.dumps(mapping, ensure_ascii=False), profile_id),
    )
    return cur.rowcount > 0


async def set_wwyd_audience_map(profile_id: int, mapping: dict[str, str]) -> bool:
    return await asyncio.to_thread(_set_map_sync, profile_id, mapping)


def _delete_sync(profile_id: int) -> bool:
    cur = _conn().execute("DELETE FROM wwyd_connections WHERE profile_id=?", (profile_id,))
    return cur.rowcount > 0


async def delete_wwyd_connection(profile_id: int) -> bool:
    return await asyncio.to_thread(_delete_sync, profile_id)


def _connected_sync() -> list[tuple[int, str]]:
    return [
        (int(r[0]), r[1])
        for r in _conn().execute(
            "SELECT w.profile_id, p.name FROM wwyd_connections w"
            " JOIN speaker_profiles p ON p.id = w.profile_id ORDER BY w.profile_id"
        ).fetchall()
    ]


async def list_wwyd_connected_profiles() -> list[tuple[int, str]]:
    """``(profile_id, name)`` of every profile with a wwyd connection."""
    return await asyncio.to_thread(_connected_sync)


# ── OAuth pending ───────────────────────────────────────────────────────


def _put_pending_sync(state: str, profile_id: int, fields: dict) -> None:
    c = _conn()
    now = time.time()
    c.execute("DELETE FROM wwyd_oauth_pending WHERE expires_at < ?", (now,))
    c.execute("DELETE FROM wwyd_oauth_pending WHERE profile_id=?", (profile_id,))
    c.execute(
        "INSERT INTO wwyd_oauth_pending (state, profile_id, base_url, resource, client_id,"
        " code_verifier, redirect_uri, token_endpoint, revocation_endpoint, expires_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            state, profile_id, fields["base_url"], fields["resource"], fields["client_id"],
            fields["code_verifier"], fields["redirect_uri"], fields["token_endpoint"],
            fields.get("revocation_endpoint"), now + PENDING_TTL_S,
        ),
    )


async def put_wwyd_pending(state: str, profile_id: int, **fields) -> None:
    await asyncio.to_thread(_put_pending_sync, state, profile_id, fields)


def _pop_pending_sync(state: str) -> dict | None:
    c = _conn()
    row = c.execute(
        "SELECT profile_id, base_url, resource, client_id, code_verifier, redirect_uri,"
        " token_endpoint, revocation_endpoint, expires_at"
        " FROM wwyd_oauth_pending WHERE state=?",
        (state,),
    ).fetchone()
    # Single use, whatever happens next.
    c.execute("DELETE FROM wwyd_oauth_pending WHERE state=?", (state,))
    if row is None or row[8] < time.time():
        return None
    keys = ("profile_id", "base_url", "resource", "client_id", "code_verifier",
            "redirect_uri", "token_endpoint", "revocation_endpoint")
    return dict(zip(keys, row[:8]))


async def pop_wwyd_pending(state: str) -> dict | None:
    return await asyncio.to_thread(_pop_pending_sync, state)
