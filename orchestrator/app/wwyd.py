"""
Client side of the wwyd persona server ("what would you do").

wwyd speaks MCP over Streamable HTTP at ``<base>/mcp`` and is its own
OAuth 2.1 authorization server (discovery, dynamic client registration,
PKCE S256, refresh-token rotation, revocation).  An API key
(``wwyd_<prefix>_<secret>``) is the alternative for owners who prefer to
paste one.  Each speaker profile connects its own wwyd; the connection
lives in ``storage/wwyd.py``.

This module is the only place that sees tokens.  Nothing here logs a
token, a request body or a tool result: tool arguments carry the
owner's words.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from urllib.parse import urlencode, urlsplit

import httpx

from .storage import get_wwyd_connection, update_wwyd_tokens

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-11-25"
SCOPE = "owner"
CLIENT_NAME = "Voice Assistant"
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
# Predict makes up to two LLM calls on the wwyd side.
_TOOL_TIMEOUT = httpx.Timeout(90.0, connect=5.0)
_REFRESH_MARGIN_S = 60
# Pre-1.0 wwyd bumps the minor on breaking changes (docs/clients.md § Versioning).
COMPATIBLE_SERIES = (0, 2)
# Test hook: an httpx transport standing in for the wwyd server.
_TRANSPORT: httpx.AsyncBaseTransport | None = None


def _client(timeout: httpx.Timeout = _TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, transport=_TRANSPORT)


class WwydError(Exception):
    """A wwyd call failed.  ``code`` is wwyd's stable error code when it sent one
    (``forbidden``, ``invalid``, ``rate_limited`` …), else ``transport`` / ``auth``
    / ``not_connected`` / ``protocol``."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url.endswith("/mcp"):
        url = url[: -len("/mcp")]
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise WwydError("invalid", "wwyd URL must be http(s)://host[:port]")
    if parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
        # Bearer tokens over plain HTTP only on loopback.
        raise WwydError("invalid", "wwyd URL must use https unless it is on localhost")
    return url


# ── OAuth ───────────────────────────────────────────────────────────────


async def discover(base_url: str) -> dict:
    """Protected-resource + authorization-server metadata (RFC 9728 / 8414)."""
    async with _client() as c:
        try:
            r = await c.get(f"{base_url}/.well-known/oauth-protected-resource/mcp")
            prm = r.json() if r.status_code == 200 else {}
            issuer = (prm.get("authorization_servers") or [base_url])[0].rstrip("/")
            r = await c.get(f"{issuer}/.well-known/oauth-authorization-server")
            r.raise_for_status()
            asm = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WwydError("transport", f"wwyd discovery failed: {exc.__class__.__name__}") from exc
    for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint"):
        if not asm.get(key):
            raise WwydError("protocol", f"wwyd metadata has no {key}")
    return {
        "resource": prm.get("resource") or f"{base_url}/mcp",
        "authorization_endpoint": asm["authorization_endpoint"],
        "token_endpoint": asm["token_endpoint"],
        "registration_endpoint": asm["registration_endpoint"],
        "revocation_endpoint": asm.get("revocation_endpoint"),
    }


async def register_client(registration_endpoint: str, redirect_uri: str) -> str:
    """Dynamic client registration (RFC 7591) as a public client."""
    body = {
        "client_name": CLIENT_NAME,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": SCOPE,
    }
    async with _client() as c:
        try:
            r = await c.post(registration_endpoint, json=body)
        except httpx.HTTPError as exc:
            raise WwydError("transport", f"registration failed: {exc.__class__.__name__}") from exc
    if r.status_code not in (200, 201):
        detail = ""
        try:
            body = r.json()
            detail = body.get("error_description") or body.get("error") or ""
        except ValueError:
            pass
        # e.g. an http redirect URI on a server that forbids insecure redirects.
        raise WwydError("protocol", f"wwyd refused to register this assistant: {detail or f'HTTP {r.status_code}'}")
    return r.json()["client_id"]


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def authorize_url(meta: dict, client_id: str, redirect_uri: str, state: str, challenge: str) -> str:
    q = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": meta["resource"],
    }
    return f"{meta['authorization_endpoint']}?{urlencode(q)}"


def _token_fields(payload: dict) -> dict:
    expires_in = payload.get("expires_in")
    return {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token"),
        "expires_at": time.time() + float(expires_in) if expires_in else None,
    }


async def _token_request(token_endpoint: str, form: dict) -> dict:
    async with _client() as c:
        try:
            r = await c.post(token_endpoint, data=form)
        except httpx.HTTPError as exc:
            raise WwydError("transport", f"token request failed: {exc.__class__.__name__}") from exc
    if r.status_code != 200:
        err = ""
        try:
            err = r.json().get("error", "")
        except ValueError:
            pass
        raise WwydError("auth", f"token endpoint refused: {err or r.status_code}")
    return _token_fields(r.json())


async def exchange_code(pending: dict, code: str) -> dict:
    return await _token_request(pending["token_endpoint"], {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": pending["redirect_uri"],
        "client_id": pending["client_id"],
        "code_verifier": pending["code_verifier"],
        "resource": pending["resource"],
    })


async def revoke(conn: dict) -> bool:
    """Best effort: revoke the grant or the API key at wwyd (RFC 7009)."""
    if conn.get("auth_kind") == "oauth":
        endpoint = conn.get("revocation_endpoint")
        token = conn.get("refresh_token") or conn.get("access_token")
        hint = "refresh_token" if conn.get("refresh_token") else "access_token"
        form = {"token": token, "token_type_hint": hint, "client_id": conn.get("client_id") or ""}
    else:
        # API keys need no client authentication.
        endpoint = conn.get("revocation_endpoint") or f"{conn['base_url']}/revoke"
        form = {"token": conn.get("access_token")}
    if not endpoint or not form["token"]:
        return False
    async with _client() as c:
        try:
            r = await c.post(endpoint, data=form)
        except httpx.HTTPError:
            log.warning("wwyd: revoke request failed")
            return False
    return r.status_code == 200


async def _fresh_access_token(profile_id: int, conn: dict, *, force: bool = False) -> str:
    if conn["auth_kind"] != "oauth" or not conn.get("refresh_token"):
        return conn["access_token"]
    exp = conn.get("expires_at")
    if not force and (exp is None or exp - time.time() > _REFRESH_MARGIN_S):
        return conn["access_token"]
    fields = await _token_request(conn["token_endpoint"], {
        "grant_type": "refresh_token",
        "refresh_token": conn["refresh_token"],
        "client_id": conn.get("client_id") or "",
        "resource": conn["resource"],
    })
    # Refresh tokens rotate on every use; keep the old one only if none came back.
    refresh = fields["refresh_token"] or conn["refresh_token"]
    await update_wwyd_tokens(profile_id, fields["access_token"], refresh, fields["expires_at"])
    conn.update(access_token=fields["access_token"], refresh_token=refresh,
                expires_at=fields["expires_at"])
    return fields["access_token"]


# ── MCP over Streamable HTTP ────────────────────────────────────────────


def _parse_rpc(r: httpx.Response, req_id: int) -> dict:
    ctype = r.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        for line in r.text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                msg = json.loads(line[5:].strip())
            except ValueError:
                continue
            if msg.get("id") == req_id:
                return msg
        raise WwydError("protocol", "no JSON-RPC response in event stream")
    try:
        return r.json()
    except ValueError as exc:
        raise WwydError("protocol", "non-JSON response") from exc


def check_version(version: str | None) -> None:
    """Refuse servers outside the wwyd series this client was written for."""
    try:
        major, minor = (int(x) for x in str(version or "").split("-")[0].split(".")[:2])
    except ValueError:
        major = minor = -1
    if (major, minor) != COMPATIBLE_SERIES:
        want = ".".join(map(str, COMPATIBLE_SERIES))
        raise WwydError("version", f"wwyd {version or '(unknown version)'} is not supported; "
                                   f"this assistant works with wwyd {want}.x")


async def _mcp_call(resource: str, token: str, method: str, params: dict) -> dict:
    return (await _mcp_session(resource, token, method, params))[1]


async def _mcp_session(resource: str, token: str, method: str, params: dict) -> tuple[dict, dict]:
    """initialize + one call; returns (initialize result, call result)."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    async with _client(_TOOL_TIMEOUT) as c:
        try:
            r = await c.post(resource, headers=headers, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "voice-assistant", "version": "1"},
                },
            })
            if r.status_code in (401, 403):
                raise WwydError("auth", f"HTTP {r.status_code}")
            if r.status_code != 200:
                raise WwydError("protocol", f"initialize: HTTP {r.status_code}")
            init = _parse_rpc(r, 1)
            if "error" in init:
                raise WwydError("protocol", "initialize refused")
            session = r.headers.get("mcp-session-id")
            if session:
                headers["Mcp-Session-Id"] = session
            headers["MCP-Protocol-Version"] = init.get("result", {}).get(
                "protocolVersion", PROTOCOL_VERSION)
            await c.post(resource, headers=headers, json={
                "jsonrpc": "2.0", "method": "notifications/initialized"})
            r = await c.post(resource, headers=headers, json={
                "jsonrpc": "2.0", "id": 2, "method": method, "params": params})
            if r.status_code in (401, 403):
                raise WwydError("auth", f"HTTP {r.status_code}")
            if r.status_code != 200:
                raise WwydError("protocol", f"{method}: HTTP {r.status_code}")
            msg = _parse_rpc(r, 2)
            if session:
                try:
                    await c.delete(resource, headers=headers)
                except httpx.HTTPError:
                    pass
        except httpx.HTTPError as exc:
            raise WwydError("transport", exc.__class__.__name__) from exc
    if "error" in msg:
        raise WwydError("protocol", str(msg["error"].get("message", "rpc error"))[:200])
    return init.get("result") or {}, msg.get("result") or {}


def _tool_payload(result: dict) -> dict:
    payload = result.get("structuredContent")
    if payload is None:
        try:
            payload = json.loads((result.get("content") or [{}])[0].get("text") or "{}")
        except (ValueError, AttributeError, IndexError):
            payload = {}
    if result.get("isError"):
        err = (payload or {}).get("error") or {}
        raise WwydError(err.get("code") or "tool_error", err.get("message") or "")
    return payload or {}


async def call_tool(profile_id: int, name: str, arguments: dict) -> dict:
    """Call a wwyd MCP tool as ``profile_id``'s connection; returns the
    structured result.  Refreshes the access token once on 401."""
    conn = await get_wwyd_connection(profile_id)
    if conn is None:
        raise WwydError("not_connected")
    token = await _fresh_access_token(profile_id, conn)
    params = {"name": name, "arguments": arguments}
    try:
        return _tool_payload(await _mcp_call(conn["resource"], token, "tools/call", params))
    except WwydError as exc:
        if exc.code != "auth" or conn["auth_kind"] != "oauth":
            raise
    token = await _fresh_access_token(profile_id, conn, force=True)
    return _tool_payload(await _mcp_call(conn["resource"], token, "tools/call", params))


async def probe(resource: str, token: str) -> list[str]:
    """Check a new connection: the server version must be compatible;
    returns the tool names the token can see."""
    init, result = await _mcp_session(resource, token, "tools/list", {})
    check_version((init.get("serverInfo") or {}).get("version"))
    return [t.get("name") for t in result.get("tools") or []]


async def list_audiences(profile_id: int) -> list[dict]:
    """wwyd audiences: ``[{name, description, is_default, claim_count}]``."""
    out = await call_tool(profile_id, "list_audiences", {})
    return list(out.get("audiences") or [])


_CATEGORY_TTL_S = 300
_categories_cache: dict[int, tuple[float, list[str]]] = {}


async def list_category_paths(profile_id: int) -> list[str]:
    """Flat category paths of the persona (they are per persona and the owner
    can add more); cached briefly per profile."""
    hit = _categories_cache.get(profile_id)
    if hit and time.monotonic() - hit[0] < _CATEGORY_TTL_S:
        return hit[1]
    out = await call_tool(profile_id, "list_categories", {})
    paths: list[str] = []
    stack = list(out.get("categories") or [])
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict) and node.get("path"):
            paths.append(str(node["path"]))
            stack.extend(node.get("children") or [])
    _categories_cache[profile_id] = (time.monotonic(), paths)
    return paths


def local_timezone() -> str | None:
    """IANA name of this machine's zone, the one its calendar is in."""
    try:
        name = os.readlink("/etc/localtime").partition("zoneinfo/")[2]
    except OSError:
        return None
    return name or None


async def sync_timezone(profile_id: int) -> None:
    """A persona still on wwyd's default UTC gets this machine's zone, so
    "tomorrow" in ``predict`` is the owner's tomorrow; an owner's own choice stays."""
    tz = local_timezone()
    if not tz or tz == "UTC":
        return
    try:
        if ((await call_tool(profile_id, "get_settings", {})).get("timezone") or "UTC") == "UTC":
            await call_tool(profile_id, "update_settings", {"timezone": tz})
    except WwydError as exc:
        log.info("wwyd: timezone not set for profile %d: %s", profile_id, exc.code)


async def is_connected(profile_id: int | None) -> bool:
    return profile_id is not None and await get_wwyd_connection(profile_id) is not None
