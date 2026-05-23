"""
Orchestrator-side client for one (or many) desktop-agents.

Surface area
────────────
Two layers of primitives:

  1. **Core (used today by computer_use / look_at_screen)**

     • :func:`run_applescript` — run an AppleScript on the agent.
       Accepts an optional ``category=`` argument so the orchestrator
       can enforce a stricter read-only verb allowlist when the
       caller is the read-only ``computer_use`` tool.
     • :func:`run_pyautogui` / :func:`run_key` — low-level input
       primitives (used by the ``desktop`` tool for explicit mode
       calls).
     • :func:`screenshot` — full-screen PNG, used by
       ``look_at_screen``.

  2. **Vision-loop infrastructure (phase 2, not wired yet)**

     The Wave 3 ``computer_use`` rework went LLM-generated-AppleScript
     first; vision-driven UI navigation was deferred.  These helpers
     stay in place so the phase-2 vision-loop fallback can compose
     them without re-implementing the wire layer:

     • :func:`resolve_default_app` — ask the agent which app the OS
       would launch for a category (mail / browser / calendar /
       files).  Backed by JXA NSWorkspace on macOS.
     • :func:`cursor_activity` — read the host's cursor position +
       idle-seconds to decide whether a vision-driven UI op would
       collide with the user's own typing.
     • :func:`screenshot` doubles as the vision-rail capture source.

Architecture (Wave 2 — still valid)
───────────────────────────────────
  • A typed :class:`AgentInfo` registry keyed by ``agent_id`` —
    multiple agents can register with the orchestrator and be addressed
    individually by callers.  All public functions accept an optional
    ``agent_id``; ``None`` resolves to the default agent (the one
    flagged ``default=True``, else most-recently-reachable, else first
    in env order).

  • Per-agent capability cache + reachability flag.  A background
    health-poll task refreshes each agent every
    ``DESKTOP_HEALTHPOLL_INTERVAL_S`` seconds, so :func:`is_reachable`
    returns the latest known state without paying an HTTP round-trip.
    Tools fast-fail with a friendly i18n message rather than blocking
    on a 5 s timeout per call when an agent is offline.

  • Optional **reverse-WSS** transport.  An agent behind NAT can dial
    the orchestrator (``/v1/agent/connect``) and stay connected; the
    orchestrator pushes calls down the socket via
    :mod:`.agent_proxy`.  This client picks the WSS path automatically
    when the agent is registered via reverse mode — callers don't care
    which transport is in use.

Configuration knobs (env, read at import time)
──────────────────────────────────────────────
``DESKTOP_AGENTS``
    JSON list of ``{agent_id, url, token, default?, mode?}``.  When
    set, becomes the registry verbatim.

``DESKTOP_URL`` / ``DESKTOP_TOKEN``
    Back-compat fallback used when ``DESKTOP_AGENTS`` is absent —
    synthesises a one-entry list ``[{"agent_id": "default", "url":
    DESKTOP_URL, "token": DESKTOP_TOKEN, "default": True}]`` so
    existing single-agent installs keep working with zero changes.

``DESKTOP_HEALTHPOLL_INTERVAL_S``
    Background-poll cadence in seconds (default 30).  Bumping this
    trades freshness for fewer agent round-trips.

Threading model: all public functions are async.  The background poll
runs as one ``asyncio.Task`` owned by the orchestrator's lifespan;
``shutdown_desktop`` cancels it cleanly on app exit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


# ── Module-level env config ────────────────────────────────────────────
#
# We snapshot env at import time — same pattern the rest of the
# orchestrator uses (memory.py, search.py, etc.).  Tests can override
# at runtime via :func:`reload_agents_from_env`.

DESKTOP_URL = os.environ.get("DESKTOP_URL", "http://localhost:9877")
DESKTOP_TOKEN = os.environ.get("DESKTOP_TOKEN", "")
DESKTOP_AGENTS_RAW = os.environ.get("DESKTOP_AGENTS", "")
try:
    DESKTOP_HEALTHPOLL_INTERVAL_S = max(
        5.0, float(os.environ.get("DESKTOP_HEALTHPOLL_INTERVAL_S", "30"))
    )
except ValueError:
    DESKTOP_HEALTHPOLL_INTERVAL_S = 30.0
    log.warning(
        "desktop_client: bad DESKTOP_HEALTHPOLL_INTERVAL_S=%r — using 30 s",
        os.environ.get("DESKTOP_HEALTHPOLL_INTERVAL_S"),
    )


# Default per-call timeouts.  AppleScript / pyautogui calls can take a
# few seconds to drive an app through a menu; longer than a few seconds
# usually means something blocked (user hasn't granted Accessibility).
# Remote (Tailscale) agents add ~50-200 ms RTT on top — already
# absorbed by the 30 s read budget so no tweak needed.
_TIMEOUT = httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=10.0)

# Capabilities cache TTL on each AgentInfo.  Background poll refreshes
# it every DESKTOP_HEALTHPOLL_INTERVAL_S anyway; this TTL is the
# upper bound when the poll task isn't running (e.g. /dev/respond
# during tests).
_CAPS_TTL_S = 60.0


class DesktopUnavailable(RuntimeError):
    """Raised when the agent is unreachable / 401 / 503."""


# ── AgentInfo registry ─────────────────────────────────────────────────


@dataclass
class AgentInfo:
    """One row in the multi-agent registry.

    Holds connection params + the latest cached health/capabilities
    snapshot.  Mutated in-place by the background poll task and by
    ``capabilities()`` on demand.
    """

    agent_id: str
    url: str
    token: str
    default: bool = False
    # "http"   — orchestrator-initiates HTTP calls to the agent.
    # "reverse"— agent dialled us via WSS; we push calls through
    #            agent_proxy.AgentConnection.  The registry stays
    #            authoritative for url/token/default — mode just
    #            switches the transport at call time.
    mode: str = "http"

    # Latest /v1/capabilities payload.  Empty dict when never fetched
    # (cold start) or when every probe so far has failed.
    capabilities_cache: dict[str, Any] = field(default_factory=dict)
    capabilities_cached_at: float = 0.0
    # True iff the most recent poll/probe completed.  Goes False on
    # any HTTPError or non-2xx; flips back True on the next successful
    # poll.  Tools check this before making a call to avoid the 3 s
    # connect-timeout penalty per dead agent.
    reachable: bool = False
    # Wall-clock of the last successful capabilities/health response.
    # Surfaced in /api/agents so the UI can render «last seen 2 min
    # ago» when the agent is currently unreachable.
    last_seen: float = 0.0


# Registry — module-level mutable dict.  ``init_desktop`` populates
# from env on startup; ``reload_agents_from_env`` rebuilds it for tests.
_AGENTS: dict[str, AgentInfo] = {}

# Background-poll task handle.  Set by ``init_desktop``, cancelled by
# ``shutdown_desktop``.  None when the poll isn't running (tests).
_POLL_TASK: asyncio.Task | None = None


def reload_agents_from_env() -> None:
    """Re-parse env and rebuild the agent registry.

    Public so tests can monkey-patch env then call this to get a fresh
    registry without restarting the process.  Production code calls it
    once from ``init_desktop()``.

    Parsing rules:
      * If ``DESKTOP_AGENTS`` is set (non-empty), parse as JSON list.
        Each entry needs ``agent_id`` and ``url``; ``token`` defaults
        to the env ``DESKTOP_TOKEN``; ``default`` defaults to False;
        ``mode`` defaults to "http".
      * Else fall back to single-entry list synthesised from
        ``DESKTOP_URL``/``DESKTOP_TOKEN`` with id="default",
        default=True — preserves existing single-agent installs.
    """
    global _AGENTS
    raw = os.environ.get("DESKTOP_AGENTS", "")
    if raw.strip():
        try:
            spec = json.loads(raw)
            if not isinstance(spec, list):
                raise ValueError("DESKTOP_AGENTS must be a JSON list")
        except (json.JSONDecodeError, ValueError) as exc:
            log.error(
                "desktop_client: bad DESKTOP_AGENTS env (%s) — falling back to single-agent mode",
                exc,
            )
            spec = []
    else:
        spec = []

    if not spec:
        # Back-compat: synthesise one entry from the legacy vars so
        # nothing changes for existing installs.
        spec = [{
            "agent_id": "default",
            "url": os.environ.get("DESKTOP_URL", "http://localhost:9877"),
            "token": os.environ.get("DESKTOP_TOKEN", ""),
            "default": True,
        }]

    new: dict[str, AgentInfo] = {}
    for entry in spec:
        agent_id = str(entry.get("agent_id") or "").strip()
        url = str(entry.get("url") or "").strip()
        if not agent_id or not url:
            log.warning("desktop_client: skipping malformed agent entry: %r", entry)
            continue
        new[agent_id] = AgentInfo(
            agent_id=agent_id,
            url=url,
            token=str(entry.get("token") or os.environ.get("DESKTOP_TOKEN", "")),
            default=bool(entry.get("default", False)),
            mode=str(entry.get("mode") or "http"),
        )
    if not new:
        log.warning("desktop_client: no agents registered — desktop tools will all degrade")
    _AGENTS = new


# Populate at import so module-level constants (e.g. tools registering
# at import time) see the registry without an explicit init call.
reload_agents_from_env()


def list_agents() -> list[AgentInfo]:
    """Snapshot of every registered agent — used by /api/agents UI."""
    return list(_AGENTS.values())


def get_agent(agent_id: str | None) -> AgentInfo | None:
    """Resolve a (possibly None) agent_id to an :class:`AgentInfo`.

    Selection order when ``agent_id`` is None:
      1. The agent flagged ``default=True``
      2. Else the most-recently-reachable agent (highest ``last_seen``)
      3. Else the first one in env order
    Returns None when the registry is empty.
    """
    if agent_id is not None:
        return _AGENTS.get(agent_id)
    if not _AGENTS:
        return None
    # 1. explicit default flag
    for info in _AGENTS.values():
        if info.default:
            return info
    # 2. most-recently-seen reachable agent
    reachable = [a for a in _AGENTS.values() if a.reachable]
    if reachable:
        return max(reachable, key=lambda a: a.last_seen)
    # 3. first in env order
    return next(iter(_AGENTS.values()))


def register_reverse_agent(agent_id: str, token: str) -> AgentInfo:
    """Plug a freshly-connected reverse-WSS agent into the registry.

    Called by :mod:`.agent_proxy` after a successful WSS hello.  If
    the agent_id already exists (e.g. a pre-configured entry), we flip
    its mode to "reverse" and update the token; otherwise we mint a
    new row.  This way the operator can pre-list a known agent in
    ``DESKTOP_AGENTS`` to mark it ``default=True`` and the WSS handshake
    just upgrades the transport.
    """
    existing = _AGENTS.get(agent_id)
    if existing is not None:
        existing.mode = "reverse"
        existing.token = token or existing.token
        return existing
    info = AgentInfo(
        agent_id=agent_id,
        url="",  # reverse-mode agents don't have a callable URL
        token=token,
        default=False,
        mode="reverse",
    )
    _AGENTS[agent_id] = info
    return info


def unregister_reverse_agent(agent_id: str) -> None:
    """Remove or demote a reverse-mode agent on disconnect.

    If the agent was added by the WSS handshake (no pre-existing
    config), drop it entirely.  If it was a pre-configured entry just
    flip the mode back to "http" so the orchestrator can still try to
    reach it via the original URL.
    """
    info = _AGENTS.get(agent_id)
    if info is None:
        return
    if not info.url:
        # WSS-only entry — fully remove.
        del _AGENTS[agent_id]
    else:
        info.mode = "http"
        info.reachable = False


# ── Auth header builder ────────────────────────────────────────────────


def _headers(info: AgentInfo) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if info.token:
        h["X-Desktop-Token"] = info.token
    return h


# ── HTTP helpers (per-agent) ────────────────────────────────────────────


async def _probe_health(info: AgentInfo) -> dict | None:
    """Probe one agent's /v1/health.  Returns None when it's down.

    Side-effect: updates ``info.reachable`` + ``info.last_seen``.  This
    is the cheap unauthenticated probe — separate from
    :func:`capabilities` which carries the token and is the source of
    truth for capability flags.
    """
    if not info.url:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{info.url}/v1/health")
            r.raise_for_status()
            data = r.json()
        info.reachable = True
        info.last_seen = time.time()
        return data
    except Exception as exc:
        info.reachable = False
        log.debug("desktop_client[%s]: health probe failed: %s", info.agent_id, exc)
        return None


async def health(agent_id: str | None = None) -> dict | None:
    """Probe one agent's health.  Returns None if it's down."""
    info = get_agent(agent_id)
    if info is None:
        return None
    return await _probe_health(info)


async def capabilities(
    *,
    agent_id: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return the chosen agent's capability map, cached per-agent.

    Shape matches the agent's /v1/capabilities response:
        {
            "agent_id": "...",
            "platform": "macos" | "windows" | "linux",
            "capabilities": {
                "screenshot": True, "applescript": True,
                "pyautogui": True, "hotkey": True,
                "default_apps_resolver": True, "cursor_activity": True,
            },
            "version": "1.1.0",
        }

    On transport / auth failure we return the last-known good cache if
    it's still warm + flag ``unreachable=True`` so callers can decide
    to render «agent offline» vs «mail not available».  On a totally
    cold start we return an empty dict so callers treat "no caps" as
    "no features advertised".
    """
    info = get_agent(agent_id)
    if info is None:
        # No agent in registry at all.
        return {}

    fresh = (time.time() - info.capabilities_cached_at) < _CAPS_TTL_S
    if not force_refresh and info.capabilities_cache and fresh:
        return info.capabilities_cache

    if not info.url:
        # Reverse-mode agent that hasn't reported its caps yet.
        return info.capabilities_cache

    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{info.url}/v1/capabilities",
                headers=_headers(info),
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        # Soft-fail: keep the stale cache if we have one; mark as
        # unreachable so tools can distinguish "no daemon" from "no
        # capability".
        log.debug(
            "desktop_client[%s]: capabilities fetch failed: %s",
            info.agent_id, exc,
        )
        info.reachable = False
        if info.capabilities_cache:
            return {**info.capabilities_cache, "unreachable": True}
        return {"unreachable": True}

    info.capabilities_cache = data
    info.capabilities_cached_at = time.time()
    info.reachable = True
    info.last_seen = time.time()
    return data


async def has_capability(feature: str, *, agent_id: str | None = None) -> bool:
    """Convenience: True iff the agent advertises ``feature``."""
    info_dict = await capabilities(agent_id=agent_id)
    return bool((info_dict.get("capabilities") or {}).get(feature))


def has_capability_cached(
    feature: str, *, agent_id: str | None = None,
) -> bool | None:
    """Sync cached capability lookup — never triggers an HTTP fetch.

    Returns:
      * ``True``  — cache says the agent supports ``feature``.
      * ``False`` — cache says it does NOT.
      * ``None``  — cache hasn't been populated yet (first call after
        boot, or a reverse-WSS agent that hasn't handshaked). The
        caller should fall back to its optimistic path; the next
        ``_health_poll_loop`` tick (≤30 s by default) will populate it.

    Used by fast-path tools (e.g. ``computer_use``) to skip an expensive
    LLM round-trip when the cache definitively rules out an engine.
    """
    info = get_agent(agent_id)
    if info is None or not info.capabilities_cache:
        return None
    return bool((info.capabilities_cache.get("capabilities") or {}).get(feature))


def is_reachable(agent_id: str | None = None) -> bool:
    """Non-blocking reachability check from the latest poll result.

    Tools call this BEFORE attempting an HTTP request so a known-dead
    agent doesn't burn 3 s of connect-timeout on every voice turn.
    Source of truth is the ``info.reachable`` flag maintained by the
    background poll task and the on-demand ``capabilities`` call.
    """
    info = get_agent(agent_id)
    if info is None:
        return False
    return info.reachable


# ── Background health-poll ─────────────────────────────────────────────


async def _health_poll_loop() -> None:
    """Background task: refresh every agent's capabilities periodically.

    Started by :func:`init_desktop` and cancelled by
    :func:`shutdown_desktop`.  Each tick walks every HTTP-mode agent
    and re-fetches /v1/capabilities; failures flip the agent to
    ``reachable=False`` but DON'T evict the cached caps (so a brief
    blip doesn't wipe the LLM's tool surface).  Reverse-mode agents
    are skipped — their liveness is the WSS connection itself,
    maintained by :mod:`.agent_proxy`.

    Polling cadence comes from ``DESKTOP_HEALTHPOLL_INTERVAL_S`` env;
    capped at the lower end to 5 s so a runaway config can't flood the
    agent.
    """
    try:
        while True:
            await asyncio.sleep(DESKTOP_HEALTHPOLL_INTERVAL_S)
            for info in list(_AGENTS.values()):
                if info.mode == "reverse":
                    continue
                # force_refresh so the poll always gets fresh caps —
                # the cache TTL is fine for tool-call paths, but the
                # poll's whole job is to keep that cache warm.
                try:
                    await capabilities(agent_id=info.agent_id, force_refresh=True)
                except Exception:
                    log.debug(
                        "desktop_client[%s]: poll iteration crashed",
                        info.agent_id, exc_info=True,
                    )
    except asyncio.CancelledError:
        log.info("desktop_client: health-poll task stopped")
        raise


# ── Lifecycle hooks ────────────────────────────────────────────────────


async def init_desktop() -> None:
    """Probe at orchestrator startup + start the background health-poll.

    Non-fatal — voice still works without desktop automation.  Logs
    once so the operator knows what state things are in.  Also primes
    each agent's capabilities cache so the first tool call doesn't
    pay a cold HTTP round-trip.
    """
    global _POLL_TASK

    # Iterate over a snapshot in case the first probe somehow mutates
    # the registry (it shouldn't, but defensive).
    for info in list(_AGENTS.values()):
        if info.mode == "reverse":
            log.info(
                "desktop_client[%s]: reverse-mode entry (will register on WSS connect)",
                info.agent_id,
            )
            continue
        caps_info = await capabilities(agent_id=info.agent_id, force_refresh=True)
        if not info.reachable:
            log.warning(
                "desktop_client[%s]: %s unreachable — start ./desktop-agent/start.sh on the host",
                info.agent_id, info.url,
            )
            continue
        caps = caps_info.get("capabilities") or {}
        have = ", ".join(name for name, ok in caps.items() if ok) or "<none>"
        log.info(
            "desktop_client[%s]: ready at %s (platform=%s version=%s caps=%s)",
            info.agent_id, info.url,
            caps_info.get("platform"),
            caps_info.get("version"),
            have,
        )

    if _POLL_TASK is None or _POLL_TASK.done():
        loop = asyncio.get_event_loop()
        _POLL_TASK = loop.create_task(_health_poll_loop())
        log.info(
            "desktop_client: started health-poll task (every %.0f s)",
            DESKTOP_HEALTHPOLL_INTERVAL_S,
        )


async def shutdown_desktop() -> None:
    """Cancel the background health-poll task on app exit."""
    global _POLL_TASK
    if _POLL_TASK is not None and not _POLL_TASK.done():
        _POLL_TASK.cancel()
        try:
            await _POLL_TASK
        except (asyncio.CancelledError, Exception):
            pass
    _POLL_TASK = None


# ── Per-call helpers (dispatch HTTP vs reverse) ─────────────────────────


async def _reverse_call(info: AgentInfo, method: str, params: dict, *, timeout: float) -> Any:
    """Dispatch one call through the reverse-WSS connection.

    Imported lazily so unit tests that don't exercise reverse mode
    can avoid pulling in the agent_proxy module (and its WS state).
    """
    from . import agent_proxy
    conn = agent_proxy.get_connection(info.agent_id)
    if conn is None:
        raise DesktopUnavailable(f"reverse: no connection for {info.agent_id}")
    return await conn.call(method, params, timeout=timeout)


# ── AppleScript / pyautogui / hotkey / screenshot ──────────────────────


async def run_applescript(
    script: str,
    *,
    timeout: float = 30.0,
    agent_id: str | None = None,
    category: str | None = None,
) -> dict:
    """Execute AppleScript on the chosen agent's host.

    Returns ``{exit, stdout, stderr, elapsed_ms}``.  Raises
    ``DesktopUnavailable`` on transport / auth failure so the LLM tool
    layer can map it to a friendly localised message.

    ``category`` (mail / browser / calendar / files) is consulted by
    the orchestrator-side ``desktop`` tool's risk classifier — passed
    through here purely so any future server-side audit log can record
    the declared category.  The wire payload doesn't yet include it
    because the agent is platform-neutral and doesn't know per-category
    semantics; that's enforced before the call by the tool layer.
    """
    info = get_agent(agent_id)
    if info is None:
        raise DesktopUnavailable("no agent configured")
    _ = category  # reserved for future agent-side audit; see docstring
    if info.mode == "reverse":
        return await _reverse_call(
            info, "applescript", {"script": script, "timeout": timeout},
            timeout=timeout + 5,
        )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.post(
                f"{info.url}/v1/applescript",
                json={"script": script, "timeout": timeout},
                headers=_headers(info),
            )
    except httpx.HTTPError as exc:
        info.reachable = False
        raise DesktopUnavailable(f"transport: {exc.__class__.__name__}") from exc
    if r.status_code == 401:
        raise DesktopUnavailable("auth failed — check DESKTOP_TOKEN")
    if r.status_code == 503:
        raise DesktopUnavailable("daemon: AppleScript not available")
    if r.status_code == 504:
        raise DesktopUnavailable("daemon: AppleScript timeout")
    r.raise_for_status()
    info.reachable = True
    info.last_seen = time.time()
    return r.json()


async def run_pyautogui(payload: dict[str, Any], *, agent_id: str | None = None) -> dict:
    """Forward a structured pyautogui action."""
    info = get_agent(agent_id)
    if info is None:
        raise DesktopUnavailable("no agent configured")
    if info.mode == "reverse":
        return await _reverse_call(info, "pyautogui", payload, timeout=35.0)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.post(
                f"{info.url}/v1/pyautogui",
                json=payload,
                headers=_headers(info),
            )
    except httpx.HTTPError as exc:
        info.reachable = False
        raise DesktopUnavailable(f"transport: {exc.__class__.__name__}") from exc
    if r.status_code in (401, 503):
        raise DesktopUnavailable(f"daemon: {r.text}")
    r.raise_for_status()
    info.reachable = True
    info.last_seen = time.time()
    return r.json()


async def run_key(keys: list[str], *, agent_id: str | None = None) -> dict:
    """Press a keyboard shortcut, e.g. ``["cmd","space"]``."""
    info = get_agent(agent_id)
    if info is None:
        raise DesktopUnavailable("no agent configured")
    if info.mode == "reverse":
        return await _reverse_call(info, "key", {"keys": keys}, timeout=35.0)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.post(
                f"{info.url}/v1/key",
                json={"keys": keys},
                headers=_headers(info),
            )
    except httpx.HTTPError as exc:
        info.reachable = False
        raise DesktopUnavailable(f"transport: {exc.__class__.__name__}") from exc
    if r.status_code in (401, 503):
        raise DesktopUnavailable(f"daemon: {r.text}")
    r.raise_for_status()
    info.reachable = True
    info.last_seen = time.time()
    return r.json()


async def screenshot(agent_id: str | None = None) -> bytes:
    """Capture the chosen agent's screen.  Returns raw PNG bytes."""
    info = get_agent(agent_id)
    if info is None:
        raise DesktopUnavailable("no agent configured")
    if info.mode == "reverse":
        result = await _reverse_call(info, "screenshot", {}, timeout=35.0)
        # Reverse transport returns the bytes b64-encoded inside a dict
        # (see agent-side handler in desktop-agent.py reverse_loop).
        import base64
        b64 = (result or {}).get("png_b64") if isinstance(result, dict) else None
        if not b64:
            raise DesktopUnavailable("reverse: malformed screenshot response")
        return base64.b64decode(b64)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(
                f"{info.url}/v1/screenshot",
                headers={"X-Desktop-Token": info.token} if info.token else {},
            )
    except httpx.HTTPError as exc:
        info.reachable = False
        raise DesktopUnavailable(f"transport: {exc.__class__.__name__}") from exc
    if r.status_code in (401, 503):
        raise DesktopUnavailable(f"daemon: {r.text}")
    r.raise_for_status()
    info.reachable = True
    info.last_seen = time.time()
    return r.content


async def submit_audit(event: str, *, agent_id: str | None = None, **payload: Any) -> None:
    """Log a non-execution event (e.g. allowlist denial).  Best-effort."""
    info = get_agent(agent_id)
    if info is None or not info.url or info.mode == "reverse":
        # Reverse-mode agents audit on their own side via the incoming
        # `call` message — no separate endpoint needed.
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            await c.post(
                f"{info.url}/v1/audit",
                json={"event": event, "payload": payload},
                headers=_headers(info),
            )
    except Exception as exc:
        log.debug("desktop_client[%s]: audit submit failed: %s", info.agent_id, exc)


# ── Default-app resolution + cursor-activity guard ─────────────────────


async def resolve_default_app(
    category: str,
    *,
    agent_id: str | None = None,
) -> dict | None:
    """Ask the chosen agent which app handles ``category`` by default.

    ``category`` ∈ {"mail", "browser", "calendar", "files"}.  Returns
    a dict ``{app_name, bundle_id, app_path, scriptable}`` or ``None``
    when no default is known (or the platform doesn't support the
    lookup).  Capability snapshots already include a pre-warmed
    ``default_apps`` map for the four known categories so we try that
    cache before paying an HTTP round-trip.

    Currently no orchestrator tool calls this helper directly — Wave
    3's ``computer_use`` rework went LLM-generated-AppleScript first
    (the model knows AppleScript dictionaries for every common app).
    Kept as a primitive for the phase-2 vision-loop fallback, which
    needs to know which app to activate before clicking around.
    """
    info = get_agent(agent_id)
    if info is None:
        raise DesktopUnavailable("no agent configured")
    # Hot path: pre-warmed cache shipped on the capabilities payload.
    pre = (info.capabilities_cache or {}).get("default_apps") or {}
    if category in pre and pre[category]:
        return pre[category]
    if info.mode == "reverse":
        payload = await _reverse_call(
            info, "default_app", {"category": category}, timeout=10.0,
        )
        # Empty dict from the agent → None to the caller.
        if not payload or not payload.get("app_name"):
            return None
        return payload
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(
                f"{info.url}/v1/default_app",
                params={"category": category},
                headers=_headers(info),
            )
    except httpx.HTTPError as exc:
        info.reachable = False
        raise DesktopUnavailable(f"transport: {exc.__class__.__name__}") from exc
    if r.status_code == 404:
        return None
    if r.status_code in (401, 503):
        raise DesktopUnavailable(f"daemon: {r.text}")
    r.raise_for_status()
    info.reachable = True
    info.last_seen = time.time()
    return r.json()


async def cursor_activity(
    *,
    agent_id: str | None = None,
) -> dict | None:
    """Read the chosen agent's cursor position + idle-seconds.

    Returns ``{x, y, idle_s, warm}`` or ``None`` if the agent doesn't
    support cursor tracking (no pyautogui → no poll task).  ``warm``
    is False until the agent's first cursor sample lands — callers
    should treat ``warm=False`` as "user activity unknown, allow the
    action" so the conflict-protection guard fails open rather than
    closed on agent boot.

    Currently no orchestrator tool calls this helper directly — Wave
    3's ``computer_use`` rework runs only AppleScript (no mouse
    contention with the user).  Kept for the phase-2 vision-loop
    fallback, which will refuse vision-driven UI ops while the user
    is actively at the keyboard.
    """
    info = get_agent(agent_id)
    if info is None:
        raise DesktopUnavailable("no agent configured")
    if info.mode == "reverse":
        return await _reverse_call(
            info, "cursor_activity", {}, timeout=5.0,
        )
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{info.url}/v1/cursor_activity",
                headers=_headers(info),
            )
    except httpx.HTTPError as exc:
        info.reachable = False
        raise DesktopUnavailable(f"transport: {exc.__class__.__name__}") from exc
    if r.status_code in (401, 503):
        return None
    r.raise_for_status()
    info.reachable = True
    info.last_seen = time.time()
    return r.json()
