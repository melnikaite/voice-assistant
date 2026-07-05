#!/usr/bin/env python3
"""
desktop-agent — host-side desktop-automation HTTP gateway.

Runs DIRECTLY ON THE HOST (not in Docker) so it can:
  • drive AppleScript (macOS) / pywinauto (Windows) / xdotool (Linux),
  • capture screenshots,
  • move the mouse / type via pyautogui,
  • resolve the user's default GUI app per task category
    (mail / browser / calendar / files),
  • report cursor-activity so the orchestrator can refuse vision-driven
    UI ops while the user is actively at the keyboard.

The voice-assistant orchestrator runs inside Docker, where it can do
NONE of those things — accessibility APIs and the GUI session live in
the user's login session, not in the container.  This daemon is the
bridge.  The orchestrator is platform-blind: it asks
``/v1/capabilities`` what's available, then issues HTTP calls and lets
the agent translate them to OS primitives.

Wave 3 reorientation
────────────────────
Previously the agent had mail-specific endpoints (``/v1/mail/search``,
``/v1/mail/read``) that read Mail.app's Envelope Index sqlite db and
.emlx files directly.  Wave 3 removes that ALTOGETHER in favour of a
universal computer-use layer in the orchestrator: the orchestrator
resolves the user's default mail app via ``/v1/default_app``, then
either drives the app's AppleScript dictionary or falls back to
screenshot + vision-driven UI navigation — same flow works for
browser, calendar, files, and anything else.  No per-app code lives
in the agent any more.

Architecture
────────────
Per-OS work is encapsulated in a :class:`Backend` subclass
(MacOSBackend / WindowsBackend / LinuxBackend) — one is auto-selected
at startup and stored as ``_BACKEND``.  HTTP routes only handle auth,
audit, and argument unpacking; the actual `subprocess`/`pyautogui`
calls live inside the backend.  Adding a new platform means adding
one subclass; adding a new capability means adding one abstract
method and one route.

Endpoints (all under /v1/):
  GET  /v1/health           — readiness + which engines + capabilities
  GET  /v1/capabilities     — agent_id, platform name, feature flags, version
  GET  /v1/platform         — OS / Python / preferred engine (legacy)
  POST /v1/applescript      — run AppleScript via osascript (macOS only)
  POST /v1/pyautogui        — structured cross-platform automation
  POST /v1/key              — keyboard shortcut (e.g. cmd+space)
  GET  /v1/screenshot            — full-screen PNG snapshot
  GET  /v1/camera                — single JPEG frame from the default camera device
  POST /v1/audit                 — write a free-form audit entry
  GET  /v1/default_app           — resolve the user's default app for a category
  GET  /v1/cursor_activity       — cursor position + idle-seconds (conflict guard)
  GET  /v1/browser/tabs          — list open Chrome page tabs (CDP)
  POST /v1/browser/navigate      — navigate a Chrome tab to a URL (CDP)
  POST /v1/browser/js            — evaluate JavaScript in a Chrome tab (CDP)
  GET  /v1/browser/screenshot    — PNG screenshot of a specific tab (CDP)
  GET  /v1/browser/page_text     — visible text of a Chrome tab (CDP)
  GET  /v1/stream/camera         — MJPEG live stream from the default camera (HTTP only)
  GET  /v1/stream/tab            — MJPEG live stream of a Chrome tab via CDP (HTTP only)
  GET  /v1/hotkey/status         — global hotkey listener state (enabled, combo, webhook URL)

Auth: every endpoint requires header ``X-Desktop-Token: <secret>``
matching ``DESKTOP_TOKEN``.  Without it the daemon answers 401.

Audit log: every accepted call appends a JSON-line to
``~/.cache/voice-assistant/desktop-audit.log`` (filename kept stable
across the desktop-server → desktop-agent rename so existing history
isn't orphaned).

Start:
    ./start.sh                  (preferred; uv handles the venv)
or:
    python3 desktop-agent.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import platform
import queue
import secrets
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx  # noqa: F401  — imported so wheel resolves; future probes may use it
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("desktop-agent")


# ─── Config ────────────────────────────────────────────────────────────

HOST_BIND = os.environ.get("DESKTOP_HOST", "127.0.0.1")
PORT = int(os.environ.get("DESKTOP_PORT", "9877"))

# Stable per-host identifier the orchestrator can use to label which
# agent it's currently talking to.  Default = hostname; override via
# env when the same orchestrator may eventually talk to multiple agents.
_AGENT_ID = os.environ.get("DESKTOP_AGENT_ID", platform.node() or "desktop-agent")

# Shared secret — match this in the orchestrator's DESKTOP_TOKEN env.
# If unset, we generate a random one and print it; the operator must
# copy it into docker-compose.yml.  Better than a hard-coded default
# that everyone forgets to change.
_DEFAULT_TOKEN_FILE = Path.home() / ".cache" / "voice-assistant" / "desktop-token"
_DEFAULT_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
if "DESKTOP_TOKEN" in os.environ:
    SHARED_TOKEN = os.environ["DESKTOP_TOKEN"]
elif _DEFAULT_TOKEN_FILE.exists():
    SHARED_TOKEN = _DEFAULT_TOKEN_FILE.read_text().strip()
else:
    SHARED_TOKEN = secrets.token_urlsafe(24)
    _DEFAULT_TOKEN_FILE.write_text(SHARED_TOKEN)
    _DEFAULT_TOKEN_FILE.chmod(0o600)
    log.warning(
        "DESKTOP_TOKEN not set — generated and saved to %s\n"
        "    copy this into docker-compose.yml DESKTOP_TOKEN env var:\n"
        "    %s",
        _DEFAULT_TOKEN_FILE, SHARED_TOKEN,
    )

# Audit log path is kept stable across the desktop-server → desktop-agent
# rename so the operator's existing tail/grep workflows still work; only
# the comment is updated.
_AUDIT_LOG = Path(
    os.environ.get(
        "DESKTOP_AUDIT_LOG",
        str(Path.home() / ".cache" / "voice-assistant" / "desktop-audit.log"),
    )
).expanduser()
_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


# ─── Chrome DevTools Protocol (CDP) state ──────────────────────────────
#
# CDP gives us a cross-platform handle on any Chrome/Chromium tab via
# the built-in debug protocol.  Chrome must be started with
#   --remote-debugging-port=<port>
# (default 9222, overridable via CHROME_DEBUG_PORT env).  The daemon
# probes at startup and every CDP_PROBE_TTL seconds; _cdp_reachable is
# the cached result that capabilities() returns to the orchestrator.
#
# Using websockets for CDP WS calls — it's already a transitive dep via
# uvicorn[standard], so no extra requirement is needed.

_CDP_PORT: int = int(os.environ.get("CHROME_DEBUG_PORT", "9222"))
_CDP_BASE: str = f"http://127.0.0.1:{_CDP_PORT}"
_CDP_PROBE_TTL: float = 30.0  # seconds between background re-probes

# Module-level bool updated by _cdp_probe().  Backends read this in
# capabilities() — safe because asyncio is single-threaded and bool
# reads are atomic under CPython's GIL.
_cdp_reachable: bool = False
_cdp_last_probe: float = 0.0          # monotonic timestamp
_CDP_POLL_TASK: asyncio.Task | None = None  # background re-probe loop


# ─── Global hotkey (#41 — Osaurus pattern) ─────────────────────────────
#
# A global keyboard shortcut lets the user trigger voice dictation from
# any application — no browser focus needed.  When the hotkey fires the
# agent POSTs to the orchestrator's /api/hotkey/ptt endpoint, which
# broadcasts a ``ptt_trigger`` event to all active browser sessions.
# The browser handles the toggle: first trigger = start PTT; second
# trigger (or 10 s timeout) = release PTT.
#
# pynput is a daemon thread — it runs in the background and doesn't
# block the asyncio event loop.  The HTTP call in the callback is
# intentionally synchronous (blocks only that thread, not the loop).

# Default combo: Ctrl+Shift+Space on all platforms.
# macOS users often prefer Cmd+Shift+Space; override via HOTKEY_COMBO.
_HOTKEY_COMBO: str = os.environ.get("HOTKEY_COMBO", "<ctrl>+<shift>+space")

# URL of the orchestrator's webhook.  In the typical single-machine
# setup (agent and orchestrator on the same host) this is localhost:8080.
_HOTKEY_WEBHOOK_URL: str = os.environ.get(
    "ORCHESTRATOR_WEBHOOK_URL", "http://localhost:8080"
).rstrip("/")

# Set to "0" / "false" / "no" to disable the global hotkey listener
# entirely (e.g. when running the agent headless on a server that has
# no keyboard access permissions).
_HOTKEY_ENABLED: bool = os.environ.get("HOTKEY_ENABLED", "1").lower() not in (
    "0", "false", "no", "off",
)

# Runtime state — set during startup.
_HOTKEY_ACTIVE: bool = False  # True once the listener started successfully
_HOTKEY_LISTENER = None       # pynput.keyboard.GlobalHotKeys instance


def _audit(event: str, **fields: Any) -> None:
    """Append one structured event line to the audit log."""
    record = {"ts": time.time(), "event": event, **fields}
    try:
        with _AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.warning("audit: failed to write event=%s", event)


# ─── Global hotkey helpers ─────────────────────────────────────────────


# Debounce window: ignore hotkey re-fires within this many seconds.  A
# global dictation key is easy to double-tap; without this, two presses
# 100 ms apart would toggle PTT on then off again.  Also bounds how often
# we spawn a webhook thread.
_HOTKEY_DEBOUNCE_S = 0.5
_hotkey_last_fired = 0.0


def _post_hotkey_webhook() -> None:
    """Blocking webhook POST — runs on its own short-lived thread."""
    url = _HOTKEY_WEBHOOK_URL + "/api/hotkey/ptt"
    try:
        import httpx as _hx
        _hx.post(url, headers={"X-Desktop-Token": SHARED_TOKEN}, timeout=2.0)
        log.info("hotkey: PTT trigger sent to %s", url)
    except Exception as exc:
        log.warning("hotkey: failed to reach orchestrator at %s: %s", url, exc)


def _hotkey_fire() -> None:
    """Called by pynput (ON its listener thread) when the hotkey fires.

    Must return immediately: pynput processes keystrokes on this same
    thread, so a blocking 2 s HTTP POST here would freeze key handling.
    We debounce, then hand the POST to a short-lived daemon thread.
    """
    global _hotkey_last_fired
    now = time.monotonic()
    if now - _hotkey_last_fired < _HOTKEY_DEBOUNCE_S:
        return  # ignore rapid re-fire / key-repeat
    _hotkey_last_fired = now
    _audit("hotkey_fired", combo=_HOTKEY_COMBO)
    threading.Thread(
        target=_post_hotkey_webhook, name="va-hotkey-webhook", daemon=True
    ).start()


def _start_hotkey_listener() -> None:
    """Attempt to start the pynput GlobalHotKeys listener.

    Runs as a daemon thread — stops automatically when the process exits.
    Sets ``_HOTKEY_ACTIVE`` on success.  On platforms where pynput is
    unavailable or the Accessibility permission is missing, logs a warning
    and leaves ``_HOTKEY_ACTIVE = False`` so capabilities() reports the
    feature as absent.
    """
    global _HOTKEY_ACTIVE, _HOTKEY_LISTENER

    if not _HOTKEY_ENABLED:
        log.info("hotkey: disabled via HOTKEY_ENABLED=0")
        return
    try:
        from pynput import keyboard as _kb

        _HOTKEY_LISTENER = _kb.GlobalHotKeys({_HOTKEY_COMBO: _hotkey_fire})
        _HOTKEY_LISTENER.start()
        _HOTKEY_ACTIVE = True
        log.info(
            "hotkey: listener started — combo=%s webhook=%s",
            _HOTKEY_COMBO, _HOTKEY_WEBHOOK_URL,
        )
    except ImportError:
        log.warning("hotkey: pynput not installed — global hotkey unavailable")
    except Exception as exc:
        log.warning(
            "hotkey: failed to start listener (missing Accessibility permission?): %s",
            exc,
        )


def _stop_hotkey_listener() -> None:
    """Gracefully stop the pynput listener (called from shutdown)."""
    global _HOTKEY_ACTIVE, _HOTKEY_LISTENER
    listener = _HOTKEY_LISTENER
    _HOTKEY_LISTENER = None
    _HOTKEY_ACTIVE = False
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass


# ─── Backend abstraction ───────────────────────────────────────────────


class Backend(ABC):
    """OS-specific host-automation backend.

    Concrete subclasses (MacOSBackend, WindowsBackend, LinuxBackend)
    implement the operations the agent advertises via /v1/capabilities.
    Methods raise NotImplementedError when a capability isn't available
    on this OS — the HTTP layer translates that into a 503 response so
    the orchestrator can render «not available on this device».

    The class is the SINGLE place per-OS quirks live; the HTTP route
    handlers are platform-blind.
    """

    name: str = "unknown"  # "macos" | "windows" | "linux"

    @abstractmethod
    def capabilities(self) -> dict:
        """Return ``{feature_name: bool}`` for what this backend supports.

        Feature names: ``screenshot``, ``applescript``, ``pyautogui``,
        ``hotkey``, ``default_apps_resolver``, ``cursor_activity``.  The
        orchestrator uses this to decide which LLM tools to register.
        """

    async def screenshot(self) -> bytes:
        raise NotImplementedError("screenshot not supported on this platform")

    async def camera_capture(self) -> bytes:
        """Capture one JPEG frame from the default camera device.

        Returns raw JPEG bytes.  Raises ``NotImplementedError`` when
        OpenCV isn't installed or no camera device is found.  Raises
        ``RuntimeError`` on a transient capture failure so the HTTP
        layer can distinguish a permanent 503 from a one-off 500.
        """
        raise NotImplementedError("camera not available on this platform")

    async def applescript(self, script: str, timeout: float) -> dict:
        raise NotImplementedError("applescript not supported on this platform")

    async def pyautogui_action(self, action: str, **kwargs: Any) -> dict:
        raise NotImplementedError("pyautogui not supported on this platform")

    async def hotkey(self, keys: list[str]) -> dict:
        raise NotImplementedError("hotkey not supported on this platform")

    async def resolve_default_app(self, category: str) -> dict | None:
        """Find the OS-default app for a task category.

        ``category`` ∈ {"mail", "browser", "calendar", "files"}.
        Returns ``{app_name, bundle_id, app_path, scriptable: bool}``
        or ``None`` when no default is known.  Implementations should
        cache for ~5 minutes — apps don't get re-defaulted often, but
        a refresh on cache miss is cheap and guarantees freshness when
        the user has changed defaults mid-session.
        """
        return None

    def cursor_activity(self) -> dict | None:
        """Return ``{x, y, idle_s}`` — None when pyautogui isn't usable.

        ``idle_s`` is seconds since the cursor last moved.  Used by the
        orchestrator as a conflict-protection signal: if the user is at
        the keyboard right now, vision-driven UI ops would fight the
        user's own mouse.
        """
        return None


# Set of pyautogui actions all backends accept.  Defined here (not on
# each subclass) so the route handler can validate before dispatch.
_ALLOWED_PYAUTOGUI_ACTIONS = {
    "click", "doubleclick", "rightclick", "move", "type", "scroll", "hotkey",
}


def _pyautogui_do(pyautogui_mod, action: str, **kwargs: Any) -> None:
    """Cross-platform implementation of one named pyautogui primitive.

    Lifted out of the subclasses so MacOSBackend/Windows/Linux share
    one source of truth — pyautogui itself is cross-platform; the only
    OS difference is whether it's importable at all (Wayland Linux,
    headless boxes).
    """
    x = kwargs.get("x")
    y = kwargs.get("y")
    text = kwargs.get("text")
    clicks = kwargs.get("clicks")
    keys = kwargs.get("keys")

    if action == "click":
        pyautogui_mod.click(x, y)
    elif action == "doubleclick":
        pyautogui_mod.doubleClick(x, y)
    elif action == "rightclick":
        pyautogui_mod.rightClick(x, y)
    elif action == "move":
        pyautogui_mod.moveTo(x, y, duration=0.05)
    elif action == "type":
        if text is None:
            raise ValueError("type requires `text`")
        pyautogui_mod.typewrite(text, interval=0.01)
    elif action == "scroll":
        pyautogui_mod.scroll(int(clicks or 0))
    elif action == "hotkey":
        if not keys:
            raise ValueError("hotkey requires `keys`")
        pyautogui_mod.hotkey(*keys)
    else:
        raise ValueError(f"unknown action: {action!r}")


def _import_cv2_safely() -> Any | None:
    """Return the cv2 (OpenCV) module or ``None`` if not installed.

    opencv-python-headless is listed in pyproject.toml; this wrapper
    keeps the agent startable on hosts where the install failed (e.g.
    stripped containers).  The camera capability is simply reported as
    False in that case.
    """
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as exc:
        log.info("cv2 unavailable — camera capture disabled: %s", exc)
        return None


def _import_pyautogui_safely() -> Any | None:
    """Return the pyautogui module or ``None`` if it can't be loaded.

    pyautogui imports raise on headless Linux (no DISPLAY), on Wayland
    in some configurations, and on stripped-down macOS where the
    Quartz framework isn't accessible.  Wrapping the import keeps the
    daemon usable for the engines that DO work on this host instead of
    crashing the whole process.
    """
    try:
        import pyautogui  # noqa: F401
        return pyautogui
    except Exception as exc:  # noqa: BLE001 — pyautogui raises bare Exception on display failure (ImportError ⊂ Exception)
        log.warning("pyautogui unavailable: %s", exc)
        return None


# ─── Cursor-activity tracker (shared across backends that have pyautogui)

# Background task tracking the last cursor position + the wall-clock at
# which it last changed.  Pyautogui's position() is a cheap syscall on
# all three platforms (~50 µs) — polling at 10 Hz is fine.  Stored as
# module state, not instance state, so the lifespan handler can spin
# the task up regardless of backend type.
_CURSOR_LAST_POS: tuple[int, int] | None = None
_CURSOR_LAST_MOVE_TS: float = 0.0
_CURSOR_TASK: asyncio.Task | None = None


async def _cursor_poll_loop(pyautogui_mod: Any) -> None:
    """10 Hz cursor poller; updates module-level last-position / last-move-ts.

    The orchestrator only needs a coarse idle-seconds number ("is the
    user active right now?"), not millisecond accuracy, so 100 ms is
    plenty.  Runs forever; cancelled on shutdown.
    """
    global _CURSOR_LAST_POS, _CURSOR_LAST_MOVE_TS
    while True:
        try:
            pos = pyautogui_mod.position()
            xy = (int(pos[0]), int(pos[1]))
            now = time.time()
            if _CURSOR_LAST_POS is None or xy != _CURSOR_LAST_POS:
                _CURSOR_LAST_POS = xy
                _CURSOR_LAST_MOVE_TS = now
        except Exception:
            # Don't kill the poll loop on a transient screen-lock /
            # permission flap — log once at debug and keep going.
            log.debug("cursor poll: transient failure", exc_info=True)
        await asyncio.sleep(0.1)


def _read_cursor_activity() -> dict | None:
    """Snapshot the latest cursor position + idle-seconds.

    Returns None when the poll task hasn't sampled yet.  Idle-seconds
    is a non-negative float; clamps tiny negatives from clock drift.
    """
    if _CURSOR_LAST_POS is None:
        return None
    idle_s = max(0.0, time.time() - _CURSOR_LAST_MOVE_TS)
    return {"x": _CURSOR_LAST_POS[0], "y": _CURSOR_LAST_POS[1], "idle_s": idle_s}


# ─── CDP helpers ───────────────────────────────────────────────────────


async def _cdp_probe() -> bool:
    """GET /json/version — update _cdp_reachable, return new value.

    Fast (2 s timeout).  Called once at startup and then every
    _CDP_PROBE_TTL seconds by the background poll task.  The
    orchestrator's 30 s health-poll cadence means a Chrome that
    comes online mid-session is detected within one poll cycle.
    """
    global _cdp_reachable, _cdp_last_probe
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{_CDP_BASE}/json/version")
        ok = r.status_code == 200
    except Exception:
        ok = False
    _cdp_reachable = ok
    _cdp_last_probe = time.monotonic()
    if ok:
        log.debug("CDP: Chrome reachable on port %d", _CDP_PORT)
    return ok


async def _cdp_ensure_reachable() -> None:
    """Re-probe if the last probe is stale; raise HTTP 503 if not reachable.

    Used at the top of every browser/* route so Chrome coming online
    after startup is detected within one _CDP_PROBE_TTL window rather
    than requiring an agent restart.
    """
    if time.monotonic() - _cdp_last_probe > _CDP_PROBE_TTL:
        await _cdp_probe()
    if not _cdp_reachable:
        raise HTTPException(
            503,
            f"Chrome not reachable on port {_CDP_PORT}. "
            f"Start Chrome with --remote-debugging-port={_CDP_PORT}",
        )


async def _cdp_list_tabs() -> list[dict]:
    """GET /json/list → filtered list of page-type tabs with WS URLs.

    Returns only tabs of type "page" (not extension popups, service
    workers, etc.) that have an active WebSocket debugger URL.  Non-page
    tabs can't be driven by Page.navigate / Runtime.evaluate.
    """
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"{_CDP_BASE}/json/list")
        r.raise_for_status()
    return [
        {
            "id": t["id"],
            "title": t.get("title") or "",
            "url": t.get("url") or "",
            "type": t.get("type") or "page",
            "ws_url": t.get("webSocketDebuggerUrl") or "",
        }
        for t in r.json()
        if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
    ]


async def _cdp_ws_call(
    ws_url: str,
    method: str,
    params: dict,
    *,
    timeout: float = 10.0,
) -> dict:
    """Open a CDP WebSocket, send one JSON-RPC command, await the response, close.

    Each call opens a fresh connection — no persistent WS state to
    manage.  Chrome tolerates concurrent per-tab connections; the ~5 ms
    overhead is negligible for voice-assistant cadence.

    CDP events (frames without an ``id``) are discarded while we wait
    for the matching response — this keeps the loop simple and correct.
    """
    import websockets as _ws

    msg_id = 1
    payload = json.dumps({"id": msg_id, "method": method, "params": params})
    deadline = time.monotonic() + timeout

    async with _ws.connect(
        ws_url, max_size=32 * 1024 * 1024, close_timeout=2.0,
    ) as ws:
        await ws.send(payload)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CDP {method!r} timed out after {timeout:.0f} s")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    err = msg["error"]
                    raise RuntimeError(
                        f"CDP error {err.get('code', '?')}: {err.get('message', '?')}"
                    )
                return msg.get("result") or {}
            # Discard CDP event frames (no id) while waiting for the response.


# Only http/https may be navigated to via CDP.  Blocking other schemes
# stops the browser-control surface from being turned into a local-file
# reader (``file:///etc/passwd`` → screenshot/page_text exfiltration),
# a JS-eval vector (``javascript:``), an internal-page opener
# (``chrome://``, ``devtools://``, ``view-source:``), or a data-URI
# injector.  The token already gates the endpoint; this bounds what a
# token holder (or a buggy orchestrator tool) can point the browser at.
_ALLOWED_NAV_SCHEMES = frozenset({"http", "https"})


def _require_web_url(url: str) -> str:
    """Validate ``url`` is a plain http(s) URL; raise HTTP 400 otherwise.

    Returns the URL unchanged on success so call sites can inline it.
    """
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_NAV_SCHEMES:
        raise HTTPException(
            400,
            f"refusing to navigate to non-web scheme {scheme or '(none)'!r}; "
            "only http/https are allowed",
        )
    return url


async def _cdp_resolve_tab(tab_id: str | None) -> dict:
    """Return the tab dict for the given id, or the first page tab if None.

    Raises HTTP 404 when the tab_id doesn't match any open tab, or
    when there are no open page tabs at all.
    """
    tabs = await _cdp_list_tabs()
    if tab_id:
        for t in tabs:
            if t["id"] == tab_id:
                return t
        raise HTTPException(404, f"no Chrome tab with id {tab_id!r}")
    if not tabs:
        raise HTTPException(404, "no open Chrome page tabs found")
    return tabs[0]


async def _cdp_poll_loop() -> None:
    """Background task: re-probe Chrome every _CDP_PROBE_TTL seconds.

    Keeps _cdp_reachable fresh so the orchestrator's next health-poll
    sees the correct browser_cdp capability flag without the agent
    requiring a restart when Chrome opens or closes.
    """
    while True:
        await asyncio.sleep(_CDP_PROBE_TTL)
        try:
            await _cdp_probe()
        except Exception:
            log.debug("CDP poll: probe failed", exc_info=True)


# ─── MacOSBackend ──────────────────────────────────────────────────────


# Bundle IDs of macOS apps we know are scriptable.  Used by
# resolve_default_app() because not every app's .sdef is in a
# predictable location — for these we don't even bother probing the
# filesystem, we just stamp scriptable=True.  Add to this set when the
# orchestrator gains direct AppleScript support for a new app (Outlook
# AppleScript dictionary, Spark, etc.).
_KNOWN_SCRIPTABLE_BUNDLE_IDS = {
    "com.apple.mail",                  # Mail.app
    "com.apple.iCal",                  # Calendar.app
    "com.microsoft.Outlook",
    "com.readdle.smartemail-Mac",      # Spark
    "it.bloop.airmail",                # Airmail
    "com.apple.Safari",
    "com.apple.finder",
    "com.google.Chrome",
    "com.brave.Browser",
    "company.thebrowser.Browser",      # Arc
}


# Mapping from logical category → URL scheme NSWorkspace will resolve
# to the user's default handler.  ``files`` uses ``file:///`` which
# Finder claims by default — this is the canonical way to discover
# "what app opens the home folder" without hard-coding ``Finder``.
_CATEGORY_TO_URL = {
    "mail":     "mailto:",
    "browser":  "https://example.com",
    "calendar": "webcal://example.com",
    "files":    "file:///",
}


class MacOSBackend(Backend):
    """macOS: AppleScript + pyautogui + NSWorkspace default-app resolution."""

    name = "macos"

    def __init__(self) -> None:
        # Engine flags — probed once at construction so the route layer
        # doesn't pay subprocess overhead per request.
        self._has_applescript = self._probe_applescript()
        self._pyautogui = _import_pyautogui_safely()
        self._has_pyautogui = self._pyautogui is not None
        self._cv2 = _import_cv2_safely()
        # Default-app resolver cache: category → (resolved_at, payload).
        # 5 min TTL matches the docstring on resolve_default_app(); the
        # operator changing their default-mail-app should see the change
        # reflected in voice within a few minutes.
        self._default_app_cache: dict[str, tuple[float, dict | None]] = {}

    @staticmethod
    def _probe_applescript() -> bool:
        try:
            r = subprocess.run(
                ["/usr/bin/osascript", "-e", "return 1"],
                capture_output=True, timeout=2,
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def capabilities(self) -> dict:
        return {
            "screenshot": self._has_pyautogui,
            "applescript": self._has_applescript,
            "pyautogui": self._has_pyautogui,
            "hotkey": self._has_pyautogui,
            "default_apps_resolver": self._has_applescript,
            "cursor_activity": self._has_pyautogui,
            "camera": self._cv2 is not None,
            # browser_cdp is cross-platform (Chrome debug protocol).
            # _cdp_reachable is updated by the background probe task.
            "browser_cdp": _cdp_reachable,
            # global_hotkey is cross-platform (pynput).
            # _HOTKEY_ACTIVE is set when the listener started cleanly.
            "global_hotkey": _HOTKEY_ACTIVE,
        }

    # ── automation ───────────────────────────────────────────────────

    async def screenshot(self) -> bytes:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available")
        pg = self._pyautogui

        def _do() -> bytes:
            img = pg.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        return await asyncio.to_thread(_do)

    async def camera_capture(self) -> bytes:
        if self._cv2 is None:
            raise NotImplementedError("opencv not installed")
        cv2 = self._cv2

        def _do() -> bytes:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("no camera device accessible")
            try:
                # Warm up auto-exposure: discard the first few frames
                # (camera sensors start dark on cold open).
                for _ in range(3):
                    cap.read()
                ret, frame = cap.read()
            finally:
                cap.release()
            if not ret or frame is None:
                raise RuntimeError("camera read returned empty frame")
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise RuntimeError("JPEG encoding of camera frame failed")
            return bytes(buf)

        return await asyncio.to_thread(_do)

    async def applescript(self, script: str, timeout: float) -> dict:
        if not self._has_applescript:
            raise NotImplementedError("osascript not available")
        try:
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise HTTPException(504, "AppleScript timeout")
        except FileNotFoundError:
            raise HTTPException(503, "osascript not found")
        return {
            "exit": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }

    async def pyautogui_action(self, action: str, **kwargs: Any) -> dict:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available")
        delay = float(kwargs.pop("delay", 0.0) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        pg = self._pyautogui

        def _do() -> None:
            _pyautogui_do(pg, action, **kwargs)

        await asyncio.to_thread(_do)
        return {"ok": True}

    async def hotkey(self, keys: list[str]) -> dict:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available")
        pg = self._pyautogui

        def _do() -> None:
            pg.hotkey(*keys)

        await asyncio.to_thread(_do)
        return {"ok": True}

    # ── default-app resolution ────────────────────────────────────────

    async def resolve_default_app(self, category: str) -> dict | None:
        """Resolve via NSWorkspace.URLForApplicationToOpenURL.

        Implementation notes:
          • We deliberately shell out to ``osascript -l JavaScript`` (JXA)
            instead of taking a PyObjC dependency — that keeps
            desktop-agent's uv venv small (~5 MB instead of ~40 MB)
            and avoids the PyObjC import-startup tax.
          • Scriptable detection: bundle-id allowlist first (fast,
            deterministic for the apps we care about); .sdef file
            probe as a fallback.  Either signal alone is sufficient.
          • 5 min cache.  Apps don't get re-defaulted often, and a
            stale value just means the orchestrator might pick the
            wrong app — recoverable, vs hammering osascript on every
            voice turn.
        """
        if category not in _CATEGORY_TO_URL:
            return None
        if not self._has_applescript:
            return None
        # Cache hit?
        cached = self._default_app_cache.get(category)
        if cached is not None:
            ts, payload = cached
            if time.time() - ts < 300.0:
                return payload

        url = _CATEGORY_TO_URL[category]
        # JXA one-liner: get the .app path NSWorkspace would launch for `url`.
        # Empty string when nothing handles the scheme.
        jxa = (
            'ObjC.import("AppKit");'
            'const ws = $.NSWorkspace.sharedWorkspace;'
            f'const url = $.NSURL.URLWithString("{url}");'
            'const appUrl = ws.URLForApplicationToOpenURL(url);'
            'appUrl ? appUrl.path.js : ""'
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/osascript", "-l", "JavaScript", "-e", jxa,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except (FileNotFoundError, asyncio.TimeoutError):
            self._default_app_cache[category] = (time.time(), None)
            return None
        app_path = stdout.decode("utf-8", errors="replace").strip()
        if not app_path:
            self._default_app_cache[category] = (time.time(), None)
            return None
        # ``/Applications/Mail.app`` → ``Mail``
        app_name = Path(app_path).stem
        bundle_id = await self._bundle_id_for_app(app_name)
        scriptable = self._is_scriptable(app_path, bundle_id)
        payload = {
            "app_name": app_name,
            "app_path": app_path,
            "bundle_id": bundle_id,
            "scriptable": scriptable,
        }
        self._default_app_cache[category] = (time.time(), payload)
        return payload

    async def _bundle_id_for_app(self, app_name: str) -> str | None:
        """``osascript -e 'id of application "X"'`` → bundle identifier.

        Empty / non-zero exit → None.  We swallow errors quietly because
        a missing bundle-id is OK (callers fall back to the scriptable
        allowlist via .sdef detection).
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/osascript", "-e", f'id of application "{app_name}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        except (FileNotFoundError, asyncio.TimeoutError):
            return None
        if proc.returncode != 0:
            return None
        bid = stdout.decode("utf-8", errors="replace").strip()
        return bid or None

    @staticmethod
    def _is_scriptable(app_path: str, bundle_id: str | None) -> bool:
        """Best-effort scriptable detection.

        Two independent signals — either is sufficient:
          1. Bundle-id is in the known-scriptable allowlist (fast, exact).
          2. ``Contents/Resources/*.sdef`` exists in the app bundle.
        """
        if bundle_id and bundle_id in _KNOWN_SCRIPTABLE_BUNDLE_IDS:
            return True
        try:
            res = Path(app_path) / "Contents" / "Resources"
            if not res.is_dir():
                return False
            for entry in res.iterdir():
                if entry.suffix == ".sdef":
                    return True
        except OSError:
            return False
        return False

    # ── cursor activity ───────────────────────────────────────────────

    def cursor_activity(self) -> dict | None:
        if not self._has_pyautogui:
            return None
        return _read_cursor_activity()


# ─── WindowsBackend (skeleton) ─────────────────────────────────────────


class WindowsBackend(Backend):
    """Windows: cross-platform pyautogui only for now.

    Default-app resolution via ``winreg`` is sketched (mailto / http
    UserChoice lookup) but kept as a stub returning None — the user
    is on macOS for this sprint and the AppleScript path doesn't apply
    here regardless.  Filling this in is a half-day project when the
    Windows host comes online.
    """

    name = "windows"

    def __init__(self) -> None:
        self._pyautogui = _import_pyautogui_safely()
        self._has_pyautogui = self._pyautogui is not None
        self._cv2 = _import_cv2_safely()
        try:
            import pywinauto  # noqa: F401
            self._has_pywinauto = True
        except ImportError:
            self._has_pywinauto = False

    def capabilities(self) -> dict:
        return {
            "screenshot": self._has_pyautogui,
            "applescript": False,
            "pyautogui": self._has_pyautogui,
            "hotkey": self._has_pyautogui,
            # TODO: winreg UserChoice → default app per category.  Stub for now.
            "default_apps_resolver": False,
            "cursor_activity": self._has_pyautogui,
            "camera": self._cv2 is not None,
            "browser_cdp": _cdp_reachable,
            # global_hotkey is cross-platform (pynput runs on all OSes).
            "global_hotkey": _HOTKEY_ACTIVE,
        }

    async def screenshot(self) -> bytes:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available")
        pg = self._pyautogui

        def _do() -> bytes:
            img = pg.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        return await asyncio.to_thread(_do)

    async def camera_capture(self) -> bytes:
        if self._cv2 is None:
            raise NotImplementedError("opencv not installed")
        cv2 = self._cv2

        def _do() -> bytes:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("no camera device accessible")
            try:
                for _ in range(3):
                    cap.read()
                ret, frame = cap.read()
            finally:
                cap.release()
            if not ret or frame is None:
                raise RuntimeError("camera read returned empty frame")
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise RuntimeError("JPEG encoding of camera frame failed")
            return bytes(buf)

        return await asyncio.to_thread(_do)

    async def pyautogui_action(self, action: str, **kwargs: Any) -> dict:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available")
        delay = float(kwargs.pop("delay", 0.0) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        pg = self._pyautogui

        def _do() -> None:
            _pyautogui_do(pg, action, **kwargs)

        await asyncio.to_thread(_do)
        return {"ok": True}

    async def hotkey(self, keys: list[str]) -> dict:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available")
        pg = self._pyautogui

        def _do() -> None:
            pg.hotkey(*keys)

        await asyncio.to_thread(_do)
        return {"ok": True}

    async def resolve_default_app(self, category: str) -> dict | None:
        # Stub — winreg lookup of HKCU\...\UrlAssociations\<scheme>\UserChoice
        # would resolve the ProgId, then resolve the .exe path.  Not
        # this sprint.
        return None

    def cursor_activity(self) -> dict | None:
        if not self._has_pyautogui:
            return None
        return _read_cursor_activity()


# ─── LinuxBackend (skeleton) ───────────────────────────────────────────


class LinuxBackend(Backend):
    """Linux: pyautogui + (optional) xdotool.

    Wayland is intentionally not special-cased — pyautogui works on
    X11, half-works on Wayland depending on compositor, and is the
    user's problem to enable.  ``resolve_default_app`` could shell out
    to ``xdg-mime query default`` per scheme; kept as stub for now.
    """

    name = "linux"

    def __init__(self) -> None:
        self._pyautogui = _import_pyautogui_safely()
        self._has_pyautogui = self._pyautogui is not None
        self._cv2 = _import_cv2_safely()
        try:
            r = subprocess.run(
                ["xdotool", "--version"], capture_output=True, timeout=2,
            )
            self._has_xdotool = r.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            self._has_xdotool = False

    def capabilities(self) -> dict:
        return {
            "screenshot": self._has_pyautogui,
            "applescript": False,
            "pyautogui": self._has_pyautogui,
            "hotkey": self._has_pyautogui or self._has_xdotool,
            # TODO: xdg-mime query default x-scheme-handler/mailto → desktop file.
            "default_apps_resolver": False,
            "cursor_activity": self._has_pyautogui,
            "camera": self._cv2 is not None,
            "browser_cdp": _cdp_reachable,
            # global_hotkey is cross-platform (pynput runs on all OSes).
            "global_hotkey": _HOTKEY_ACTIVE,
        }

    async def screenshot(self) -> bytes:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available (X11/Wayland)")
        pg = self._pyautogui

        def _do() -> bytes:
            img = pg.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        return await asyncio.to_thread(_do)

    async def camera_capture(self) -> bytes:
        if self._cv2 is None:
            raise NotImplementedError("opencv not installed")
        cv2 = self._cv2

        def _do() -> bytes:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("no camera device accessible")
            try:
                for _ in range(3):
                    cap.read()
                ret, frame = cap.read()
            finally:
                cap.release()
            if not ret or frame is None:
                raise RuntimeError("camera read returned empty frame")
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise RuntimeError("JPEG encoding of camera frame failed")
            return bytes(buf)

        return await asyncio.to_thread(_do)

    async def pyautogui_action(self, action: str, **kwargs: Any) -> dict:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available (X11/Wayland)")
        delay = float(kwargs.pop("delay", 0.0) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        pg = self._pyautogui

        def _do() -> None:
            _pyautogui_do(pg, action, **kwargs)

        await asyncio.to_thread(_do)
        return {"ok": True}

    async def hotkey(self, keys: list[str]) -> dict:
        if not self._has_pyautogui:
            raise NotImplementedError("pyautogui not available (X11/Wayland)")
        pg = self._pyautogui

        def _do() -> None:
            pg.hotkey(*keys)

        await asyncio.to_thread(_do)
        return {"ok": True}

    async def resolve_default_app(self, category: str) -> dict | None:
        # Stub — xdg-mime query would do it; not this sprint.
        return None

    def cursor_activity(self) -> dict | None:
        if not self._has_pyautogui:
            return None
        return _read_cursor_activity()


# ─── Backend selection ─────────────────────────────────────────────────


def _make_backend() -> Backend:
    s = platform.system()
    if s == "Darwin":
        return MacOSBackend()
    if s == "Windows":
        return WindowsBackend()
    if s == "Linux":
        return LinuxBackend()
    raise RuntimeError(f"unsupported OS: {s}")


_BACKEND: Backend = _make_backend()


# ─── Schemas ───────────────────────────────────────────────────────────


class ApplescriptRequest(BaseModel):
    script: str = Field(..., description="AppleScript source to execute.")
    timeout: float = 30.0


class PyautoguiRequest(BaseModel):
    """Structured cross-platform automation.

    ``action`` enumerates the supported primitives; arg shape varies.
    Only the named actions are accepted (no string eval) — every path
    is auditable.
    """
    action: str  # click | doubleclick | rightclick | move | type | scroll | hotkey
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: list[str] | None = None
    clicks: int | None = None
    # Optional delay before executing (seconds), e.g. to let a menu
    # appear after a previous click.
    delay: float = 0.0


class KeyRequest(BaseModel):
    keys: list[str]  # e.g. ["cmd", "space"] or ["ctrl", "shift", "t"]


class AuditRequest(BaseModel):
    event: str
    payload: dict = Field(default_factory=dict)


class BrowserNavigateRequest(BaseModel):
    """Navigate an existing Chrome tab to a URL, or open a new tab.

    ``tab_id`` — omit to open a new tab.  The orchestrator gets the
    resolved ``tab_id`` back so it can anchor follow-up calls.
    """

    url: str
    tab_id: str | None = None


class BrowserJsRequest(BaseModel):
    """Evaluate JavaScript in a Chrome tab.

    ``code`` — the JS expression to run (Runtime.evaluate).
    ``tab_id`` — omit to target the first page tab.
    ``timeout`` — CDP WS call timeout in seconds.
    ``return_by_value`` — when True the result is JSON-serialised; when
        False the result is a remote object reference (advanced use).
    """

    code: str
    tab_id: str | None = None
    timeout: float = 10.0
    return_by_value: bool = True


# ─── App ───────────────────────────────────────────────────────────────


_VERSION = "1.5.0"
app = FastAPI(title="desktop-agent", version=_VERSION)


@app.on_event("startup")
async def startup() -> None:
    """Log readiness; spin up cursor-activity and CDP background poll tasks.

    CDP probe runs first so capabilities() in the log line already shows
    the correct browser_cdp value; subsequent probes run every
    _CDP_PROBE_TTL seconds so Chrome coming online after startup is
    detected within one poll cycle.
    """
    global _CURSOR_TASK, _CDP_POLL_TASK

    # Probe Chrome before reading capabilities so the log shows the
    # real browser_cdp value rather than the uninitialised False.
    await _cdp_probe()

    caps = _BACKEND.capabilities()
    log.info(
        "desktop-agent ready: id=%s platform=%s version=%s cdp_port=%d caps=%s",
        _AGENT_ID, _BACKEND.name, _VERSION, _CDP_PORT,
        ",".join(k for k, v in caps.items() if v) or "<none>",
    )

    # Cursor-activity poll — only when pyautogui is available.
    pg = getattr(_BACKEND, "_pyautogui", None)
    if pg is not None and (_CURSOR_TASK is None or _CURSOR_TASK.done()):
        _CURSOR_TASK = asyncio.create_task(_cursor_poll_loop(pg))

    # CDP re-probe loop — keeps browser_cdp capability fresh.
    if _CDP_POLL_TASK is None or _CDP_POLL_TASK.done():
        _CDP_POLL_TASK = asyncio.create_task(_cdp_poll_loop())

    # Global hotkey listener — starts a daemon thread (not asyncio).
    # Must run after the event loop is up so pynput's internal asyncio
    # hooks (macOS) initialise against the running loop.
    _start_hotkey_listener()


@app.on_event("shutdown")
async def shutdown() -> None:
    global _CURSOR_TASK, _CDP_POLL_TASK

    async def _cancel(task: asyncio.Task | None) -> None:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    await _cancel(_CURSOR_TASK)
    _CURSOR_TASK = None
    await _cancel(_CDP_POLL_TASK)
    _CDP_POLL_TASK = None
    # Stop the pynput listener thread.
    _stop_hotkey_listener()


def _check_auth(token: str | None) -> None:
    if not token or not secrets.compare_digest(token, SHARED_TOKEN):
        raise HTTPException(401, "missing or invalid X-Desktop-Token")


def _capability_or_503(feature: str) -> None:
    """Translate the backend's capability flag into a 503 with a clear
    detail string so the orchestrator can render «not available on
    this device» without parsing exception text."""
    caps = _BACKEND.capabilities()
    if not caps.get(feature):
        raise HTTPException(
            503,
            f"{feature} not supported on this {_BACKEND.name}",
        )


# ─── Health / capabilities / platform ──────────────────────────────────


@app.get("/v1/health")
async def health() -> dict:
    """Liveness + the legacy ``engines`` map (kept for orchestrator
    versions that haven't switched to /v1/capabilities yet).

    New orchestrator code should prefer /v1/capabilities; this endpoint
    answers without auth so a friend in the same shell can `curl` and
    see the daemon is up.

    Deliberately omits ``agent_id`` (hostname) and ``audit_log`` (an
    absolute filesystem path that leaks the home dir / username) — those
    are recon fodder on a remote (0.0.0.0) deployment.  The full detail
    is available at the auth-gated /v1/capabilities.  The boolean
    ``engines`` map is retained for pre-1.1 orchestrator compatibility.
    """
    caps = _BACKEND.capabilities()
    return {
        "ok": True,
        "platform": _BACKEND.name,
        "version": _VERSION,
        # Legacy keys — orchestrator's pre-1.1 code reads these.
        "engines": {
            "applescript": caps.get("applescript", False),
            "pyautogui": caps.get("pyautogui", False),
            "xdotool": getattr(_BACKEND, "_has_xdotool", False),
            "pywinauto": getattr(_BACKEND, "_has_pywinauto", False),
        },
    }


@app.get("/v1/capabilities")
async def capabilities(x_desktop_token: str | None = Header(default=None)) -> dict:
    """Capabilities — auth-gated, intended for orchestrator wiring.

    Returns a snapshot the orchestrator can cache (60 s on the client
    side); changes between calls only when the operator restarts the
    daemon to pick up new permissions.  When the platform supports
    default-app resolution we pre-warm the cache for the four known
    categories so the orchestrator can plan without an extra
    round-trip per voice turn.
    """
    _check_auth(x_desktop_token)
    caps = _BACKEND.capabilities()
    out: dict = {
        "agent_id": _AGENT_ID,
        "platform": _BACKEND.name,
        "capabilities": caps,
        "version": _VERSION,
    }
    if caps.get("default_apps_resolver"):
        # Best-effort pre-resolve — failures here just mean the
        # orchestrator will pay a round-trip per category, not a hard
        # error.  Keep individual timeouts low so a broken category
        # can't stall capability reporting.
        resolved: dict[str, dict | None] = {}
        for cat in ("mail", "browser", "calendar", "files"):
            try:
                resolved[cat] = await _BACKEND.resolve_default_app(cat)
            except Exception:
                log.debug("capabilities: default_app %s probe failed", cat, exc_info=True)
                resolved[cat] = None
        out["default_apps"] = resolved
    return out


@app.get("/v1/platform")
async def platform_info(x_desktop_token: str | None = Header(default=None)) -> dict:
    _check_auth(x_desktop_token)
    caps = _BACKEND.capabilities()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version.split()[0],
        "preferred_engine": (
            "applescript" if caps.get("applescript")
            else "pywinauto" if getattr(_BACKEND, "_has_pywinauto", False)
            else "xdotool" if getattr(_BACKEND, "_has_xdotool", False)
            else "pyautogui" if caps.get("pyautogui")
            else None
        ),
    }


# ─── AppleScript ───────────────────────────────────────────────────────


@app.post("/v1/applescript")
async def run_applescript(
    req: ApplescriptRequest,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Run AppleScript via /usr/bin/osascript and return stdout / exit.

    The orchestrator's `desktop` tool keeps the per-app allowlist; we
    don't re-check it here — the boundary is "is this a trusted client"
    (token), not "is this script wise" (that's a per-tenant policy).
    """
    _check_auth(x_desktop_token)
    _capability_or_503("applescript")
    t0 = time.monotonic()
    try:
        result = await _BACKEND.applescript(req.script, req.timeout)
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc))
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    _audit(
        "applescript",
        script=req.script[:400],
        exit=result.get("exit"),
        elapsed_ms=elapsed_ms,
    )
    return {**result, "elapsed_ms": elapsed_ms}


# ─── pyautogui ─────────────────────────────────────────────────────────


@app.post("/v1/pyautogui")
async def run_pyautogui(
    req: PyautoguiRequest,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Structured cross-platform desktop input via pyautogui.

    Only the actions in ``_ALLOWED_PYAUTOGUI_ACTIONS`` are honoured;
    anything else returns 400.  No ``eval``-able code path, just a
    switch over named primitives (implemented in :func:`_pyautogui_do`).
    """
    _check_auth(x_desktop_token)
    _capability_or_503("pyautogui")
    if req.action not in _ALLOWED_PYAUTOGUI_ACTIONS:
        raise HTTPException(400, f"unknown action: {req.action!r}")

    t0 = time.monotonic()
    try:
        result = await _BACKEND.pyautogui_action(
            req.action,
            x=req.x, y=req.y, text=req.text,
            keys=req.keys, clicks=req.clicks, delay=req.delay,
        )
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    _audit(
        "pyautogui",
        action=req.action,
        x=req.x, y=req.y,
        text=(req.text[:40] if req.text else None),
        keys=req.keys,
        elapsed_ms=elapsed_ms,
    )
    return {**result, "elapsed_ms": elapsed_ms}


# ─── Key ────────────────────────────────────────────────────────────────


@app.post("/v1/key")
async def run_key(
    req: KeyRequest,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Press a keyboard shortcut.

    Kept as its own endpoint because keyboard shortcuts are by far the
    most common automation primitive — having a dedicated path keeps
    the audit log readable ("key cmd+space" instead of "pyautogui
    action=hotkey keys=...").
    """
    _check_auth(x_desktop_token)
    _capability_or_503("hotkey")
    t0 = time.monotonic()
    try:
        result = await _BACKEND.hotkey(req.keys)
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc))
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    _audit("key", keys=req.keys, elapsed_ms=elapsed_ms)
    return {**result, "elapsed_ms": elapsed_ms}


# ─── Screenshot ─────────────────────────────────────────────────────────


@app.get("/v1/screenshot")
async def take_screenshot(
    x_desktop_token: str | None = Header(default=None),
) -> Response:
    """Full-screen PNG capture, consumed by the orchestrator's
    ``look_at_screen`` tool and the image-attached pipeline branch."""
    _check_auth(x_desktop_token)
    _capability_or_503("screenshot")
    try:
        png = await _BACKEND.screenshot()
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc))
    _audit("screenshot", bytes=len(png))
    return Response(content=png, media_type="image/png")


@app.get("/v1/camera")
async def take_camera_frame(
    x_desktop_token: str | None = Header(default=None),
) -> Response:
    """Capture a single JPEG frame from the default camera device.

    Used by the orchestrator's ``look_at_camera`` tool for utterances
    like "посмотри камерой" / "look through the camera".  Returns the
    raw JPEG so the orchestrator can pass it directly to the vision LLM.

    503 — opencv not installed or no camera permission/device.
    500 — transient capture failure (camera busy, driver error).
    """
    _check_auth(x_desktop_token)
    _capability_or_503("camera")
    t0 = time.monotonic()
    try:
        jpg = await _BACKEND.camera_capture()
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    _audit("camera_capture", bytes=len(jpg), elapsed_ms=elapsed_ms)
    return Response(content=jpg, media_type="image/jpeg")


# ─── Browser (Chrome DevTools Protocol) ─────────────────────────────────
#
# All routes require Chrome running with --remote-debugging-port=<CDP_PORT>
# (default 9222 / CHROME_DEBUG_PORT env).  Each call opens a fresh CDP
# WebSocket connection for simplicity — the sub-millisecond overhead is
# fine at voice-assistant cadence.
#
# Route architecture mirrors the existing capability gates: every route
# calls _cdp_ensure_reachable() which does an inline re-probe if the
# last probe is stale (_CDP_PROBE_TTL), then raises 503 if Chrome isn't
# up.  This means Chrome coming online after startup is detected within
# one _CDP_PROBE_TTL window without a daemon restart.


@app.get("/v1/browser/tabs")
async def browser_list_tabs(
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """List open Chrome page tabs.

    Returns only "page"-type tabs that have an active WS debugger URL —
    extension popups and service workers are excluded.  Use tab ``id``
    from the response to anchor navigate / js / screenshot calls to a
    specific tab.
    """
    _check_auth(x_desktop_token)
    await _cdp_ensure_reachable()
    tabs = await _cdp_list_tabs()
    _audit("browser_tabs", count=len(tabs))
    return {"tabs": tabs}


@app.post("/v1/browser/navigate")
async def browser_navigate(
    req: BrowserNavigateRequest,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Navigate a Chrome tab to a URL.

    ``tab_id`` present → navigate that tab in-place via Page.navigate.
    ``tab_id`` absent  → open a new tab via Chrome's /json/new endpoint.
    Returns the resolved ``tab_id`` so the caller can chain follow-up
    calls without re-listing tabs.
    """
    _check_auth(x_desktop_token)
    # Only http/https — never file://, chrome://, javascript:, data:, etc.
    _require_web_url(req.url)
    await _cdp_ensure_reachable()

    if req.tab_id:
        tab = await _cdp_resolve_tab(req.tab_id)
        result = await _cdp_ws_call(
            tab["ws_url"], "Page.navigate", {"url": req.url}, timeout=15.0,
        )
        _audit("browser_navigate", url=req.url, tab_id=tab["id"])
        return {
            "ok": True,
            "tab_id": tab["id"],
            "url": result.get("url") or req.url,
            "frame_id": result.get("frameId"),
        }

    # New tab: Chrome's HTTP endpoint creates the tab and starts loading.
    # The WS debugger URL isn't immediately available — we return the
    # tab_id immediately; the tab appears in /browser/tabs within ~500 ms.
    # Percent-encode the URL so it can't inject extra query params into
    # Chrome's /json/new endpoint (e.g. an embedded '&').
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"{_CDP_BASE}/json/new?{quote(req.url, safe='')}")
        r.raise_for_status()
        new_info = r.json()
    tab_id = new_info.get("id", "")
    _audit("browser_navigate", url=req.url, tab_id=tab_id, action="new_tab")
    return {"ok": True, "tab_id": tab_id, "url": req.url}


@app.post("/v1/browser/js")
async def browser_execute_js(
    req: BrowserJsRequest,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Evaluate JavaScript in a Chrome tab and return the result.

    Uses CDP Runtime.evaluate — the expression runs in the tab's main
    frame.  The daemon does NOT restrict what code runs; that policy
    lives in the orchestrator's ``run_browser_js`` tool (risk=high_write).
    Trusted-client gateway: the DESKTOP_TOKEN is the only gate here.
    """
    _check_auth(x_desktop_token)
    await _cdp_ensure_reachable()
    tab = await _cdp_resolve_tab(req.tab_id)
    result = await _cdp_ws_call(
        tab["ws_url"],
        "Runtime.evaluate",
        {"expression": req.code, "returnByValue": req.return_by_value},
        timeout=req.timeout,
    )
    _audit("browser_js", code=req.code[:200], tab_id=tab["id"])
    rv = result.get("result") or {}
    return {
        "ok": True,
        "tab_id": tab["id"],
        "type": rv.get("type"),
        "value": rv.get("value"),
        "description": rv.get("description"),
    }


@app.get("/v1/browser/screenshot")
async def browser_tab_screenshot(
    tab_id: str | None = None,
    x_desktop_token: str | None = Header(default=None),
) -> Response:
    """PNG screenshot of a specific Chrome tab (not the full screen).

    Uses CDP Page.captureScreenshot so the captured area is exactly the
    tab's viewport — no chrome / taskbar / other windows included.
    Useful for feeding a specific page to the vision LLM without
    capturing the whole desktop.
    """
    _check_auth(x_desktop_token)
    await _cdp_ensure_reachable()
    tab = await _cdp_resolve_tab(tab_id)
    result = await _cdp_ws_call(
        tab["ws_url"],
        "Page.captureScreenshot",
        {"format": "png", "fromSurface": True},
        timeout=15.0,
    )
    png_b64 = result.get("data") or ""
    if not png_b64:
        raise HTTPException(500, "CDP returned empty screenshot data")
    png = base64.b64decode(png_b64)
    _audit("browser_screenshot", tab_id=tab["id"], bytes=len(png))
    return Response(content=png, media_type="image/png")


@app.get("/v1/browser/page_text")
async def browser_page_text(
    tab_id: str | None = None,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Extract the visible text of a Chrome tab.

    Runs ``document.body.innerText`` (falls back to ``textContent``)
    in the tab via Runtime.evaluate.  The orchestrator's
    ``read_browser_tab`` tool passes this to the LLM to answer
    user questions about the current page.
    """
    _check_auth(x_desktop_token)
    await _cdp_ensure_reachable()
    tab = await _cdp_resolve_tab(tab_id)
    result = await _cdp_ws_call(
        tab["ws_url"],
        "Runtime.evaluate",
        {
            "expression": (
                "(function(){"
                "  var b = document.body;"
                "  return b ? (b.innerText || b.textContent || '') : '';"
                "})()"
            ),
            "returnByValue": True,
        },
        timeout=10.0,
    )
    rv = result.get("result") or {}
    text = rv.get("value") or ""
    _audit("browser_page_text", tab_id=tab["id"], chars=len(text))
    return {
        "text": text,
        "url": tab["url"],
        "title": tab["title"],
        "tab_id": tab["id"],
    }


# ─── Global hotkey status ──────────────────────────────────────────────


@app.get("/v1/hotkey/status")
async def hotkey_status(
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Return the current state of the global hotkey listener.

    ``enabled``       — True when the listener started successfully.
    ``combo``         — The configured hotkey combination string.
    ``webhook_url``   — Orchestrator webhook the hotkey POSTs to.

    Useful for operator diagnostics: if ``enabled`` is False, check
    that pynput is installed (``uv sync`` in the desktop-agent dir)
    and that the terminal has Accessibility permission on macOS.
    """
    _check_auth(x_desktop_token)
    return {
        "enabled": _HOTKEY_ACTIVE,
        "combo": _HOTKEY_COMBO,
        "webhook_url": _HOTKEY_WEBHOOK_URL,
    }


# ─── MJPEG live streaming ──────────────────────────────────────────────
#
# Each endpoint yields a ``multipart/x-mixed-replace`` stream of JPEG
# frames, which browsers render directly in an ``<img>`` tag — no JS
# needed on the client side.
#
# HTTP-only (not implemented in ``_reverse_dispatch``).  The orchestrator
# proxies these through ``/api/stream/{source}`` so family devices only
# need the ``va_session`` cookie, not the DESKTOP_TOKEN.

_STREAM_BOUNDARY = "va_frame"

# Maximum FPS the caller may request for each source.
# Camera goes up to 30 fps; CDP tab screenshots are heavier and capped
# at 15 fps to keep the event loop free.
_CAMERA_MAX_FPS: float = 30.0
_TAB_MAX_FPS: float = 15.0


async def _mjpeg_frame(jpg: bytes) -> bytes:
    """Wrap a JPEG image in a MJPEG multipart frame."""
    header = (
        f"--{_STREAM_BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpg)}\r\n"
        f"\r\n"
    ).encode()
    return header + jpg + b"\r\n"


def _camera_grab_loop(
    cv2: Any,
    fps: float,
    stop: "threading.Event",
    frames: "queue.Queue",
) -> None:
    """Worker thread: own ONE ``VideoCapture`` for the stream's lifetime.

    Opening/closing the device per frame (the old behaviour) re-triggered
    the camera privacy LED + AVFoundation session negotiation 15–30×/sec,
    which both thrashed the device and made the target fps unreachable.
    Here the capture is opened once and released in ``finally`` when the
    consumer signals ``stop`` (client disconnect).  Latest-frame-wins: a
    bounded queue means a slow consumer drops stale frames rather than
    lagging.  A ``None`` sentinel ends the consumer cleanly.
    """
    interval = 1.0 / fps
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            frames.put(None)
            return
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            enc_ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if enc_ok:
                # Drop the previous frame if the consumer hasn't taken it,
                # so we always hold only the freshest frame.
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frames.put_nowait(bytes(buf))
                except queue.Full:
                    pass
            stop.wait(interval)
    finally:
        cap.release()
        frames.put(None)  # sentinel: tell the consumer the stream ended


async def _mjpeg_gen_camera(fps: float):
    """Async generator — yields MJPEG parts from the host camera at ``fps``.

    A worker thread owns the ``VideoCapture`` (opened once) and pushes
    JPEG frames through a queue; the blocking grab/encode never touches
    the event loop.  On client disconnect the generator is closed, the
    ``finally`` sets ``stop``, and the worker releases the device exactly
    once.
    """
    cv2 = getattr(_BACKEND, "_cv2", None)
    if cv2 is None:
        log.warning("stream/camera: opencv not available")
        return
    stop = threading.Event()
    frames: "queue.Queue" = queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=_camera_grab_loop,
        args=(cv2, fps, stop, frames),
        name="va-camera-stream",
        daemon=True,
    )
    worker.start()
    try:
        while True:
            jpg = await asyncio.to_thread(frames.get)
            if jpg is None:  # camera open failed or stream ended
                return
            yield await _mjpeg_frame(jpg)
    finally:
        # Client disconnected (GeneratorExit) or stream ended: stop the
        # worker and let it release the camera device.
        stop.set()
        try:
            frames.get_nowait()  # unblock the worker if the queue is full
        except queue.Empty:
            pass
        await asyncio.to_thread(worker.join, 2.0)


async def _mjpeg_gen_tab(fps: float, tab_id: str | None):
    """Async generator — yields MJPEG parts from a CDP tab at ``fps``.

    Opens a fresh CDP WebSocket per frame (stateless, sub-ms overhead at
    this frame rate).  Stops if the tab disappears or CDP becomes
    unreachable.
    """
    interval = 1.0 / fps
    try:
        while True:
            t0 = asyncio.get_event_loop().time()
            try:
                tab = await _cdp_resolve_tab(tab_id)
                result = await _cdp_ws_call(
                    tab["ws_url"],
                    "Page.captureScreenshot",
                    {"format": "jpeg", "quality": 70, "fromSurface": True},
                )
            except Exception as exc:
                # Tab closed / CDP unreachable — end the stream cleanly.
                log.warning("stream/tab: capture failed: %s", exc)
                return
            data_b64 = result.get("data") or ""
            if data_b64:
                jpg = base64.b64decode(data_b64)
                yield await _mjpeg_frame(jpg)
            elapsed = asyncio.get_event_loop().time() - t0
            delay = interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
    finally:
        # GeneratorExit on client disconnect lands here — nothing to
        # release (each frame uses a short-lived CDP WS), but log the end
        # for symmetry with the camera stream and future-proofing.
        log.debug("stream/tab: generator closed")


@app.get("/v1/stream/camera")
async def stream_camera(
    fps: float = 15.0,
    x_desktop_token: str | None = Header(default=None),
) -> StreamingResponse:
    """MJPEG live stream from the default camera device.

    ``fps`` — target frame rate (1–30; default 15).  Capped to
    ``_CAMERA_MAX_FPS``.  The actual rate may be lower depending on
    camera sensor speed and host load.

    Returns a ``multipart/x-mixed-replace`` response — display directly
    in an ``<img>`` tag or proxy through the orchestrator's
    ``/api/stream/camera`` endpoint.

    HTTP-only — not available in reverse-WSS mode.
    """
    _check_auth(x_desktop_token)
    _capability_or_503("camera")
    fps = max(1.0, min(float(fps), _CAMERA_MAX_FPS))
    _audit("stream_camera_start", fps=fps)
    return StreamingResponse(
        _mjpeg_gen_camera(fps),
        media_type=f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}",
    )


@app.get("/v1/stream/tab")
async def stream_tab(
    tab_id: str | None = None,
    fps: float = 5.0,
    x_desktop_token: str | None = Header(default=None),
) -> StreamingResponse:
    """MJPEG live stream of a Chrome tab via CDP.

    ``tab_id`` — CDP page id.  Omit to use the first open page tab.
    ``fps`` — target frame rate (1–15; default 5).  Capped to
    ``_TAB_MAX_FPS``.

    Requires Chrome running with ``--remote-debugging-port``.
    HTTP-only — not available in reverse-WSS mode.
    """
    _check_auth(x_desktop_token)
    await _cdp_ensure_reachable()
    fps = max(1.0, min(float(fps), _TAB_MAX_FPS))
    # Resolve the tab now so we can 404 early if tab_id is bogus.
    tab = await _cdp_resolve_tab(tab_id)
    _audit("stream_tab_start", tab_id=tab["id"], fps=fps)
    return StreamingResponse(
        _mjpeg_gen_tab(fps, tab["id"]),
        media_type=f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}",
    )


# ─── Free-form audit endpoint ──────────────────────────────────────────


@app.post("/v1/audit")
async def submit_audit(
    req: AuditRequest,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Let the orchestrator log a non-execution event (e.g. a tool call
    that was denied by allowlist) so the audit log captures intent, not
    just executions."""
    _check_auth(x_desktop_token)
    _audit(req.event, **req.payload)
    return {"ok": True}


# ─── Default-app resolution + cursor activity ──────────────────────────


@app.get("/v1/default_app")
async def get_default_app(
    category: str,
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Resolve the OS-default application for ``category``.

    ``category`` is one of ``mail | browser | calendar | files``.
    On a backend without a default-app resolver, returns 503; when
    the category is recognised but no default is set, returns 404.
    """
    _check_auth(x_desktop_token)
    _capability_or_503("default_apps_resolver")
    if category not in _CATEGORY_TO_URL:
        raise HTTPException(400, f"unknown category: {category!r}")
    try:
        payload = await _BACKEND.resolve_default_app(category)
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc))
    if payload is None:
        raise HTTPException(404, f"no default app for category {category!r}")
    _audit("default_app", category=category,
           app_name=payload.get("app_name"),
           bundle_id=payload.get("bundle_id"),
           scriptable=payload.get("scriptable"))
    return payload


@app.get("/v1/cursor_activity")
async def cursor_activity(
    x_desktop_token: str | None = Header(default=None),
) -> dict:
    """Return ``{x, y, idle_s}`` — seconds since the cursor last moved.

    Used by the orchestrator's ``computer_use`` tool as a
    conflict-protection guard: when the user is currently at the
    keyboard, vision-driven UI ops would fight the user's own mouse,
    so the orchestrator refuses them.  AppleScript-only paths (no
    mouse / cursor manipulation) still proceed.
    """
    _check_auth(x_desktop_token)
    _capability_or_503("cursor_activity")
    snapshot = _BACKEND.cursor_activity()
    if snapshot is None:
        # Cursor poll hasn't sampled yet — report a huge idle so callers
        # don't refuse vision ops on first turn after boot.
        return {"x": 0, "y": 0, "idle_s": 999.0, "warm": False}
    return {**snapshot, "warm": True}


# ─── Reverse-WSS mode (Phase 3c) ───────────────────────────────────────
#
# Default mode (DESKTOP_MODE=server, unset) starts uvicorn and serves
# HTTP — the orchestrator dials us.  Reverse mode (DESKTOP_MODE=reverse)
# inverts the polarity: this agent dials the orchestrator's
# /v1/agent/connect WSS endpoint and holds the connection.  The
# orchestrator pushes RPC calls down the same socket.
#
# Use case: agent behind NAT (laptop at a coffee shop, work-pc on
# another LAN).  Tailscale/Wireguard is the recommended NAT-traversal
# stack but reverse-WSS works without one — the agent just needs
# outbound TLS to the orchestrator.
#
# Protocol: see orchestrator/app/agent_proxy.py for the v1 spec.  No
# schema versioning yet — bump the ``version`` field on a breaking
# change.

REVERSE_MODE = os.environ.get("DESKTOP_MODE", "server").strip().lower() == "reverse"
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "")


# Method dispatch table for reverse-mode RPC.  Maps the orchestrator's
# ``method`` field to a coroutine that takes ``params`` and returns a
# JSON-serialisable dict.  Keep this table here (not on the Backend
# ABC) so HTTP and WSS share the same backend code without coupling
# the backend to a transport-specific message shape.
async def _reverse_dispatch(method: str, params: dict) -> dict:
    """Execute one reverse-mode RPC.

    Mirrors the HTTP route layer's audit + capability semantics so the
    operator's audit log captures both transports identically.  Raises
    on unsupported method/capability so the WSS handler can surface
    ``ok=False`` to the orchestrator.
    """
    t0 = time.monotonic()
    caps = _BACKEND.capabilities()

    def _require(feature: str) -> None:
        if not caps.get(feature):
            raise RuntimeError(f"{feature} not supported on this {_BACKEND.name}")

    if method == "applescript":
        _require("applescript")
        result = await _BACKEND.applescript(
            params.get("script") or "",
            float(params.get("timeout") or 30.0),
        )
        _audit("applescript", script=(params.get("script") or "")[:400],
               exit=result.get("exit"),
               elapsed_ms=int((time.monotonic() - t0) * 1000), mode="reverse")
        return result
    if method == "pyautogui":
        _require("pyautogui")
        action = params.get("action") or ""
        if action not in _ALLOWED_PYAUTOGUI_ACTIONS:
            raise RuntimeError(f"unknown action: {action!r}")
        out = await _BACKEND.pyautogui_action(
            action,
            x=params.get("x"), y=params.get("y"),
            text=params.get("text"), keys=params.get("keys"),
            clicks=params.get("clicks"), delay=params.get("delay") or 0.0,
        )
        _audit("pyautogui", action=action, x=params.get("x"), y=params.get("y"),
               text=(params.get("text") or "")[:40], keys=params.get("keys"),
               elapsed_ms=int((time.monotonic() - t0) * 1000), mode="reverse")
        return out
    if method == "key":
        _require("hotkey")
        keys = params.get("keys") or []
        out = await _BACKEND.hotkey(keys)
        _audit("key", keys=keys, elapsed_ms=int((time.monotonic() - t0) * 1000),
               mode="reverse")
        return out
    if method == "screenshot":
        _require("screenshot")
        png = await _BACKEND.screenshot()
        _audit("screenshot", bytes=len(png),
               elapsed_ms=int((time.monotonic() - t0) * 1000), mode="reverse")
        # WSS is text-only — base64 the PNG so the orchestrator can
        # round-trip it through json.loads/dumps.
        return {"png_b64": base64.b64encode(png).decode("ascii")}
    if method == "camera":
        _require("camera")
        jpg = await _BACKEND.camera_capture()
        _audit("camera_capture", bytes=len(jpg),
               elapsed_ms=int((time.monotonic() - t0) * 1000), mode="reverse")
        # Same base64 envelope as screenshot.
        return {"jpg_b64": base64.b64encode(jpg).decode("ascii")}
    if method == "default_app":
        _require("default_apps_resolver")
        cat = params.get("category") or ""
        payload = await _BACKEND.resolve_default_app(cat)
        _audit("default_app", category=cat,
               app_name=(payload or {}).get("app_name"),
               elapsed_ms=int((time.monotonic() - t0) * 1000), mode="reverse")
        # None payload → empty dict; reverse callers check for app_name.
        return payload or {}
    if method == "cursor_activity":
        _require("cursor_activity")
        snap = _BACKEND.cursor_activity() or {
            "x": 0, "y": 0, "idle_s": 999.0, "warm": False,
        }
        return snap

    # ── Browser (CDP) — cross-platform, no _require() guard needed ──────
    # _cdp_ensure_reachable() raises HTTPException(503) when Chrome isn't
    # up; the WSS recv loop catches all exceptions and maps them to
    # {"ok": False, "error": <str>} responses, so HTTPException propagates
    # correctly without any special handling here.

    if method == "browser_tabs":
        await _cdp_ensure_reachable()
        tabs = await _cdp_list_tabs()
        _audit("browser_tabs", count=len(tabs), mode="reverse")
        return {"tabs": tabs}

    if method == "browser_navigate":
        await _cdp_ensure_reachable()
        url = params.get("url") or ""
        _require_web_url(url)  # http/https only — see _require_web_url
        tab_id = params.get("tab_id")
        if tab_id:
            tab = await _cdp_resolve_tab(tab_id)
            result = await _cdp_ws_call(
                tab["ws_url"], "Page.navigate", {"url": url}, timeout=15.0,
            )
            _audit("browser_navigate", url=url, tab_id=tab["id"], mode="reverse")
            return {
                "ok": True, "tab_id": tab["id"],
                "url": result.get("url") or url,
                "frame_id": result.get("frameId"),
            }
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{_CDP_BASE}/json/new?{quote(url, safe='')}")
            r.raise_for_status()
            new_info = r.json()
        tab_id_new = new_info.get("id", "")
        _audit("browser_navigate", url=url, tab_id=tab_id_new, action="new_tab", mode="reverse")
        return {"ok": True, "tab_id": tab_id_new, "url": url}

    if method == "browser_js":
        await _cdp_ensure_reachable()
        tab = await _cdp_resolve_tab(params.get("tab_id"))
        code = params.get("code") or ""
        result = await _cdp_ws_call(
            tab["ws_url"],
            "Runtime.evaluate",
            {
                "expression": code,
                "returnByValue": bool(params.get("return_by_value", True)),
            },
            timeout=float(params.get("timeout") or 10.0),
        )
        _audit("browser_js", code=code[:200], tab_id=tab["id"], mode="reverse")
        rv = result.get("result") or {}
        return {
            "ok": True, "tab_id": tab["id"],
            "type": rv.get("type"),
            "value": rv.get("value"),
            "description": rv.get("description"),
        }

    if method == "browser_screenshot":
        await _cdp_ensure_reachable()
        tab = await _cdp_resolve_tab(params.get("tab_id"))
        result = await _cdp_ws_call(
            tab["ws_url"],
            "Page.captureScreenshot",
            {"format": "png", "fromSurface": True},
            timeout=15.0,
        )
        png_b64 = result.get("data") or ""
        if not png_b64:
            raise RuntimeError("CDP returned empty screenshot data")
        png = base64.b64decode(png_b64)
        _audit("browser_screenshot", tab_id=tab["id"], bytes=len(png), mode="reverse")
        # Same base64 envelope as screenshot / camera — orchestrator decodes it.
        return {"png_b64": base64.b64encode(png).decode("ascii")}

    if method == "browser_page_text":
        await _cdp_ensure_reachable()
        tab = await _cdp_resolve_tab(params.get("tab_id"))
        result = await _cdp_ws_call(
            tab["ws_url"],
            "Runtime.evaluate",
            {
                "expression": (
                    "(function(){"
                    "  var b = document.body;"
                    "  return b ? (b.innerText || b.textContent || '') : '';"
                    "})()"
                ),
                "returnByValue": True,
            },
            timeout=10.0,
        )
        rv = result.get("result") or {}
        text = rv.get("value") or ""
        _audit("browser_page_text", tab_id=tab["id"], chars=len(text), mode="reverse")
        return {"text": text, "url": tab["url"], "title": tab["title"], "tab_id": tab["id"]}

    raise RuntimeError(f"unknown method: {method!r}")


async def _reverse_loop_once(url: str) -> None:
    """One connection attempt.  Returns on disconnect; caller retries.

    Uses ``websockets.connect`` (pulled in transitively by
    ``uvicorn[standard]``).  On each frame:
      • ``call`` → execute via _reverse_dispatch, send back a
        ``result`` with the data or the error string.
      • ``ping``/``pong`` → keepalive.
      • anything else → log + ignore (forward-compat).
    """
    import websockets

    log.info("reverse: connecting to %s", url)
    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        # 1. Hello.  Send agent_id, token, capabilities, version so the
        # orchestrator can route incoming calls.
        await ws.send(json.dumps({
            "type": "hello",
            "agent_id": _AGENT_ID,
            "token": SHARED_TOKEN,
            "platform": _BACKEND.name,
            "capabilities": _BACKEND.capabilities(),
            "version": _VERSION,
        }))
        # 2. Wait for hello_ack OR reject.
        ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        ack = json.loads(ack_raw)
        if ack.get("type") == "reject":
            log.error("reverse: rejected by orchestrator: %s", ack.get("reason"))
            return
        if ack.get("type") != "hello_ack":
            log.error("reverse: expected hello_ack, got %r", ack.get("type"))
            return
        log.info("reverse: handshake ok (session=%s)", ack.get("session_id"))

        # 3. Keepalive task — ping the orchestrator every 30 s.  Done
        # as a side task so we can read+ping concurrently without
        # blocking either path.
        async def _keepalive():
            try:
                while True:
                    await asyncio.sleep(30.0)
                    await ws.send(json.dumps({"type": "ping"}))
            except Exception:
                pass

        ka_task = asyncio.create_task(_keepalive())
        try:
            # 4. Recv loop.
            async for raw in ws:
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("reverse: bad JSON frame, skipping")
                    continue
                ftype = frame.get("type")
                if ftype == "call":
                    call_id = frame.get("call_id")
                    method = frame.get("method") or ""
                    params = frame.get("params") or {}
                    try:
                        data = await _reverse_dispatch(method, params)
                        await ws.send(json.dumps({
                            "type": "result", "call_id": call_id,
                            "ok": True, "data": data,
                        }))
                    except Exception as exc:
                        await ws.send(json.dumps({
                            "type": "result", "call_id": call_id,
                            "ok": False, "error": str(exc),
                        }))
                elif ftype == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                elif ftype == "pong":
                    pass
                else:
                    log.warning("reverse: unknown frame type %r", ftype)
        finally:
            ka_task.cancel()


async def reverse_loop() -> None:
    """Persistent reconnect loop — exponential backoff up to 30 s.

    Spec: 1 s, 2 s, 4 s, 8 s, 16 s, then capped at 30 s.  On a
    successful connection the backoff resets to 1 s.
    """
    if not ORCHESTRATOR_URL:
        log.error(
            "reverse: DESKTOP_MODE=reverse requires ORCHESTRATOR_URL "
            "(e.g. wss://my-orch.tailnet.ts.net/v1/agent/connect)"
        )
        return
    backoff = 1.0
    while True:
        try:
            await _reverse_loop_once(ORCHESTRATOR_URL)
            backoff = 1.0  # clean disconnect — reset
        except Exception as exc:
            log.warning("reverse: connection error: %s", exc)
        # Reconnect after backoff; cap at 30 s.
        log.info("reverse: reconnecting in %.1f s", backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


# ─── Entry point ───────────────────────────────────────────────────────


if __name__ == "__main__":
    if REVERSE_MODE:
        log.info(
            "desktop-agent starting in REVERSE mode → %s (platform=%s, audit=%s)",
            ORCHESTRATOR_URL or "<unset!>", _BACKEND.name, _AUDIT_LOG,
        )
        asyncio.run(reverse_loop())
    else:
        log.info(
            "desktop-agent starting on %s:%d (platform=%s, audit=%s)",
            HOST_BIND, PORT, _BACKEND.name, _AUDIT_LOG,
        )
        uvicorn.run(app, host=HOST_BIND, port=PORT, log_level="info")
