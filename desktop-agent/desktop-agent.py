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
  GET  /v1/screenshot       — full-screen PNG snapshot
  POST /v1/audit            — write a free-form audit entry
  GET  /v1/default_app      — resolve the user's default app for a category
  GET  /v1/cursor_activity  — cursor position + idle-seconds (conflict guard)

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
import secrets
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx  # noqa: F401  — imported so wheel resolves; future probes may use it
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
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


def _audit(event: str, **fields: Any) -> None:
    """Append one structured event line to the audit log."""
    record = {"ts": time.time(), "event": event, **fields}
    try:
        with _AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.warning("audit: failed to write event=%s", event)


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
    except (ImportError, Exception) as exc:  # noqa: BLE001 — pyautogui raises bare Exception on display failure
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


# ─── App ───────────────────────────────────────────────────────────────


_VERSION = "1.2.0"
app = FastAPI(title="desktop-agent", version=_VERSION)


@app.on_event("startup")
async def startup() -> None:
    """Log readiness and spin up the cursor-activity poll task.

    Without the poll the /v1/cursor_activity endpoint would have no
    data to report; with it, every poll tick (10 Hz) updates the
    module-level snapshot the endpoint reads.
    """
    global _CURSOR_TASK
    caps = _BACKEND.capabilities()
    log.info(
        "desktop-agent ready: id=%s platform=%s version=%s caps=%s",
        _AGENT_ID, _BACKEND.name, _VERSION,
        ",".join(k for k, v in caps.items() if v) or "<none>",
    )
    # Start the cursor poll task IFF the backend has pyautogui — without
    # it the position() call below would crash.  Track the handle so
    # shutdown can cancel cleanly under uvicorn.
    pg = getattr(_BACKEND, "_pyautogui", None)
    if pg is not None and (_CURSOR_TASK is None or _CURSOR_TASK.done()):
        _CURSOR_TASK = asyncio.create_task(_cursor_poll_loop(pg))


@app.on_event("shutdown")
async def shutdown() -> None:
    global _CURSOR_TASK
    if _CURSOR_TASK is not None and not _CURSOR_TASK.done():
        _CURSOR_TASK.cancel()
        try:
            await _CURSOR_TASK
        except (asyncio.CancelledError, Exception):
            pass
    _CURSOR_TASK = None


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
    """
    caps = _BACKEND.capabilities()
    return {
        "ok": True,
        "agent_id": _AGENT_ID,
        "platform": _BACKEND.name,
        "version": _VERSION,
        "capabilities": caps,
        # Legacy keys — orchestrator's pre-1.1 code reads these.
        "engines": {
            "applescript": caps.get("applescript", False),
            "pyautogui": caps.get("pyautogui", False),
            "xdotool": getattr(_BACKEND, "_has_xdotool", False),
            "pywinauto": getattr(_BACKEND, "_has_pywinauto", False),
        },
        "audit_log": str(_AUDIT_LOG),
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
