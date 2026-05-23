"""
desktop — universal host-side automation tool.

ONE LLM-visible tool, three execution modes selected by the LLM:

  • mode='applescript' + script=<AppleScript source> + target_app=<name>
      Runs on macOS via /v1/applescript.  Used for everything that has
      a scripting bridge: Calendar, Mail, Music, Notes, Reminders,
      Safari, Finder, Messages, etc.

  • mode='pyautogui' + action=click|type|hotkey|scroll|move +
      x/y/text/keys/clicks
      Cross-platform GUI input — works on macOS / Linux / Windows.
      Used when AppleScript isn't enough (non-scriptable apps, browser
      content, dialogs).

  • mode='key' + keys=[…]
      Shortcut keystroke (cmd+space, ctrl+s, …).  Convenience wrapper
      over pyautogui hotkey with a cleaner audit-log line.

Risk classification
───────────────────
The LLM declares ``risk`` per call — ``read`` / ``low_write`` /
``high_write``.  We never trust that field blindly:

  1. Static pattern check (``_classify_applescript_risk``) bumps risk
     upward if the AppleScript text looks destructive (``delete``,
     ``new event``, ``send``, ``do shell script``, …).  Never bumps
     downward — the LLM can voluntarily over-classify.

  2. ``mode='applescript'`` calls that target a non-allow-listed app
     are denied outright (audit-logged via the daemon).  The allowlist
     lives in the speaker's ``settings.json.custom.desktop_allowed_apps``;
     empty list → nothing allowed (safer default).

  3. ``risk='high_write'`` calls without an active passphrase auth
     window are deferred into ``pending_actions`` instead of executed.
     This MUST happen INSIDE the tool because the tool decorator's
     ``risk`` is static — we register the tool as ``risk='read'`` so
     the agent loop never auto-defers it, and handle gating manually
     based on the per-call risk.

Audit
─────
Every accepted call AND every denial is logged twice:
  • orchestrator side via Python ``log.info`` — visible in compose logs.
  • desktop-agent side via append-only JSONL — survives orchestrator
    restarts, grep-able for forensics.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..desktop_client import (
    DesktopUnavailable,
    health as desktop_health,
    run_applescript,
    run_key,
    run_pyautogui,
    submit_audit,
)
from ..i18n import t
from ..storage import enqueue_action
from ..user_files import read_settings
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


# ── Risk classifiers ───────────────────────────────────────────────────

# AppleScript verbs that are usually destructive.  Matching is case-
# insensitive and word-bounded.  This is INTENTIONALLY conservative —
# false positives just defer-to-queue (annoying, not dangerous);
# false negatives let the LLM hand-wave its way past auth (dangerous).
_DESTRUCTIVE_PATTERNS = [
    r"\bdelete\b",
    r"\bremove\b",
    r"\bclose\b",
    r"\bsend\b",            # tell application "Mail" … send
    r"\bnew\s+event\b",     # Calendar event creation
    r"\bnew\s+message\b",
    r"\bnew\s+note\b",
    r"\bnew\s+reminder\b",
    r"\bnew\s+document\b",
    r"\bmake\s+new\b",
    r"\bset\s+(content|body|name|location|due\s+date|start\s+date)\b",
    r"\bdo\s+shell\s+script\b",     # !!!
    r"\bemptry\s+trash\b",
    r"\bshut\s*down\b",
    r"\brestart\b",
    r"\bdo\s+JavaScript\b",
]
_DESTRUCTIVE_RE = re.compile("|".join(_DESTRUCTIVE_PATTERNS), re.IGNORECASE)


# Category-aware read-only verb policy.  The computer_use tool tells
# the desktop tool the task category for each AppleScript call (mail
# / calendar / files / browser); when set, we enforce a stricter
# "this script must be read-only" policy that REJECTS calls containing
# any mutation verb regardless of risk classification.  This is a
# defence-in-depth gate above the regular destructive-pattern detector
# — the regular detector upgrades risk and defers; this category gate
# refuses outright with a special sentinel.
_FORBIDDEN_IN_READONLY = {
    "delete", "remove", "empty", "send", "make new", "duplicate",
    "move", "save", "close", "quit", "kill", "do shell script",
    "set read", "set flagged", "set status",
}


# Sentinel string returned by :func:`_classify_applescript_risk` when
# the category gate rejects the script.  Treated specially by the
# desktop tool: it short-circuits to a localised "read-only violation"
# message without going through the usual risk-deferral pipeline.
DENIED_READONLY_VIOLATION = "DENIED_READONLY_VIOLATION"


def _classify_applescript_risk(
    script: str,
    declared: str,
    category: str | None = None,
) -> str:
    """Classify the effective risk of an AppleScript call.

    Risk ordering: read < low_write < high_write.

    Two independent gates:

    1. ``category``-strict read-only check.  If a category is passed,
       any token from ``_FORBIDDEN_IN_READONLY`` appearing in the
       script (case-insensitive substring match) forces a hard reject
       via the :data:`DENIED_READONLY_VIOLATION` sentinel.  The
       computer_use tool sets ``category`` per intent so a "search
       mail" call can't smuggle in a `delete` verb even if the LLM
       declared it ``risk="read"``.

    2. Destructive-pattern UPGRADE.  Existing behaviour preserved:
       any match in ``_DESTRUCTIVE_RE`` bumps ``declared`` up to
       ``high_write``.  Complements the category gate rather than
       replacing it — the LLM can still over-classify voluntarily.

    Returns either the higher risk string, ``declared`` unchanged,
    or :data:`DENIED_READONLY_VIOLATION` on a category-strict reject.
    """
    if category:
        script_lower = script.lower()
        if any(tok in script_lower for tok in _FORBIDDEN_IN_READONLY):
            return DENIED_READONLY_VIOLATION
    order = {"read": 0, "low_write": 1, "high_write": 2}
    if _DESTRUCTIVE_RE.search(script):
        return "high_write" if order.get(declared, 0) < 2 else declared
    return declared


# ── AppleScript app-name extractor ─────────────────────────────────────
#
# Used for the allowlist check.  Looks for the common pattern
#   tell application "<Name>"
# anywhere in the script.  Multiple matches → list of all unique names.
# No matches → empty list (which means "doesn't target any app", e.g.
# `display dialog "hi"` — still gated by general allowlist).

_TELL_APP_RE = re.compile(r'tell\s+application\s+"([^"]+)"', re.IGNORECASE)


def _extract_target_apps(script: str) -> list[str]:
    return list({m.group(1) for m in _TELL_APP_RE.finditer(script)})


# ── The tool ───────────────────────────────────────────────────────────


@tool(
    name="desktop",
    description=(
        "Drive any application on the host machine — AppleScript on "
        "macOS, pyautogui anywhere.  Use for:\n"
        "  • Calendar / Reminders / Mail / Notes / Messages on macOS "
        "(AppleScript)\n"
        "  • Music / Spotify / Safari / Finder (AppleScript)\n"
        "  • System volume, brightness, screen lock (AppleScript or "
        "pyautogui hotkey)\n"
        "  • Click / type / scroll in non-scriptable apps (pyautogui)\n"
        "\nModes:\n"
        "  • mode='applescript' + script=<source> + target_app=<name> + "
        "risk=<read|low_write|high_write>\n"
        "  • mode='pyautogui' + action=<click|doubleclick|type|hotkey|"
        "scroll|move> + per-action args + risk=<…>\n"
        "  • mode='key' + keys=[\"cmd\",\"space\"] + risk=<…>\n"
        "\nPick `risk` honestly:\n"
        "  • read       — reading state (calendar list, get track name, "
        "current volume)\n"
        "  • low_write  — reversible tweaks (music play/pause, volume +/-, "
        "brightness, mute)\n"
        "  • high_write — anything that mutates user data or sends "
        "things (create event, send message, write file, empty trash, "
        "do shell script).  Will be deferred for passphrase approval "
        "if not authenticated.\n"
        "\nThe AppleScript pattern matcher will UPGRADE your risk if "
        "you under-classify — pick correctly the first time so the "
        "user isn't surprised by an unexpected queue entry."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["applescript", "pyautogui", "key"],
            },
            "risk": {
                "type": "string",
                "enum": ["read", "low_write", "high_write"],
                "description": (
                    "Honest risk classification for THIS call.  See main "
                    "description; AppleScript scripts are also pattern-"
                    "checked and upgraded if destructive verbs appear."
                ),
            },
            # AppleScript-mode
            "script": {
                "type": "string",
                "description": (
                    "AppleScript source (mode='applescript').  Use only "
                    "Apple Event verbs of the target application; do NOT "
                    "use 'do shell script' unless you absolutely must "
                    "and intend high_write risk."
                ),
            },
            "target_app": {
                "type": "string",
                "description": (
                    "Name of the application this call drives (mode="
                    "'applescript').  Must match an entry in the "
                    "speaker's `settings.custom.desktop_allowed_apps` "
                    "list."
                ),
            },
            # pyautogui-mode
            "action": {
                "type": "string",
                "enum": ["click", "doubleclick", "rightclick", "move", "type", "scroll", "hotkey"],
            },
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "text": {"type": "string"},
            "clicks": {"type": "integer"},
            # key-mode + pyautogui hotkey
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keyboard shortcut, e.g. ['cmd','space'].",
            },
            "summary": {
                "type": "string",
                "description": (
                    "One short sentence describing what this "
                    "call does — used in the deferred-action queue and "
                    "the spoken confirmation when executed.  Example: "
                    "'Create event \"Meeting with Igor\" tomorrow at 2pm'."
                ),
            },
            # Multi-agent routing.  Static-included in the schema (vs
            # conditionally-omitted like mail/look_at_screen) because
            # desktop is by far the most likely tool to be deliberately
            # targeted at a non-default device (e.g. "open Spotify on my
            # Mac"); keeping the field visible nudges the LLM to use it.
            "agent_id": {
                "type": "string",
                "description": (
                    "Which desktop-agent to drive.  Omit to use the "
                    "user's default device.  Use when the user names "
                    "one (e.g. 'open Spotify on my Mac', 'press the "
                    "button on the work computer')."
                ),
            },
            # Wave 3: when present, the AppleScript risk classifier
            # enforces a category-strict read-only verb policy.  Set
            # by the orchestrator's computer_use tool when it routes
            # an intent through here; never set by the LLM directly.
            "category": {
                "type": "string",
                "enum": ["mail", "calendar", "files", "browser"],
                "description": (
                    "Task category for category-strict read-only "
                    "enforcement.  Internal — set by computer_use, "
                    "not by free-form LLM use of this tool."
                ),
            },
        },
        "required": ["mode", "risk", "summary"],
    },
    # Registered as `read` so the agent-loop's tier-2 gating doesn't
    # auto-defer EVERY desktop call.  We do per-call gating ourselves
    # below based on the LLM-declared (and pattern-validated) risk.
    risk="read",
)
async def desktop(
    *,
    ctx,
    mode: str,
    risk: str,
    summary: str,
    script: str | None = None,
    target_app: str | None = None,
    action: str | None = None,
    x: int | None = None,
    y: int | None = None,
    text: str | None = None,
    clicks: int | None = None,
    keys: list[str] | None = None,
    agent_id: str | None = None,
    category: str | None = None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang
    await cx.progress("desktop", mode)

    # Resolve which agent to drive.  Stamped into every ToolResult.data
    # below so the WS layer can surface `target_agent` to the UI — the
    # agents panel flashes the matching row, the tool line shows
    # "desktop · @macbook" so the user sees which device just acted.
    from .. import desktop_client as _dc
    _agent = _dc.get_agent(agent_id)
    _resolved_agent_id: str | None = _agent.agent_id if _agent else None

    # ── 1. Daemon reachable? ─────────────────────────────────────────
    info = await desktop_health(agent_id=_resolved_agent_id)
    if info is None:
        return ToolResult(
            text=t("desktop.legacy_unavailable", lang),
            data={
                "error": "daemon_unreachable",
                **({"agent_id": _resolved_agent_id} if _resolved_agent_id else {}),
            },
        )
    engines = info.get("engines", {})

    # ── 2. Allowlist check (AppleScript only) ────────────────────────
    if mode == "applescript":
        if not script:
            return ToolResult(text=t("desktop.bad_request", lang), data={"error": "no_script"})
        if not engines.get("applescript"):
            return ToolResult(
                text=t("desktop.not_apple", lang),
                data={"error": "applescript_not_available"},
            )
        # Extract every "tell application X" target in the script.  The
        # LLM's declared target_app is the primary source of truth, but
        # we cross-check against actual references in case the LLM
        # forgot to mention one ("tell application \"Finder\" to … and
        # tell application \"Mail\" to …").
        declared_targets = [target_app] if target_app else []
        actual_targets = _extract_target_apps(script)
        all_targets = list({*declared_targets, *actual_targets})

        allowed = await _get_allowed_apps(cx.profile_id)
        denied = [t for t in all_targets if t and t.lower() not in {a.lower() for a in allowed}]
        if denied:
            await submit_audit(
                "desktop_denied_allowlist",
                profile_id=cx.profile_id,
                client_id=cx.client_id,
                targets=all_targets,
                denied=denied,
                summary=summary,
            )
            return ToolResult(
                text=t("desktop.denied_app", lang, app=denied[0]),
                data={"error": "app_not_allowed", "denied": denied},
            )

        # Pattern-driven risk upgrade (+ category-strict reject when
        # category is set by the orchestrator's computer_use tool).
        effective_risk = _classify_applescript_risk(script, risk, category=category)
        if effective_risk == DENIED_READONLY_VIOLATION:
            # Defence-in-depth: category-strict reject.  The category
            # is set by computer_use only — we know the orchestrator
            # asked for a read-only flow, and the script contains a
            # mutation verb.  Refuse without going through the usual
            # deferral pipeline; the LLM should have used a different
            # intent for state changes (and won't, because computer_use
            # is registered risk="read" with no high_write path).
            await submit_audit(
                "desktop_readonly_violation",
                profile_id=cx.profile_id,
                category=category,
                summary=summary,
                script=(script or "")[:400],
            )
            return ToolResult(
                text=t("desktop.readonly_violation", lang, category=category),
                data={
                    "error": "readonly_violation",
                    "category": category,
                    **({"agent_id": _resolved_agent_id} if _resolved_agent_id else {}),
                },
            )
    else:
        effective_risk = risk  # pyautogui/key — no static analysis, trust LLM
        if not engines.get("pyautogui"):
            return ToolResult(
                text=t("desktop.not_available", lang),
                data={"error": "pyautogui_not_available"},
            )

    # ── 3. Tier-2 gating (per-call risk, not decorator risk) ─────────
    if effective_risk == "high_write" and not cx.is_authenticated:
        # Build a fully-replayable args bundle so the pending_executor
        # can run this verbatim once approved.
        replay_args: dict[str, Any] = {"mode": mode, "risk": effective_risk, "summary": summary}
        for k, v in (
            ("script", script),
            ("target_app", target_app),
            ("action", action),
            ("x", x), ("y", y),
            ("text", text), ("clicks", clicks),
            ("keys", keys),
        ):
            if v is not None:
                replay_args[k] = v
        action_id = await enqueue_action(
            profile_id=cx.profile_id,
            client_id=cx.client_id,
            tool_name="desktop",
            tool_args=replay_args,
            summary=summary,
        )
        await submit_audit(
            "desktop_deferred",
            profile_id=cx.profile_id,
            pending_action_id=action_id,
            mode=mode,
            risk=effective_risk,
            summary=summary,
        )
        log.info("desktop: deferred → pending_actions id=%d (%s)", action_id, summary)
        return ToolResult(
            text=t("auth.action_deferred", lang, summary=summary),
            data={
                "deferred": True,
                "pending_action_id": action_id,
                "risk": effective_risk,
            },
        )

    # Common data tail stamped on every ToolResult below.  Centralised
    # so adding a new field (like ``target_agent`` later) doesn't mean
    # touching N call sites.
    def _data(**extra):
        if _resolved_agent_id:
            extra["agent_id"] = _resolved_agent_id
        return extra

    # ── 4. Execute ───────────────────────────────────────────────────
    try:
        if mode == "applescript":
            result = await run_applescript(
                script or "", agent_id=_resolved_agent_id, category=category,
            )
            ok = result.get("exit") == 0
            stdout = (result.get("stdout") or "").strip()
            note = stdout[:120] if stdout else t("desktop.done_note_default", lang)
            log.info(
                "desktop applescript: exit=%s elapsed=%dms targets=%s agent=%s",
                result.get("exit"), result.get("elapsed_ms", 0), all_targets,
                _resolved_agent_id,
            )
            return ToolResult(
                text=(
                    t("desktop.done", lang, summary=summary, note=note)
                    if ok else
                    t("desktop.failed", lang, summary=summary, why=result.get("stderr", "")[:120])
                ),
                data=_data(mode=mode, exit=result.get("exit"), stdout=stdout),
            )

        if mode == "pyautogui":
            payload: dict[str, Any] = {"action": action}
            for k, v in (("x", x), ("y", y), ("text", text), ("clicks", clicks), ("keys", keys)):
                if v is not None:
                    payload[k] = v
            r = await run_pyautogui(payload, agent_id=_resolved_agent_id)
            log.info(
                "desktop pyautogui: action=%s elapsed=%dms agent=%s",
                action, r.get("elapsed_ms", 0), _resolved_agent_id,
            )
            return ToolResult(
                text=t(
                    "desktop.done", lang,
                    summary=summary, note=t("desktop.done_note_default", lang),
                ),
                data=_data(mode=mode, action=action, elapsed_ms=r.get("elapsed_ms")),
            )

        if mode == "key":
            if not keys:
                return ToolResult(text=t("desktop.bad_request", lang), data=_data(error="no_keys"))
            r = await run_key(keys, agent_id=_resolved_agent_id)
            log.info(
                "desktop key: %s elapsed=%dms agent=%s",
                "+".join(keys), r.get("elapsed_ms", 0), _resolved_agent_id,
            )
            return ToolResult(
                text=t(
                    "desktop.done", lang,
                    summary=summary,
                    note=t("desktop.done_note_keys", lang, keys=" + ".join(keys)),
                ),
                data=_data(mode=mode, keys=keys, elapsed_ms=r.get("elapsed_ms")),
            )

        return ToolResult(text=t("desktop.bad_request", lang), data=_data(error=f"unknown_mode:{mode!r}"))

    except DesktopUnavailable as exc:
        log.warning("desktop: %s", exc)
        return ToolResult(
            text=t("desktop.legacy_unavailable", lang),
            data=_data(error=str(exc)),
        )


# ── Per-speaker allowlist ──────────────────────────────────────────────


async def _get_allowed_apps(profile_id: int | None) -> list[str]:
    """Read the speaker's app allowlist from ``settings.custom.desktop_allowed_apps``.

    Empty list → nothing is allowed.  This is the SAFE default — the
    user explicitly opts apps in through the Settings tab once.  Avoids
    a "first-run can scriptable everything" surprise.
    """
    if profile_id is None:
        return []
    settings = await read_settings(profile_id)
    apps = (settings.custom or {}).get("desktop_allowed_apps") or []
    if not isinstance(apps, list):
        log.warning(
            "desktop: profile=%d has non-list desktop_allowed_apps (%r) — treating as empty",
            profile_id, type(apps).__name__,
        )
        return []
    return [str(a) for a in apps if isinstance(a, str)]
