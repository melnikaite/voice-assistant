"""
computer_use — agentic GUI driver via LLM-generated AppleScript.

The LLM passes a free-form ``goal`` string ("set volume to 80", "open
google.com", "switch keyboard layout", "show today's calendar events").
computer_use translates the goal into a single AppleScript snippet via
a focused secondary LLM call, classifies it through the existing
read-only verb gate, and executes it on the host's desktop-agent.

Why this shape (vs. per-intent ``elif`` branches)
─────────────────────────────────────────────────
The previous version hard-coded ``search_mail`` / ``read_mail`` /
``open_url`` / ``show_calendar`` / ``find_file`` / ``open_app`` as
six discrete intents, each with its own builder.  That model doesn't
scale: macOS users have hundreds of scriptable apps (Mail, Calendar,
Music, Spotify, Slack, Telegram, OmniFocus, Photoshop, …) plus
system primitives (volume, brightness, keyboard layout, brightness)
— encoding each in Python is just duplicating the LLM's pretrained
knowledge of AppleScript dictionaries.

Generation pipeline
───────────────────
1.  Goal → AppleScript via a focused chat prompt with examples and an
    ``UNKNOWN`` sentinel for goals the model can't express.
2.  Risk classification via
    :func:`app.tools.desktop._classify_applescript_risk` with
    ``category="computer_use"`` — forces the read-only verb allowlist.
3.  Execution via
    :func:`app.desktop_client.run_applescript` on the resolved agent.

Read-only stance
────────────────
Three independent gates refuse destructive scripts:

  •  Generator system prompt rejects delete / send / empty / save / do
     shell script and returns ``UNKNOWN``.
  •  Static classifier rejects substring matches of forbidden verbs
     (defence against a creative generator that ignores its prompt).
  •  Static classifier upgrades risk for ``new event`` / ``set name``
     style mutations; we refuse anything classified above ``read``.

Phase-2 vision-rail fallback
────────────────────────────
When the generator returns ``UNKNOWN`` (goals that can't be expressed
as a single AppleScript snippet — multi-step UI navigation, dialog-
driven flows, free-form file search), we drop into the agentic
vision-loop in :mod:`.vision_loop`.  Each round of the loop is one
screenshot → multimodal-LLM-plans-next-action → execute.  The loop
carries its own three-layer defence (cursor-activity refusal,
destructive-text blacklist on every planned click, strict JSON action
contract) — see :mod:`.vision_loop` for the full surface.

Auth
────
Same gate as the rest of the tools: voice ID is auth.  Without a
recognised ``profile_id`` on the context we refuse before calling the
LLM (saves a token-spend on an unauthenticated speaker).
"""
from __future__ import annotations

import logging
import re

from .. import desktop_client, llm_utils, vision_loop
from ..i18n import t
from .base import ToolResult, tool, unwrap_ctx
from .desktop import DENIED_READONLY_VIOLATION, _classify_applescript_risk

log = logging.getLogger(__name__)


# ── LLM generation prompt ────────────────────────────────────────────
#
# Kept short and example-rich because Gemma 4 E4B picks up patterns
# from few-shot examples more reliably than from prose.  Examples cover
# the four task families we expect: system-state mutations (volume,
# layout), app activation, scripted reads (calendar/mail/music), and
# URL launching — plus one negative example so the model learns when
# to emit UNKNOWN instead of forcing a script.

_GENERATOR_SYSTEM_PROMPT = """\
You are an AppleScript expert.  Given a user's goal in plain language,
output the SIMPLEST AppleScript that accomplishes it on macOS.

Rules:
- Output ONLY the script.  No markdown fences, no explanation, no
  commentary, no leading or trailing prose.
- Read-only operations and system-state changes (volume, brightness,
  keyboard layout, screen lock, app activation) are allowed.
- NEVER output destructive verbs: delete, remove, empty, send, save,
  do shell script, close, quit, make new, duplicate, move.
- If the goal CANNOT be expressed in a single AppleScript snippet,
  output exactly one word: UNKNOWN

Examples:

goal: set volume to 80
answer: set volume output volume 80

goal: mute the speakers
answer: set volume with output muted

goal: switch keyboard layout
answer: tell application "System Events" to key code 49 using control down

goal: open google.com
answer: open location "https://google.com"

goal: bring telegram to foreground
answer: tell application "Telegram" to activate

goal: what is the current track in music
answer: tell application "Music" to return (name of current track) & " — " & (artist of current track)

goal: list today's calendar events
answer: tell application "Calendar" to return summary of every event of every calendar whose start date ≥ (current date) - (time of (current date)) and start date < (current date) + 1 * days

goal: find unread emails from github in the last week
answer: tell application "Mail" to return subject of (messages of inbox whose read status is false and (sender contains "github") and (date received > ((current date) - 7 * days)))

goal: delete all photos from 2020
answer: UNKNOWN

goal: send a message to my boss
answer: UNKNOWN
"""


# Strip markdown fences if the generator emits them despite the prompt
# — some models can't resist wrapping code.  Anchored at line edges
# so we don't munch inline backticks inside an AppleScript string.
_FENCE_RE = re.compile(r"^```(?:applescript|osascript)?\s*|\s*```$",
                       re.MULTILINE)


# Reject runaway generations.  A legitimate AppleScript one-liner fits
# comfortably under 600 chars; anything bigger is either a misfire or
# an attempt to smuggle in long destructive blocks the regex won't
# catch by accident.
_MAX_SCRIPT_LEN = 1500


# Token budget for the generation call.  AppleScript one-liners fit
# in 50-200 tokens; the cap exists so a stuck model can't burn through
# a session's budget.
_GEN_MAX_TOKENS = 256


async def _llm_generate_applescript(
    goal: str,
    *,
    client_id: str | None,
) -> str | None:
    """Ask the local LLM to translate a free-form goal to AppleScript.

    Returns the cleaned script string on success, or ``None`` for
    ``UNKNOWN`` / parse failure / transport error.  Caller MUST still
    risk-classify before executing — the generator's prompt forbids
    destructive verbs but we don't trust prompt-only constraints.
    """
    messages = [
        {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"goal: {goal.strip()}\nanswer:"},
    ]
    try:
        choice = await llm_utils.chat(
            messages=messages,
            temperature=0.1,   # deterministic — we want the canonical script
            max_tokens=_GEN_MAX_TOKENS,
            client_id=client_id,
            tool_name="computer_use",
        )
    except Exception as exc:
        log.warning("computer_use: LLM generation failed: %s", exc)
        return None

    text = (choice.get("message") or {}).get("content") or ""
    text = _FENCE_RE.sub("", text).strip()
    if not text or text.upper() == "UNKNOWN":
        return None
    if len(text) > _MAX_SCRIPT_LEN:
        log.warning(
            "computer_use: rejecting %d-char script (max %d)",
            len(text), _MAX_SCRIPT_LEN,
        )
        return None
    return text


@tool(
    name="computer_use",
    description=(
        "Drive the user's computer to accomplish a goal.  Free-form: "
        "pass what you want to happen in plain language ('set volume "
        "to 80', 'switch keyboard layout', 'open google.com', 'show "
        "today's calendar', 'play next track in music', 'find unread "
        "emails from github', 'open settings and switch to dark mode').  "
        "The tool first tries to express the goal as a single "
        "AppleScript snippet; goals that need multi-step UI navigation "
        "fall back to a vision-driven loop (screenshot → plan → click/"
        "type/key → repeat) that drives the screen the same way a "
        "human would.  Read-only — destructive verbs (delete, send, "
        "empty, save) are refused at multiple layers.  Use this for "
        "any 'open X', 'set Y to N', 'show me Z', 'switch …', 'find Y "
        "in the app' request that targets an app or system setting.  "
        "This is ALSO the tool for changing browser or window state — "
        "closing / focusing / reordering tabs and windows ('close every "
        "Chrome tab except the first', 'switch to the other window').  "
        "`list_browser_tabs` only enumerates tabs; route anything that "
        "acts on them here and let the refusal layers decide.  Pick this "
        "tool even when the goal may be refused — do not silently answer "
        "in text instead."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "Plain-language description of what to do on the "
                    "computer.  Be specific — 'open google.com in "
                    "browser', 'set volume to 80', 'switch keyboard "
                    "to russian', 'show today's calendar events', "
                    "'play next track'.  Pass the user's original "
                    "phrasing if unsure."
                ),
            },
            "agent_id": {
                "type": "string",
                "description": (
                    "Which desktop-agent to drive.  Omit to use the "
                    "default; pass when the user names a device."
                ),
            },
        },
        "required": ["goal"],
    },
    risk="read",
    tier="device",
    device_kind="macos_agent",
)
async def computer_use(
    *, ctx, goal: str, agent_id: str | None = None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    if cx.profile_id is None:
        return ToolResult(
            text=t("computer_use.auth_required", cx.user_lang),
            data={"error": "auth_required"},
        )

    # ── Resolve agent + reachability ─────────────────────────────────
    info = desktop_client.get_agent(agent_id)
    chosen = info.agent_id if info else None
    if chosen is None or not desktop_client.is_reachable(chosen):
        return ToolResult(
            text=t("desktop.unreachable", cx.user_lang),
            data={
                "error": "desktop_unreachable", "goal": goal,
                **({"agent_id": chosen} if chosen else {}),
            },
        )

    # ── Skip AppleScript on non-macOS agents ─────────────────────────
    # AppleScript is macOS-only — Linux and Windows agents advertise
    # ``applescript: False`` in /v1/capabilities.  When the cache says
    # NO we skip the generator entirely (saves a 1-2 s LLM round-trip
    # on a script that wouldn't execute) and go straight to the
    # platform-agnostic vision-loop fallback.  Cache miss (``None``)
    # falls through to the optimistic path; the next health-poll tick
    # populates it and subsequent calls take the fast skip.
    has_applescript = desktop_client.has_capability_cached(
        "applescript", agent_id=chosen,
    )
    if has_applescript is False:
        log.info(
            "computer_use: agent %r lacks applescript — vision-loop fallback for %r",
            chosen, goal,
        )
        return await _route_vision_loop(cx, chosen, goal)

    # ── Step 1: LLM → AppleScript ────────────────────────────────────
    script = await _llm_generate_applescript(goal, client_id=cx.client_id)
    if script is None:
        # Generator said UNKNOWN — goal can't be expressed as a single
        # AppleScript snippet.  Fall to the phase-2 vision-loop:
        # screenshot → multimodal planner → execute → repeat.  The
        # loop carries its own three-layer defence (cursor-activity
        # refusal, destructive-text blacklist, strict JSON action
        # contract) so it's safe to invoke from the read-only tool.
        log.info("computer_use: generator UNKNOWN — falling to vision loop for %r", goal)
        return await _route_vision_loop(cx, chosen, goal)

    # ── Step 2: risk-classify with the strict read-only gate ─────────
    # category="computer_use" forces _FORBIDDEN_IN_READONLY checks
    # (delete/send/empty/etc.) and refuses outright via the sentinel.
    effective = _classify_applescript_risk(
        script, declared="read", category="computer_use",
    )
    if effective == DENIED_READONLY_VIOLATION:
        log.info(
            "computer_use: refused destructive script for goal %r: %s",
            goal, script[:200],
        )
        return ToolResult(
            text=t("computer_use.readonly_violation", cx.user_lang),
            data={
                "error": "readonly_violation", "goal": goal,
                "script_excerpt": script[:200],
                **({"agent_id": chosen} if chosen else {}),
            },
        )
    if effective != "read":
        # The destructive-pattern detector upgraded above "read" but
        # the strict gate didn't reject — likely a `set name` / `new
        # event` pattern.  We refuse here too because the read-only
        # contract on this tool is absolute; the user can ask through
        # the lower-level `desktop` tool if they really want mutation.
        log.info(
            "computer_use: refused %s-risk script for goal %r: %s",
            effective, goal, script[:200],
        )
        return ToolResult(
            text=t("computer_use.readonly_violation", cx.user_lang),
            data={
                "error": "risk_too_high", "risk": effective, "goal": goal,
                "script_excerpt": script[:200],
                **({"agent_id": chosen} if chosen else {}),
            },
        )

    # ── Step 3: execute on the desktop-agent ─────────────────────────
    try:
        result = await desktop_client.run_applescript(
            script, agent_id=chosen, category="computer_use",
        )
    except desktop_client.DesktopUnavailable as exc:
        log.info("computer_use: desktop unavailable mid-run: %s", exc)
        return ToolResult(
            text=t("desktop.unreachable", cx.user_lang),
            data={
                "error": "desktop_unavailable", "detail": str(exc),
                "goal": goal,
                **({"agent_id": chosen} if chosen else {}),
            },
        )
    except Exception as exc:
        log.exception("computer_use: applescript execution failed")
        return ToolResult(
            text=t("computer_use.execution_failed", cx.user_lang),
            data={
                "error": "execution_failed", "detail": str(exc),
                "goal": goal, "script_excerpt": script[:200],
                **({"agent_id": chosen} if chosen else {}),
            },
        )

    # ── Step 4: format reply ─────────────────────────────────────────
    # AppleScript stdout is usually empty for mutations and a single
    # line for reads.  Speak the stdout back when present; otherwise a
    # generic confirmation.
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if not result.get("ok", True) and stderr:
        # `osascript` returned non-zero — surface the error so the user
        # can adjust ("permission needed", "no such app", etc.).
        return ToolResult(
            text=t("computer_use.execution_failed", cx.user_lang),
            data={
                "error": "applescript_error", "stderr": stderr,
                "goal": goal, "script_excerpt": script[:200],
                **({"agent_id": chosen} if chosen else {}),
            },
        )
    reply = stdout or t("computer_use.done", cx.user_lang)
    return ToolResult(
        text=reply,
        data={
            "goal": goal, "script": script,
            "stdout": stdout, "stderr": stderr,
            **({"agent_id": chosen} if chosen else {}),
        },
    )


# ── Phase-2 vision-loop fallback ─────────────────────────────────────


# Maps :mod:`.vision_loop` error keys → user-facing i18n keys.  Keeps
# the public refusal vocabulary small (three messages); the loop's
# internal error taxonomy stays in ``ToolResult.data`` for debugging.
_LOOP_ERROR_I18N = {
    "user_active":       "computer_use.user_busy",
    "screenshot_failed": "computer_use.execution_failed",
    "plan_unparseable":  "computer_use.loop_failed",
    "max_steps":         "computer_use.loop_failed",
    "planner_fail":      "computer_use.loop_failed",
    "transport":         "desktop.unreachable",
}


async def _route_vision_loop(
    cx, agent_id: str | None, goal: str,
) -> ToolResult:
    """Translate :func:`vision_loop.run_vision_loop` results to ToolResult.

    Success: the planner's terminal ``done.result`` is spoken back
    verbatim (it's already free-form natural language describing what
    happened).  Failure: maps the error key to one of the existing
    refusal i18n keys + stashes the loop's step trace in
    ``ToolResult.data`` for the operator to inspect.
    """
    outcome = await vision_loop.run_vision_loop(
        goal, agent_id=agent_id, client_id=cx.client_id,
    )
    base_data = {
        "goal": goal, "path": "vision_loop",
        "steps": outcome.get("steps") or [],
        **({"agent_id": agent_id} if agent_id else {}),
    }
    if outcome.get("ok"):
        result = (outcome.get("result") or "").strip()
        return ToolResult(
            text=result or t("computer_use.done", cx.user_lang),
            data={**base_data, "loop_result": result},
        )
    err_key = outcome.get("error") or "loop_failed"
    i18n_key = _LOOP_ERROR_I18N.get(err_key, "computer_use.loop_failed")
    return ToolResult(
        text=t(i18n_key, cx.user_lang),
        data={
            **base_data,
            "error": err_key,
            **({"detail": outcome["detail"]} if outcome.get("detail") else {}),
        },
    )
