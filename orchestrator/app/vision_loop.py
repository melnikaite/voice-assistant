"""
Vision-driven agentic loop — phase-2 fallback for ``computer_use``.

When the AppleScript generator inside :mod:`.tools.computer_use` returns
``UNKNOWN`` (the goal can't be expressed as a single AppleScript
snippet), this module takes over: screenshot → multimodal LLM plans
one atomic action → execute via desktop-agent → repeat until done or
budget exhausted.

The contract is intentionally small.  Six action types cover the vast
majority of GUI tasks the AppleScript generator can't handle:

  * ``click``  — click at ``(x, y)``.  Caller MUST declare
    ``target_text`` so the destructive-text blacklist can refuse
    "Delete" / "Send" / "Empty Trash" buttons before the click lands.
  * ``type``   — type a string at the current focus.
  * ``key``    — keyboard shortcut, e.g. ``["cmd", "space"]``.
  * ``wait``   — sleep ``ms`` milliseconds so the next screenshot
    catches the result of a previous action.
  * ``scroll`` — scroll the viewport by ``dy`` lines.
  * ``done``   — goal achieved; ``result`` is the speakable reply.
  * ``fail``   — driver cannot make progress; ``reason`` surfaces to
    the user.

Read-only stance
────────────────
Identical contract to :mod:`.tools.computer_use`.  Three defences:

  1.  Planning system prompt forbids clicks on destructive UI elements.
  2.  :func:`vision_primitives.is_destructive_text` blacklist applied
      to every ``target_text`` before the click runs (defence in
      depth — model could ignore prompt).
  3.  No filesystem / shell escape — pyautogui only knows about pixel
      coordinates + keyboard input.  An AppleScript path is NOT
      available from here.

Conflict protection
───────────────────
The loop refuses to start (and aborts mid-loop) when the cursor-
activity probe reports the user is actively at the keyboard.  Vision-
driven mouse moves fighting the user's own input is the worst
possible failure mode.

Budget
──────
Six steps maximum.  Each step costs ~1 multimodal call (~10s on a
local model) plus one screenshot.  Most reachable goals complete in
3-5 steps; the cap exists so a stuck loop can't burn a session's
token budget on a hopeless plan.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from . import desktop_client, vision, vision_primitives

log = logging.getLogger(__name__)


# Tunables — kept module-level so tests can monkey-patch them without
# touching the loop body.

# Maximum vision-plan rounds before we give up.  Tight by design: a
# six-step plan that can't reach the goal is more likely stuck than
# making slow progress, and each round is ~10 s of wall time.
MAX_STEPS = 6

# Cursor-idle threshold below which we refuse to act.  Tighter than
# :mod:`.tools.computer_use`'s gate (which was 30 s for AppleScript-
# only ops): vision-driven mouse moves are physically visible and
# physically fight the user.  Five seconds is "the user took their
# hands off the keyboard, paused, and asked me to do something".
CURSOR_IDLE_THRESHOLD_S = 5.0

# Token budget per planning call.  Action JSON fits in 80-200 tokens;
# 300 leaves headroom for the model's chain-of-thought before the
# final JSON.
PLAN_MAX_TOKENS = 300

# Cap on free-form text fields the planner can ask us to type.  Long
# typing operations are almost always a wrong plan (the model is
# trying to write a document instead of completing a navigation
# task).  Refuses with a fail action if exceeded.
MAX_TYPE_LEN = 200


# Coordinate bounds — sanity-check the model's pixel claims so a
# hallucinated (-1, 99999) click doesn't go to the desktop-agent.
# Real macOS displays don't exceed 8K either way today.
_COORD_MAX = 8192
_COORD_MIN = 0


# System prompt for the planner.  Strict JSON output; no prose.
# Examples cover the six action types plus a refusal example so the
# model learns to fail-gracefully instead of guessing.
_PLAN_SYSTEM_PROMPT = """\
You drive a Mac computer.  Given the user's goal and a screenshot of
the current screen, output ONE next action that progresses toward
the goal.  Output STRICT JSON — no markdown fences, no commentary.

Allowed actions:

  {"type": "click",  "x": int, "y": int, "target_text": "<label>", "reason": "..."}
  {"type": "type",   "text": "...", "reason": "..."}
  {"type": "key",    "keys": ["cmd", "space"], "reason": "..."}
  {"type": "wait",   "ms": int, "reason": "..."}
  {"type": "scroll", "x": int, "y": int, "dy": int, "reason": "..."}
  {"type": "done",   "result": "<spoken reply>", "reason": "..."}
  {"type": "fail",   "reason": "..."}

Rules:

  * Output ONE action per call.  Loop until the goal is achieved,
    then output {"type": "done", "result": "..."}.
  * NEVER suggest clicks on destructive UI elements: Delete, Remove,
    Send, Empty Trash, Archive, Discard, Trash, "Send Message",
    "Move to Trash".  If the goal would require such a click, output
    {"type": "fail", "reason": "destructive_action_required"}.
  * Use display pixel coordinates; origin top-left.
  * `target_text` is REQUIRED on click — must match the visible text
    label of the element being clicked (used for safety verification).
  * Prefer keyboard navigation (cmd+space for Spotlight, tab/arrow
    keys) over mouse clicks when both work — keyboard is more
    reliable across resolutions.
  * If you can't make progress, output {"type": "fail", "reason":
    "<short explanation>"}.  Do not output click-actions that you
    aren't confident about.

Example sequence to open Spotlight and search:

  step 1: {"type": "key", "keys": ["cmd", "space"], "reason": "open spotlight"}
  step 2: {"type": "wait", "ms": 300, "reason": "let spotlight render"}
  step 3: {"type": "type", "text": "calendar", "reason": "search for app"}
  step 4: {"type": "key", "keys": ["return"], "reason": "open first result"}
  step 5: {"type": "done", "result": "Calendar is open.", "reason": "goal reached"}
"""


# Fence-stripper — same defensive parsing as vision_primitives.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# ── Validation ─────────────────────────────────────────────────────────


_VALID_TYPES = {"click", "type", "key", "wait", "scroll", "done", "fail"}


def _validate_action(raw: Any) -> dict | None:
    """Shape-check the planner's JSON; return cleaned action or None.

    Defence against (a) malformed model output, (b) hallucinated
    coordinates outside any sane display, (c) destructive-text on
    click, (d) over-long type payloads.
    """
    if not isinstance(raw, dict):
        return None
    t = raw.get("type")
    if t not in _VALID_TYPES:
        return None

    if t == "click":
        x, y = raw.get("x"), raw.get("y")
        target = raw.get("target_text") or ""
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        if not (_COORD_MIN <= x <= _COORD_MAX and _COORD_MIN <= y <= _COORD_MAX):
            return None
        if not isinstance(target, str) or not target.strip():
            # Forcing target_text — without it we can't run the
            # destructive blacklist.  Refuse the action.
            return None
        if vision_primitives.is_destructive_text(target):
            return {"type": "fail", "reason": f"destructive_click_refused: {target}"}
        return {"type": "click", "x": x, "y": y,
                "target_text": target.strip(),
                "reason": str(raw.get("reason") or "")}

    if t == "type":
        text = raw.get("text")
        if not isinstance(text, str):
            return None
        if len(text) > MAX_TYPE_LEN:
            return {"type": "fail", "reason": f"type_too_long: {len(text)} chars"}
        return {"type": "type", "text": text,
                "reason": str(raw.get("reason") or "")}

    if t == "key":
        keys = raw.get("keys")
        if not isinstance(keys, list) or not keys:
            return None
        if not all(isinstance(k, str) and k for k in keys):
            return None
        return {"type": "key", "keys": [str(k) for k in keys],
                "reason": str(raw.get("reason") or "")}

    if t == "wait":
        ms = raw.get("ms")
        if not isinstance(ms, int) or ms < 0:
            return None
        # Cap wait at 5 s — long sleeps stall the user; if a step
        # genuinely needs more, the loop should run multiple short waits.
        ms = min(ms, 5000)
        return {"type": "wait", "ms": ms,
                "reason": str(raw.get("reason") or "")}

    if t == "scroll":
        x, y = raw.get("x"), raw.get("y")
        dy = raw.get("dy")
        if not isinstance(x, int) or not isinstance(y, int) or not isinstance(dy, int):
            return None
        if not (_COORD_MIN <= x <= _COORD_MAX and _COORD_MIN <= y <= _COORD_MAX):
            return None
        # Cap scroll magnitude — outrageous values are usually a misread.
        dy = max(-2000, min(2000, dy))
        return {"type": "scroll", "x": x, "y": y, "dy": dy,
                "reason": str(raw.get("reason") or "")}

    if t == "done":
        result = raw.get("result") or ""
        return {"type": "done", "result": str(result).strip(),
                "reason": str(raw.get("reason") or "")}

    # t == "fail"
    return {"type": "fail",
            "reason": str(raw.get("reason") or "unspecified")}


# ── Action execution ──────────────────────────────────────────────────


async def _apply_action(action: dict, *, agent_id: str | None) -> None:
    """Translate a planner action into a desktop-agent call.

    Raises :exc:`desktop_client.DesktopUnavailable` if the agent went
    away mid-loop; the caller treats that as a transport failure and
    aborts the loop.
    """
    t = action["type"]
    if t == "click":
        await desktop_client.run_pyautogui(
            {"action": "click", "x": action["x"], "y": action["y"]},
            agent_id=agent_id,
        )
        return
    if t == "type":
        await desktop_client.run_pyautogui(
            {"action": "type", "text": action["text"]},
            agent_id=agent_id,
        )
        return
    if t == "key":
        await desktop_client.run_key(action["keys"], agent_id=agent_id)
        return
    if t == "wait":
        await asyncio.sleep(action["ms"] / 1000.0)
        return
    if t == "scroll":
        await desktop_client.run_pyautogui(
            {"action": "scroll",
             "x": action["x"], "y": action["y"], "dy": action["dy"]},
            agent_id=agent_id,
        )
        return
    # done / fail are terminal — caller handles, never reach here
    raise ValueError(f"unexecutable action type: {t!r}")


# ── Planning call ─────────────────────────────────────────────────────


async def _plan_next_action(
    goal: str,
    screenshot: bytes,
    history: list[dict],
    *,
    client_id: str | None,
) -> dict | None:
    """One planning round: screenshot → action JSON.

    Returns a validated action dict, or None if the model returned
    unparseable / shape-invalid output.  The caller treats None as
    "abort the loop" — better to surface a generic failure than to
    execute a guessed action.
    """
    # History condensation — a long sequence of past actions bloats
    # the prompt without proportional benefit.  Send only the last
    # 4 actions; the screenshot carries the rest of the state.
    recent = history[-4:] if len(history) > 4 else history
    history_lines = [
        f"  step {i+1}: {a.get('type')} — {a.get('reason') or ''}"
        for i, a in enumerate(recent)
    ]
    question = (
        f"Goal: {goal}\n"
        f"Previous actions ({len(history)} total, showing last {len(recent)}):\n"
        + ("\n".join(history_lines) if history_lines else "  (none yet)")
        + "\n\nWhat is the single next action?  Output JSON only."
    )
    try:
        raw = await vision.analyze_image_bytes(
            screenshot, question,
            client_id=client_id,
            tool_name="computer_use_vision_loop",
            system_prompt=_PLAN_SYSTEM_PROMPT,
            max_tokens=PLAN_MAX_TOKENS,
        )
    except Exception:
        log.exception("vision_loop: plan call failed")
        return None
    text = (raw or "").strip()
    if not text:
        return None
    # Strip ```json fences if the model wrapped its reply.
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.debug("vision_loop: non-JSON plan reply: %r", text[:200])
        return None
    return _validate_action(parsed)


# ── Cursor-activity guard ─────────────────────────────────────────────


async def _user_is_active(agent_id: str | None) -> bool:
    """Cheap probe — returns True iff cursor-activity says user is at
    the keyboard right now.  False on probe failure (fail-open: a
    broken cursor service shouldn't lock out the loop forever).
    """
    try:
        activity = await desktop_client.cursor_activity(agent_id=agent_id)
    except desktop_client.DesktopUnavailable:
        return False
    if not activity:
        return False
    if not activity.get("warm"):
        return False
    idle = activity.get("idle_s")
    if not isinstance(idle, (int, float)):
        return False
    return float(idle) < CURSOR_IDLE_THRESHOLD_S


# ── Public driver ─────────────────────────────────────────────────────


async def run_vision_loop(
    goal: str,
    *,
    agent_id: str | None,
    client_id: str | None,
) -> dict:
    """Drive the screen toward ``goal`` using the vision-LLM planner.

    Returns a dict with terminal status, suitable for the caller to
    map to a :class:`ToolResult`:

      {"ok": True,  "result": "<speakable reply>", "steps": [...]}
      {"ok": False, "error": "<error_key>", "detail": "...", "steps": [...]}

    Error keys (stable, for i18n routing):

      * ``user_active``  — cursor activity says user is at keyboard
      * ``screenshot_failed`` — agent couldn't capture
      * ``plan_unparseable``  — planner returned bad JSON
      * ``max_steps``    — budget exhausted without ``done``
      * ``planner_fail`` — planner emitted ``{"type": "fail", ...}``
      * ``transport``    — desktop-agent went away mid-loop
    """
    if await _user_is_active(agent_id):
        return {"ok": False, "error": "user_active", "steps": []}

    steps: list[dict] = []
    for step_idx in range(MAX_STEPS):
        # Mid-loop user-activity check — the user may have started
        # typing between actions.  Cheap probe; refuse fast.
        if step_idx > 0 and await _user_is_active(agent_id):
            return {
                "ok": False, "error": "user_active",
                "steps": steps,
            }

        try:
            png = await desktop_client.screenshot(agent_id=agent_id)
        except desktop_client.DesktopUnavailable as exc:
            return {
                "ok": False, "error": "transport",
                "detail": str(exc), "steps": steps,
            }
        if not png:
            return {
                "ok": False, "error": "screenshot_failed",
                "steps": steps,
            }

        action = await _plan_next_action(
            goal, png, steps, client_id=client_id,
        )
        if action is None:
            return {
                "ok": False, "error": "plan_unparseable",
                "steps": steps,
            }

        steps.append(action)

        if action["type"] == "done":
            return {
                "ok": True,
                "result": action.get("result") or "",
                "steps": steps,
            }
        if action["type"] == "fail":
            return {
                "ok": False, "error": "planner_fail",
                "detail": action.get("reason") or "",
                "steps": steps,
            }

        try:
            await _apply_action(action, agent_id=agent_id)
        except desktop_client.DesktopUnavailable as exc:
            return {
                "ok": False, "error": "transport",
                "detail": str(exc), "steps": steps,
            }
        except Exception as exc:
            log.exception("vision_loop: apply_action failed")
            return {
                "ok": False, "error": "transport",
                "detail": str(exc), "steps": steps,
            }

    return {"ok": False, "error": "max_steps", "steps": steps}
