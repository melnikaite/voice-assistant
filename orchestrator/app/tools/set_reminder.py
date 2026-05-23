"""
reminders — unified CRUD tool for scheduling.

Actions
───────
  create  Schedule a new reminder or countdown timer.
  list    Speak the upcoming reminders for this client.
  cancel  Cancel a pending reminder by fuzzy-matching its text.

The tool is still registered in set_reminder.py so __init__.py doesn't
need to change; the LLM-visible name is ``reminders``.

Spoken durations and time anchors come from ``i18n.format_duration_seconds``
and ``i18n.format_when`` (Babel under the hood) — see ``i18n.py`` for
the locale wiring.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from .. import scheduler as sched
from ..i18n import format_duration_seconds, format_when, t
from ..storage import (
    add_reminder,
    cancel_reminder_db,
    list_upcoming_reminders,
)
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

# Hard caps for countdown timers / absolute reminders.
_MIN_SECONDS = 1
_MAX_DURATION_SECONDS = 7 * 86400      # 7 days for countdown
_MAX_ABSOLUTE_HORIZON_S = 30 * 86400   # 30 days for calendar entries


def _fmt_reminder_for_list(r: dict, now: float, lang: str | None) -> str:
    """One spoken entry for the list action, e.g. '«eggs» in 4 minutes'."""
    push_text = r["push_text"]
    when = format_when(r["fire_at"], now, lang)
    return f"«{push_text}» {when}"


# ── Fuzzy cancel matching ──────────────────────────────────────────────────

def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets (case-insensitive)."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _best_match(
    query: str, reminders: list[dict]
) -> tuple[dict | None, float]:
    """Return the reminder most similar to ``query`` and its score."""
    best: dict | None = None
    best_score = 0.0
    for r in reminders:
        score = _word_overlap(query, r["push_text"])
        if score > best_score:
            best_score = score
            best = r
    return best, best_score


# ── Tool ──────────────────────────────────────────────────────────────────

@tool(
    name="reminders",
    description=(
        "Manage reminders and countdown timers.\n"
        "  • action='create' + trigger='duration' + seconds=N — countdown "
        "timer ('set a timer for 10 minutes', 'remind me in half an hour').\n"
        "  • action='create' + trigger='absolute' + fire_at=ISO8601 — calendar "
        "entry ('remind me at 15:30', 'meeting tomorrow at 9'). Use the "
        "current local time provided in this tool's description (below) to "
        "convert natural phrasing to ISO-8601.\n"
        "  • action='list' — read out upcoming reminders for this client.\n"
        "  • action='cancel' + text=<what to cancel> — cancel the reminder "
        "whose text best matches the given phrase.\n"
        "`text` is: the reminder phrase for create (e.g. 'eggs', 'call mom'); "
        "the match query for cancel (e.g. 'eggs', 'meeting')."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "cancel"],
                "description": "What to do.",
            },
            "trigger": {
                "type": "string",
                "enum": ["duration", "absolute"],
                "description": (
                    "Required for action=create. "
                    "'duration' = countdown, 'absolute' = calendar datetime."
                ),
            },
            "seconds": {
                "type": "integer",
                "description": (
                    "Countdown seconds — required for create+duration."
                ),
            },
            "fire_at": {
                "type": "string",
                "description": (
                    "ISO-8601 datetime — required for create+absolute, "
                    "e.g. '2026-05-15T15:30:00'."
                ),
            },
            "text": {
                "type": "string",
                "description": (
                    "For create: short reminder phrase the user hears when it fires. "
                    "For cancel: phrase to match against existing reminders."
                ),
            },
        },
        "required": ["action"],
    },
    # Timers/reminders are reversible (every action has a matching cancel),
    # so they sit in `low_write` — voice-ID is enough, no passphrase needed
    # for "set a timer" or "cancel the eggs reminder".
    risk="low_write",
)
async def set_reminder(
    *,
    ctx,  # AgentContext — injected by dispatch()
    action: str,
    trigger: str | None = None,
    seconds: int | None = None,
    fire_at: str | None = None,
    text: str | None = None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang
    if not cx.client_id:
        return ToolResult(text=t("reminders.no_client", lang), data={"error": "no_client_id"})

    # Emit a progress step matching the action so the UI shows
    # something specific instead of the stale "Launching tool" placeholder.
    step_for_action = {
        "list": "list_reminders",
        "cancel": "cancel_reminder",
        "create": "create_reminder",
    }.get(action, "tool")
    await cx.progress(step_for_action, None)

    # ── list ──────────────────────────────────────────────────────────────
    if action == "list":
        upcoming = await list_upcoming_reminders(cx.client_id)
        if not upcoming:
            return ToolResult(
                text=t("reminders.no_upcoming", lang),
                data={"reminders": []},
            )
        now = time.time()
        entries = [_fmt_reminder_for_list(r, now, lang) for r in upcoming]
        if len(entries) == 1:
            reply = t("reminders.list_one", lang, entry=entries[0])
        else:
            reply = t(
                "reminders.list_many", lang,
                count=len(entries), joined="; ".join(entries),
            )
        return ToolResult(text=reply, data={"reminders": upcoming})

    # ── cancel ────────────────────────────────────────────────────────────
    if action == "cancel":
        if not text:
            return ToolResult(
                text=t("reminders.cancel_no_text", lang), data={"error": "no_text"}
            )
        upcoming = await list_upcoming_reminders(cx.client_id)
        if not upcoming:
            return ToolResult(
                text=t("reminders.no_upcoming", lang), data={"reminders": []}
            )
        match, score = _best_match(text, upcoming)
        if match is None or score < 0.1:
            return ToolResult(
                text=t("reminders.not_found", lang),
                data={"query": text, "score": score},
            )
        reminder_id = match["id"]
        sched.cancel(reminder_id)
        await cancel_reminder_db(reminder_id, cx.client_id)
        log.info(
            "cancelled reminder %d (%r) for client %.8s… (score=%.2f)",
            reminder_id,
            match["push_text"][:40],
            cx.client_id,
            score,
        )
        return ToolResult(
            text=t("reminders.cancelled", lang, push_text=match["push_text"]),
            data={"reminder_id": reminder_id, "push_text": match["push_text"]},
        )

    # ── create ────────────────────────────────────────────────────────────
    if action == "create":
        now = time.time()

        if trigger == "duration":
            if not isinstance(seconds, int) or seconds < _MIN_SECONDS:
                return ToolResult(
                    text=t("reminders.bad_time", lang), data={"error": "bad_duration"}
                )
            secs = min(int(seconds), _MAX_DURATION_SECONDS)
            fire_ts = now + secs
            human_duration = format_duration_seconds(secs, lang)
            if text:
                push_text = t(
                    "reminders.timer_fired_with_label", lang,
                    duration=human_duration, label=text,
                )
                confirm = t(
                    "reminders.timer_set_with_label", lang,
                    duration=human_duration, label=text,
                )
            else:
                push_text = t("reminders.timer_fired", lang, duration=human_duration)
                confirm = t("reminders.timer_set", lang, duration=human_duration)

        elif trigger == "absolute":
            if not fire_at:
                return ToolResult(
                    text=t("reminders.bad_time", lang), data={"error": "no_fire_at"}
                )
            try:
                fire_ts = datetime.fromisoformat(fire_at).timestamp()
            except (ValueError, TypeError):
                return ToolResult(
                    text=t("reminders.bad_time", lang), data={"error": "bad_iso"}
                )
            if fire_ts <= now:
                return ToolResult(
                    text=t("reminders.past_time", lang), data={"error": "past_time"}
                )
            if fire_ts > now + _MAX_ABSOLUTE_HORIZON_S:
                return ToolResult(
                    text=t("reminders.too_far", lang), data={"error": "too_far"}
                )
            push_text = t(
                "reminders.reminder_fired", lang,
                text=text or t("reminders.default_fire_text", lang),
            )
            when_phrase = format_when(fire_ts, now, lang)
            if text:
                confirm = t(
                    "reminders.set_with_what", lang,
                    when=when_phrase, what=text,
                )
            else:
                confirm = t("reminders.set", lang, when=when_phrase)

        else:
            return ToolResult(
                text=t("reminders.bad_time", lang),
                data={"error": f"unknown_trigger:{trigger!r}"},
            )

        reminder_id = await add_reminder(cx.client_id, fire_ts, push_text)
        sched.schedule(reminder_id, cx.client_id, fire_ts, push_text)
        return ToolResult(
            text=confirm,
            data={
                "reminder_id": reminder_id,
                "fire_at": fire_ts,
                "trigger": trigger,
                "label": text,
            },
        )

    return ToolResult(
        text=t("reminders.bad_time", lang),
        data={"error": f"unknown_action:{action!r}"},
    )
