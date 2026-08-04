"""
Scheduling tools — one LLM-visible tool per action, one implementation.

  set_timer        Countdown timer     ('set a timer for 10 minutes')
  set_reminder_at  Calendar entry      ('remind me at 15:30')
  list_reminders   Speak what's upcoming
  cancel_reminder  Cancel by fuzzy-matching its text

All four are thin wrappers over :func:`_run`, which holds the whole
implementation — the split exists for the LLM's benefit, not to fork the
logic.  Why it exists: JSON-Schema ``required`` is unconditional, so a
single multiplexed tool could only ever require ``action``, leaving the
field that actually matters (the duration, the datetime, the match text)
optional.  Measured consequence: models that decline to emit optional
fields called the tool with no duration at all and the timer errored out.
One action per tool makes ``required`` mean what it says.

``set_timer`` takes an ISO-8601 duration string (``PT1H30M``) rather than
an integer count of seconds: transcribing a spoken length into a fixed
format is reliable, deriving 5400 from "полтора часа" is not.  Same
normalisation Alexa/Lex apply to spoken durations.

Spoken durations and time anchors come from ``i18n.format_duration_seconds``
and ``i18n.format_when`` (Babel under the hood) — see ``i18n.py`` for
the locale wiring.
"""
from __future__ import annotations

import logging
import re
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

# Tokenise on word characters, NOT whitespace: the stored push_text comes out
# of an i18n template that punctuates it ("boil the eggs! 10 minutes has
# passed."), so a raw .split() yields "eggs!" and never matches a user saying
# "eggs" — which silently made cancel-by-label impossible for every labelled
# reminder.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets (case-insensitive, punctuation-blind)."""
    wa = set(_WORD_RE.findall(a.lower()))
    wb = set(_WORD_RE.findall(b.lower()))
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


# ── ISO-8601 durations ────────────────────────────────────────────────────

# Only the time part is accepted (plus whole days) — a countdown measured in
# months makes no sense and `_MAX_DURATION_SECONDS` would reject it anyway.
_ISO_DURATION_RE = re.compile(
    r"""^P(?:(?P<days>\d+(?:\.\d+)?)D)?
         (?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?
             (?:(?P<minutes>\d+(?:\.\d+)?)M)?
             (?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$""",
    re.VERBOSE,
)
_ISO_UNIT_SECONDS = {"days": 86400, "hours": 3600, "minutes": 60, "seconds": 1}


def _parse_iso_duration(value: str) -> int | None:
    """``'PT1H30M'`` → ``5400``.  None when the string isn't a duration.

    Tolerates lower case and surrounding whitespace, since that is what a
    model actually emits.  Returns None for ``'P'``/``'PT'`` (no components)
    so an empty duration can't silently become a zero-second timer.
    """
    m = _ISO_DURATION_RE.match(str(value).strip().upper())
    if not m or not any(m.groupdict().values()):
        return None
    total = sum(
        float(raw) * _ISO_UNIT_SECONDS[unit]
        for unit, raw in m.groupdict().items()
        if raw
    )
    return int(total)


# ── Shared implementation ─────────────────────────────────────────────────
#
# The four tools below differ only in what they advertise to the LLM; every
# one of them lands here.  Keep the branching in this function — do NOT
# inline per-action logic into the wrappers.

async def _run(
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


# ── LLM-visible tools ─────────────────────────────────────────────────────
#
# Timers/reminders are reversible (every one has a matching cancel), so the
# writing actions sit in `low_write` — voice-ID is enough, no passphrase for
# "set a timer" or "cancel the eggs reminder".  Listing is pure read.

_LABEL_DESC = (
    "Short phrase the user hears when it fires, e.g. 'eggs', 'call mom'.  "
    "Omit when the user named no subject."
)


@tool(
    name="set_timer",
    description=(
        "Start a countdown timer — 'set a timer for 10 minutes', 'напомни "
        "через полчаса', 'разбуди меня через 45 минут'.  For a specific "
        "clock time or date use `set_reminder_at` instead."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "duration": {
                "type": "string",
                "description": (
                    "How long to count down, as an ISO-8601 duration string.  "
                    "Transcribe the spoken length directly — do NOT convert it "
                    "to seconds yourself.  Examples: '20 минут' → 'PT20M'; "
                    "'half an hour' → 'PT30M'; 'полтора часа' → 'PT1H30M'; "
                    "'два с половиной часа' → 'PT2H30M'; '90 секунд' → 'PT90S'."
                ),
            },
            "text": {"type": "string", "description": _LABEL_DESC},
        },
        "required": ["duration"],
    },
    risk="low_write",
)
async def set_timer(*, ctx, duration: str, text: str | None = None) -> ToolResult:
    seconds = _parse_iso_duration(duration)
    if seconds is None:
        log.info("set_timer: unparseable duration %r", duration)
        return ToolResult(
            text=t("reminders.bad_time", unwrap_ctx(ctx).user_lang),
            data={"error": "bad_iso_duration", "duration": duration},
        )
    return await _run(
        ctx=ctx, action="create", trigger="duration", seconds=seconds, text=text
    )


@tool(
    name="set_reminder_at",
    description=(
        "Schedule a reminder for a specific clock time or date — 'remind me "
        "at 15:30', 'напомни завтра в 9 утра', 'meeting on Friday at noon'.  "
        "For a countdown from now use `set_timer` instead."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "fire_at": {
                "type": "string",
                "description": (
                    "When to fire, as an ISO-8601 local datetime, e.g. "
                    "'2026-05-15T15:30:00'.  Resolve 'tomorrow' / 'in the "
                    "morning' against the current local time given below."
                ),
            },
            "text": {"type": "string", "description": _LABEL_DESC},
        },
        "required": ["fire_at"],
    },
    risk="low_write",
)
async def set_reminder_at(*, ctx, fire_at: str, text: str | None = None) -> ToolResult:
    return await _run(
        ctx=ctx, action="create", trigger="absolute", fire_at=fire_at, text=text
    )


@tool(
    name="list_reminders",
    description=(
        "Read out the timers and reminders that are still pending — 'what "
        "timers do I have', 'какие у меня напоминания'."
    ),
    params_schema={"type": "object", "properties": {}, "required": []},
    risk="read",
)
async def list_reminders(*, ctx) -> ToolResult:
    return await _run(ctx=ctx, action="list")


@tool(
    name="cancel_reminder",
    description=(
        "Cancel a pending timer or reminder, matched by what it is about — "
        "'cancel the eggs timer', 'отмени напоминание про молоко'."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "What the reminder is about, as the user referred to it "
                    "('eggs', 'молоко', 'meeting').  Matched fuzzily against "
                    "pending reminders."
                ),
            },
        },
        "required": ["text"],
    },
    risk="low_write",
)
async def cancel_reminder(*, ctx, text: str) -> ToolResult:
    return await _run(ctx=ctx, action="cancel", text=text)
