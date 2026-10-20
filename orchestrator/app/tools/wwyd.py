"""
wwyd tools — predict how a household member would answer, and teach
their persona explicitly.  Both are ``tier="api"`` and hidden unless the
turn's ``AgentContext.services`` covers them (see ``wwyd_services``):

  * ``wwyd_remember`` needs ``"wwyd"``: the identified, single speaker
    has wwyd connected.  Claims land APPROVED — the owner said it.
  * ``how_would_answer`` needs ``"wwyd_ask"``: someone on this server has
    wwyd connected.  The asker never picks the audience: a non-owner is
    mapped through the owner's ``audience_map`` (Settings), the owner
    asking about themself names it.

``predict`` answers are sendable as is; when one comes with ``depends_on``,
the local LLM names the time it is about, the target's own Mac calendar
(times only) is checked in code, and only the one-line conclusion goes
back as ``context``.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

from .. import wwyd
from ..i18n import t
from ..storage import get_wwyd_connection, list_wwyd_connected_profiles
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

GUEST_KEY = "guest"
FREEBUSY_DAYS = 7
DEFAULT_CATEGORY = "communication/style"
KINDS = ["trait", "preference", "habit", "decision_rule", "style", "boundary"]

async def wwyd_services(profile_id: int | None, mixed: bool) -> frozenset[str]:
    """Services for ``AgentContext.services``; never for mixed turns."""
    if mixed:
        return frozenset()
    connected = await list_wwyd_connected_profiles()
    out: set[str] = set()
    if connected:
        out.add("wwyd_ask")
    if profile_id is not None and any(pid == profile_id for pid, _ in connected):
        out.add("wwyd")
    return frozenset(out)


_LATIN = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
})


def _match_name(spoken: str, profiles: list[tuple[int, str]]) -> tuple[int, str] | None:
    needle = spoken.strip().lower()
    if not needle:
        return None
    for pid, name in profiles:
        if name.lower() == needle:
            return pid, name
    # Inflected forms ("Маше" → "Маша") and the other script ("Анне" →
    # "Anna"): shared 3-char stem after transliteration.
    stem = needle.translate(_LATIN)[:3]
    hits = [(pid, name) for pid, name in profiles
            if len(needle) >= 3 and name.lower().translate(_LATIN)[:3] == stem]
    return min(hits, key=lambda h: len(h[1])) if hits else None


def _pick_audience(spoken: str | None, audiences: list[dict]) -> str | None:
    """The owner's spoken audience ("clients", "Familie") → a wwyd name;
    None means wwyd's default audience."""
    if not spoken:
        return None
    needle = spoken.strip().lower()
    for a in audiences:
        if a.get("name", "").lower() == needle:
            return a["name"]
    for a in audiences:
        hay = f"{a.get('name', '')} {a.get('description') or ''}".lower()
        if len(needle) >= 3 and needle[:4] in hay:
            return a["name"]
    return None


def _pick_category(spoken: str, paths: list[str]) -> str:
    """The model's category → an existing path of this persona."""
    needle = (spoken or "").strip().strip("/").lower()
    by_lower = {p.lower(): p for p in paths}
    if needle in by_lower:
        return by_lower[needle]
    # "meetings" → "work/meetings"; "work/x" → "work".
    for p in paths:
        if p.lower().rsplit("/", 1)[-1] == needle.rsplit("/", 1)[-1]:
            return p
    head = needle.split("/", 1)[0]
    if head in by_lower:
        return by_lower[head]
    if DEFAULT_CATEGORY in paths or not paths:
        return DEFAULT_CATEGORY
    return paths[0]


_FREEBUSY_SCRIPT = """
set out to ""
set t0 to current date
set t1 to t0 + ({days} * days)
tell application "Calendar"
  repeat with cal in calendars
    repeat with e in (every event of cal whose start date < t1 and end date > t0)
      set out to out & ((start date of e) as «class isot» as string) & "|" & ((end date of e) as «class isot» as string) & linefeed
    end repeat
  end repeat
end tell
return out
"""


def _busy_spans(stdout: str) -> list[tuple[dt.datetime, dt.datetime]]:
    spans: list[tuple[dt.datetime, dt.datetime]] = []
    for line in stdout.splitlines():
        try:
            a, b = line.strip().split("|")
            spans.append((dt.datetime.fromisoformat(a), dt.datetime.fromisoformat(b)))
        except ValueError:
            continue
    spans.sort()
    merged: list[tuple[dt.datetime, dt.datetime]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


async def calendar_busy(profile_id: int) -> list[tuple[dt.datetime, dt.datetime]] | None:
    """Busy spans of the profile's own Mac calendar — start/end only, never
    titles, attendees or places.  None when no Mac is reachable."""
    from .. import desktop_client
    from ..device_router import resolve_agent_id_for_profile

    agent_id = await resolve_agent_id_for_profile(profile_id, "macos_agent")
    if agent_id is None:
        return None
    try:
        res = await desktop_client.run_applescript(
            _FREEBUSY_SCRIPT.format(days=FREEBUSY_DAYS), timeout=30.0,
            agent_id=agent_id, category="calendar",
        )
    except Exception as exc:
        log.info("wwyd: free/busy unavailable: %s", exc.__class__.__name__)
        return None
    if res.get("exit") not in (0, None):
        return None
    return _busy_spans(res.get("stdout") or "")


# The 8B model reliably names the time but not whether a schedule covers it,
# so it only extracts the time and the overlap is checked in code.
WHEN_PROMPT = (
    "Find the time a condition is about.  You get today's date with the next days, a "
    "situation, and the condition a person's reply depends on.  If the condition depends "
    "on the person being free at some time, reply with one line:\n"
    "When: YYYY-MM-DD HH:MM  (a moment)  or  When: YYYY-MM-DD HH:MM-HH:MM  (a span; "
    "\"evening\" is 18:00-22:00, \"morning\" 08:00-12:00, \"afternoon\" 12:00-18:00)\n"
    "using the dates listed.  If it depends on anything else (weather, other people, "
    "money) or names no day, reply: When: NONE"
)
_WHEN = re.compile(r"When:\s*(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})(?:\s*-\s*(\d{1,2}:\d{2}))?\s*$")
_MOMENT = dt.timedelta(minutes=30)


def _day_label(day: dt.date, today: dt.date) -> str:
    rel = {0: "today ", 1: "tomorrow "}.get((day - today).days, "")
    return f"{rel}({day:%a %Y-%m-%d})" if rel else f"{day:%a %Y-%m-%d}"


def _conclude(spans: list[tuple[dt.datetime, dt.datetime]], start: dt.datetime,
              end: dt.datetime, now: dt.datetime) -> str:
    """Whether [start, end) is free, as a sentence about that window only."""
    free, cur = [], start
    for a, b in spans:
        if b <= cur or a >= end:
            continue
        if a > cur:
            free.append((cur, a))
        cur = max(cur, b)
    if cur < end:
        free.append((cur, end))
    window = f"{_day_label(start.date(), now.date())} {start:%H:%M}-{end:%H:%M}"
    if free == [(start, end)]:
        return f"{window} is free"
    if not free:
        return f"{window} is taken"
    parts = ", ".join(f"{a:%H:%M}-{b:%H:%M}" for a, b in free)
    return f"Within {window} only {parts} is free"


async def _calendar_context(profile_id: int, situation: str, depends_on: str) -> str | None:
    """A one-line conclusion from the owner's calendar for ``depends_on``;
    the schedule itself never leaves this machine."""
    from ..llm_utils import chat, extract_text

    spans = await calendar_busy(profile_id)
    if spans is None:
        return None
    now = dt.datetime.now().replace(second=0, microsecond=0)
    days = "\n".join(_day_label(now.date() + dt.timedelta(days=n), now.date())
                     for n in range(FREEBUSY_DAYS + 1))
    try:
        choice = await chat(
            [{"role": "system", "content": WHEN_PROMPT},
             {"role": "user", "content": f"Days:\n{days}\nSituation: {situation[:2000]}\n"
                                         f"Depends on: {depends_on}"}],
            temperature=0.0, max_tokens=40, tool_name="<wwyd_context>",
        )
    except Exception as exc:  # the first answer stands
        log.info("wwyd: context step failed: %s", exc.__class__.__name__)
        return None
    m = _WHEN.search(extract_text(choice["message"]).strip())
    if not m:
        return None
    try:
        start = dt.datetime.fromisoformat(f"{m[1]}T{int(m[2][:-3]):02d}:{m[2][-2:]}")
        end = (dt.datetime.fromisoformat(f"{m[1]}T{int(m[3][:-3]):02d}:{m[3][-2:]}")
               if m[3] else start + _MOMENT)
    except ValueError:
        return None
    # Outside what was read from the calendar the answer is unknown.
    if end <= start or start.date() < now.date() or end > now + dt.timedelta(days=FREEBUSY_DAYS):
        return None
    return _conclude(spans, start, end, now)


async def _resolve_target(person: str | None, speaker_id: int | None) -> tuple[int, str] | str:
    connected = await list_wwyd_connected_profiles()
    if person:
        hit = _match_name(person, connected)
        return hit if hit else "unknown"
    mine = [p for p in connected if p[0] == speaker_id]
    if mine:
        return mine[0]
    if len(connected) == 1:
        return connected[0]
    return "ambiguous"


@tool(
    name="how_would_answer",
    description=(
        "Predict how a person on this server would reply to a message or "
        "decide in a situation, from their wwyd persona (their own style, "
        "habits and rules).  Use for «как бы я ответил клиенту, который "
        "просит скидку», «what would Eugene say if I invite him on Friday», "
        "«würde Mama zusagen, wenn …», «draft my reply to my boss: …».  "
        "Omit `person` when the speaker asks about themself.  `audience` is "
        "who the reply is FOR (clients, family, colleagues) — only when the "
        "speaker asks about their own reply.  Do NOT use for facts about the "
        "world, for sending anything, or for what the assistant thinks."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "situation": {
                "type": "string",
                "description": "The message or situation, in the user's words.",
            },
            "person": {
                "type": "string",
                "description": "Whose answer to predict, by name; omit for the speaker.",
            },
            "audience": {
                "type": "string",
                "description": "Who the reply is for, e.g. 'clients', 'family'.",
            },
            "kind": {"type": "string", "enum": ["reply", "decision"]},
        },
        "required": ["situation"],
    },
    risk="read",
    tier="api",
    requires_token=("wwyd_ask",),
    focus={"person": "person"},
)
async def how_would_answer(
    situation: str,
    person: str | None = None,
    audience: str | None = None,
    kind: str = "reply",
    *,
    ctx=None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang
    target = await _resolve_target(person, cx.profile_id)
    if target == "unknown":
        return ToolResult(t("wwyd.unknown_person", lang, name=person or ""), {"error": "unknown_person"})
    if target == "ambiguous":
        return ToolResult(t("wwyd.which_person", lang), {"error": "ambiguous_person"})
    target_id, target_name = target  # type: ignore[misc]
    is_self = target_id == cx.profile_id and not getattr(ctx, "mixed", False)
    await cx.progress("wwyd", target_name)

    conn = await get_wwyd_connection(target_id)
    if conn is None:
        return ToolResult(t("tool.service_not_connected", lang), {"error": "service_not_connected"})
    if is_self:
        aud = None
        if audience:
            try:
                aud = _pick_audience(audience, await wwyd.list_audiences(target_id))
            except wwyd.WwydError:
                pass
    else:
        # A third party never chooses the audience it is answered as.
        key = f"profile:{cx.profile_id}" if cx.profile_id is not None else GUEST_KEY
        aud = conn["audience_map"].get(key)
    args: dict = {
        "situation": situation[:8000],
        # Unknown names fall back to wwyd's default audience.
        "audience": aud or "default",
        "kind": kind if kind in ("reply", "decision") else "reply",
        "include_private": is_self,
    }
    try:
        out = await wwyd.call_tool(target_id, "predict", args)
        # The answer is sendable as is; the calendar can only make it concrete.
        depends_on = str(out.get("depends_on") or "").strip() if out.get("type") == "answer" else ""
        if depends_on:
            await cx.progress("wwyd", "calendar")
            context = await _calendar_context(target_id, situation, depends_on)
            if context:
                out = await wwyd.call_tool(target_id, "predict", {**args, "context": context})
    except wwyd.WwydError as exc:
        log.info("wwyd: predict failed: %s", exc.code)
        return ToolResult(t("wwyd.unavailable", lang), {"error": exc.code})

    typ = out.get("type")
    data = {"type": typ, "person": target_name, "audience": aud,
            "used_claims": out.get("used_claims") or []}
    if typ == "answer":
        answer = str(out.get("answer") or "").strip()
        data["confidence"] = out.get("confidence")
        data["depends_on"] = out.get("depends_on")
        if is_self:
            return ToolResult(answer, data)
        # Third parties always hear that this is automated (wwyd P5).
        disclosure = str(out.get("disclosure") or "").strip()
        return ToolResult(f"{disclosure} {answer}".strip(), data)
    data["escalate"] = out.get("escalate")
    if out.get("escalate") in ("rate_limited", "spend_cap", "llm_unavailable"):
        return ToolResult(t("wwyd.unavailable", lang), data)
    return ToolResult(t("wwyd.ask_directly", lang, name=target_name), data)


def _remember_summary(args: dict) -> str:
    return str(args.get("text", ""))[:120]


@tool(
    name="wwyd_remember",
    description=(
        "Save how the SPEAKER themself talks, decides or behaves — their "
        "reply style, habits, decision rules, boundaries — to their wwyd "
        "persona.  Use for «запомни, что я отвечаю клиентам коротко и на вы», "
        "«remember that I never take calls before 10», «merk dir, dass ich "
        "freitags keine Meetings mache».  `text` is one statement about the "
        "speaker in the third person ('Answers clients briefly and formally').  "
        "Do NOT use for facts, lists or notes about other people — use "
        "`remember` for those."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "One statement, third person, ≤ 500 chars."},
            "kind": {"type": "string", "enum": KINDS},
            # Categories are per persona; validated against list_categories at call time.
            "category": {
                "type": "string",
                "description": "Category path, e.g. communication/style, work/meetings, "
                               "schedule/routine, food, family.",
            },
            "audience": {
                "type": "string",
                "description": "Who it applies to (e.g. 'clients'); omit for everyone.",
            },
        },
        "required": ["text", "kind", "category"],
    },
    risk="low_write",
    tier="api",
    requires_token=("wwyd",),
    summary=_remember_summary,
)
async def wwyd_remember(
    text: str,
    kind: str,
    category: str,
    audience: str | None = None,
    *,
    ctx=None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang
    if cx.profile_id is None:
        return ToolResult(t("tool.service_not_connected", lang), {"error": "no_profile"})
    await cx.progress("wwyd", None)
    claim: dict = {
        "kind": kind if kind in KINDS else "preference",
        "text": text.strip()[:500],
        "status": "approved",
    }
    try:
        claim["category"] = _pick_category(category, await wwyd.list_category_paths(cx.profile_id))
        if audience:
            name = _pick_audience(audience, await wwyd.list_audiences(cx.profile_id))
            if name:
                claim["audiences"] = [name]
        out = await wwyd.call_tool(cx.profile_id, "add_claims", {"claims": [claim]})
    except wwyd.WwydError as exc:
        log.info("wwyd: add_claims failed: %s", exc.code)
        return ToolResult(t("wwyd.unavailable", lang), {"error": exc.code})
    data = {"claim": claim, "result": out}
    if out.get("created"):
        return ToolResult(t("wwyd.remembered", lang, text=claim["text"]), data)
    if out.get("absorbed"):
        return ToolResult(t("wwyd.remember_absorbed", lang), data)
    reason = ((out.get("dropped") or [{}])[0]).get("reason", "invalid")
    return ToolResult(t("wwyd.remember_dropped", lang, reason=reason), {**data, "error": reason})
