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
the target's own Mac calendar (times only) is turned into a one-line
conclusion by the local LLM, and only that goes back as ``context``.
"""
from __future__ import annotations

import datetime as dt
import logging

from .. import wwyd
from ..i18n import t
from ..storage import get_wwyd_connection, list_wwyd_connected_profiles
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

GUEST_KEY = "guest"
FREEBUSY_DAYS = 7
_FACT_MAX = 2000
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


def _match_name(spoken: str, profiles: list[tuple[int, str]]) -> tuple[int, str] | None:
    needle = spoken.strip().lower()
    if not needle:
        return None
    for pid, name in profiles:
        if name.lower() == needle:
            return pid, name
    # Inflected forms ("Маше" → "Маша"): shared 3-char stem.
    hits = [(pid, name) for pid, name in profiles
            if len(needle) >= 3 and name.lower()[:3] == needle[:3]]
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


def _format_busy(stdout: str, now: dt.datetime) -> str:
    spans: list[tuple[dt.datetime, dt.datetime]] = []
    for line in stdout.splitlines():
        try:
            a, b = line.strip().split("|")
            spans.append((dt.datetime.fromisoformat(a), dt.datetime.fromisoformat(b)))
        except ValueError:
            continue
    spans.sort()
    merged: list[list[dt.datetime]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    end = now + dt.timedelta(days=FREEBUSY_DAYS)
    head = f"Busy times from {now:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} (local time); everything else is free:"
    if not merged:
        return f"{head} none."
    body = "; ".join(f"{a:%a %Y-%m-%d %H:%M}-{b:%H:%M}" if a.date() == b.date()
                     else f"{a:%a %Y-%m-%d %H:%M}-{b:%a %Y-%m-%d %H:%M}" for a, b in merged)
    return f"{head} {body}"[:_FACT_MAX]


async def calendar_free_busy(profile_id: int) -> str | None:
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
    return _format_busy(res.get("stdout") or "", dt.datetime.now().replace(second=0, microsecond=0))


CONTEXT_PROMPT = (
    "A person's real reply to a situation depends on something their persona cannot "
    "know.  You get the situation, what the reply depends on, and the person's busy "
    "times.  If the busy times settle it, answer with one short sentence that answers "
    "the condition itself (for example \"14:55-15:25 tomorrow is free\" or \"Tuesday "
    "15:00 is taken\"), in the language of the dependency, never listing the schedule "
    "or a wider free window.  If they do not, answer NONE."
)


async def _calendar_context(profile_id: int, situation: str, depends_on: str) -> str | None:
    """A one-line conclusion from the owner's calendar for ``depends_on``;
    the schedule itself never leaves this machine."""
    from ..llm_utils import chat, extract_text

    busy = await calendar_free_busy(profile_id)
    if not busy:
        return None
    try:
        choice = await chat(
            [{"role": "system", "content": CONTEXT_PROMPT},
             {"role": "user", "content": f"Situation: {situation[:2000]}\n"
                                         f"Depends on: {depends_on}\nBusy times: {busy}"}],
            temperature=0.0, max_tokens=80, tool_name="<wwyd_context>",
        )
    except Exception as exc:  # the first answer stands
        log.info("wwyd: context step failed: %s", exc.__class__.__name__)
        return None
    text = extract_text(choice["message"]).strip().strip('"')
    if not text or text.upper().startswith("NONE"):
        return None
    return text[:_FACT_MAX]


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
