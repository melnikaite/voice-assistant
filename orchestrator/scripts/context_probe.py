"""Referential follow-up probe (architecture.md §13) — NOT part of the pytest suite.

Drives ``app.agent.run_agent`` against a live OpenAI-compatible LLM (LocalAI on
:1240 by default).  Tool dispatch is stubbed: the probe records which tool the
model picked and with which arguments, and never touches the network, the DB or
a desktop-agent.

Each case = subject turn + N filler pairs (identical or topically varied) +
a referential follow-up.  Pass = the expected tool is called and its arguments
contain the resolved subject.

History *variants* are pluggable: a variant is an async function
``(turns, followup) -> (messages, followup, prep_ms, note)``.  ``turns`` keep the
transcript, the reply AND the tool calls of each past turn, so a variant decides
how much of that reaches the model.  Phase 2 adds its real implementation as one
more entry in ``VARIANTS`` and compares against ``baseline``.

Usage (from orchestrator/):
    uv run python scripts/context_probe.py --model qwen3-vl-8b-instruct \
        --n 4 8 16 --runs 3 --variants baseline focus replay --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

# Env must be set before importing app.* (llm_utils reads it at import time).
_ap = argparse.ArgumentParser()
_ap.add_argument("--url", default="http://localhost:1240")
_ap.add_argument("--model", default="qwen3-vl-8b-instruct")
_ap.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
_ap.add_argument("--fill", nargs="+", default=["identical", "varied"])
_ap.add_argument("--runs", type=int, default=3)
_ap.add_argument("--cases", nargs="+", default=None)
_ap.add_argument("--variants", nargs="+", default=["baseline"])
_ap.add_argument("--json", default=None, help="write raw results here")
ARGS = _ap.parse_args()

os.environ["LLM_URL"] = ARGS.url
os.environ["LLM_MODEL"] = ARGS.model
os.environ.setdefault("DB_PATH", str(Path(tempfile.mkdtemp()) / "probe.db"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import agent, llm_utils  # noqa: E402
from app.agent import AgentContext, ToolInvocation, run_agent  # noqa: E402
from app.tools.set_reminder import _parse_iso_duration, reminder_focus  # noqa: E402

llm_utils._fire_record_usage = lambda *a, **k: None  # no token_usage rows

_CALLS: list[ToolInvocation] = []


async def _fake_execute(tc, ctx):  # replaces agent._execute_one
    inv = ToolInvocation(name=tc.name, args=tc.args, text="OK.", data={}, terminal=True)
    _CALLS.append(inv)
    return inv


agent._execute_one = _fake_execute


# ── Conversation model ───────────────────────────────────────────────────
# A past turn keeps what production *could* keep: the transcript, the reply,
# and the tool calls the agent made (name, args, tool output).  Today ws.py
# keeps only the first two; the variants below decide how much is replayed.

@dataclass
class Turn:
    user: str
    reply: str
    calls: list[tuple] = field(default_factory=list)  # (tool, args, output[, data])


@dataclass
class Case:
    name: str
    subject: Turn                # the turn establishing the subject
    followup: str                # referential question
    check: Callable[[list[ToolInvocation], str], bool]


def _args_text(inv: ToolInvocation) -> str:
    return json.dumps(inv.args, ensure_ascii=False).lower()


def _called(tool: str | tuple, *needles: str, reply_ok: bool = False):
    """Pass iff `tool` was called with a needle in its args (or, with
    ``reply_ok``, a free-text reply names the subject — fine for pure facts)."""
    tools = (tool,) if isinstance(tool, str) else tool

    def check(calls: list[ToolInvocation], reply: str) -> bool:
        if reply_ok and not calls and any(n in reply.lower() for n in needles):
            return True
        return any(
            c.name in tools and any(n in _args_text(c) for n in needles)
            for c in calls
        )
    return check


def U(s): return {"role": "user", "content": s}


_DENTIST = (datetime.now() + timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0)


def _fires_at(calls: list[ToolInvocation], want: datetime) -> bool:
    """A created reminder fires within 5 min of ``want``."""
    for c in calls:
        try:
            if c.name == "set_reminder_at":
                got = datetime.fromisoformat(c.args["fire_at"])
                if got.tzinfo:
                    got = got.astimezone().replace(tzinfo=None)
            elif c.name == "set_timer":
                got = datetime.now() + timedelta(seconds=_parse_iso_duration(c.args["duration"]) or 0)
            else:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        if abs((got - want).total_seconds()) <= 300:
            return True
    return False


_MUC = "Мюнхен — столица Баварии, третий по величине город Германии, известен Октоберфестом."
_LIS = "Лиссабон — столица Португалии, стоит на семи холмах у устья Тежу."
CASES = [
    Case("weather",
         Turn("Расскажи коротко про Мюнхен.", _MUC,
              [("general_answer", {"question": "Расскажи коротко про Мюнхен"}, _MUC)]),
         "А какая там погода?",
         _called("weather", "мюнхен", "munich", "münchen", "munchen")),
    Case("shopping_list",
         Turn("Создай чек-лист «Покупки».", "Создал чек-лист «Покупки».",
              [("categories", {"action": "create", "name": "Покупки", "kind": "checklist"},
                "Создал чек-лист «Покупки».", {"focus": {"list": "Покупки"}})]),
         "Добавь туда молоко.",
         _called("items", "покуп")),
    Case("reminder",
         Turn("Когда у меня завтра стоматолог?", "Запись к стоматологу завтра в 15:00.",
              [("list_reminders", {}, "Завтра 15:00 — стоматолог.",
                {"focus": {"reminder": reminder_focus("«стоматолог» завтра в 15:00", _DENTIST.timestamp())}})]),
         "Напомни мне об этом за час.",
         lambda calls, reply: _called(("set_reminder_at", "set_timer"), "стомат", "dentist", "зуб")(calls, reply)
         and _fires_at(calls, _DENTIST - timedelta(hours=1))),
    Case("population",
         Turn("Расскажи коротко про Лиссабон.", _LIS,
              [("general_answer", {"question": "Расскажи коротко про Лиссабон"}, _LIS)]),
         "А сколько там живёт людей?",
         _called(("web_search", "general_answer"), "лиссабон", "lisbon", "lisboa",
                 reply_ok=True)),
]


def _ga(q, a): return Turn(q, a, [("general_answer", {"question": q}, a)])


IDENTICAL = Turn("Сколько будет два плюс два?", "Четыре.",
                 [("calculator", {"expression": "2+2"}, "4")])
VARIED = [
    Turn("Сколько будет семь умножить на восемь?", "Пятьдесят шесть.",
         [("calculator", {"expression": "7*8"}, "56")]),
    Turn("Переведи «доброе утро» на немецкий.", "Guten Morgen.",
         [("translate", {"text": "доброе утро", "target_lang": "de"}, "Guten Morgen.")]),
    _ga("Кто написал «Войну и мир»?", "Лев Толстой."),
    Turn("Поставь таймер на пять минут.", "Таймер на пять минут запущен.",
         [("set_timer", {"duration": "PT5M", "text": "таймер"}, "Таймер на пять минут запущен.")]),
    _ga("Какая самая высокая гора в мире?", "Эверест, 8849 метров."),
    _ga("Сколько калорий в банане?", "Около ста калорий."),
    _ga("Как называется столица Канады?", "Оттава."),
    _ga("Что такое фотосинтез, в двух словах?", "Растения превращают свет, воду и CO2 в сахар и кислород."),
    _ga("Посоветуй фильм на вечер.", "Попробуй «Интерстеллар»."),
    _ga("Сколько дней в високосном году?", "Триста шестьдесят шесть."),
    Turn("Как будет «кошка» по-английски?", "Cat.",
         [("translate", {"text": "кошка", "target_lang": "en"}, "Cat.")]),
    _ga("Когда началась Вторая мировая война?", "Первого сентября 1939 года."),
    Turn("Сколько будет корень из ста сорока четырёх?", "Двенадцать.",
         [("calculator", {"expression": "sqrt(144)"}, "12")]),
    _ga("Кто изобрёл телефон?", "Обычно называют Александра Белла."),
    _ga("Какой океан самый большой?", "Тихий."),
    _ga("Сколько ног у паука?", "Восемь."),
]


def build_turns(case: Case, n: int, fill: str) -> list[Turn]:
    return [case.subject] + [
        IDENTICAL if fill == "identical" else VARIED[i % len(VARIED)] for i in range(n)
    ]


# ── Renderers: turns → OpenAI messages ───────────────────────────────────

def render_text(turns: list[Turn]) -> list[dict]:
    """What ws.py stores today: transcript + final reply only."""
    out: list[dict] = []
    for t_ in turns:
        out += [U(t_.user), {"role": "assistant", "content": t_.reply}]
    return out


def render_replay(turns: list[Turn]) -> list[dict]:
    """(v) Replay tool calls as assistant tool_calls + tool messages.

    The tool output is the reply for terminal tools, so no separate trailing
    assistant text is added — mirrors the shape run_agent itself builds."""
    out: list[dict] = []
    for i, t_ in enumerate(turns):
        out.append(U(t_.user))
        if not t_.calls:
            out.append({"role": "assistant", "content": t_.reply})
            continue
        tcs = [{"id": f"h{i}_{j}", "type": "function",
                "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
               for j, (n, a, *_) in enumerate(t_.calls)]
        out.append({"role": "assistant", "content": "", "tool_calls": tcs})
        for j, (_, _, o, *_) in enumerate(t_.calls):
            out.append({"role": "tool", "tool_call_id": f"h{i}_{j}", "content": o})
        out.append({"role": "assistant", "content": t_.reply})
    return out


# ── Focus state (iv): deterministic, no LLM call ─────────────────────────
# slot ← (tool, arg).  Later calls overwrite earlier ones per slot; each slot
# remembers how many turns ago it was set.  Phase 2's real version would live
# next to the history and be updated from AgentResult.invocations.

FOCUS_SLOTS: dict[tuple[str, str], str] = {
    ("weather", "location"): "location",
    ("categories", "name"): "list",
    ("items", "category"): "list",
    ("set_reminder_at", "text"): "reminder",
    ("set_timer", "text"): "reminder",
    ("cancel_reminder", "text"): "reminder",
    ("summarize_url", "url"): "url",
    ("general_answer", "question"): "topic",
    ("web_search", "query"): "topic",
}


def build_focus(turns: list[Turn]) -> str:
    slots: dict[str, tuple[str, str, int]] = {}
    for age, t_ in enumerate(reversed(turns), start=1):
        for name, args, *_ in t_.calls:
            for arg, val in args.items():
                slot = FOCUS_SLOTS.get((name, arg))
                if slot and slot not in slots and isinstance(val, str) and val:
                    slots[slot] = (val, name, age)
    if not slots:
        return ""
    parts = [f"{s}={v!r} ({tool}, {age} turn{'s' if age > 1 else ''} ago)"
             for s, (v, tool, age) in sorted(slots.items(), key=lambda kv: kv[1][2])]
    return ("[Conversation focus — resolve 'there/it/this/туда/там/об этом' "
            "against these unless the message names something else: "
            + "; ".join(parts) + "]")


# ── Variants ─────────────────────────────────────────────────────────────
# (turns, followup) -> (messages, followup, prep_ms, note)
Variant = Callable[[list[Turn], str], Awaitable[tuple[list[dict], str, int, str]]]


def _transcript(msgs: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs if m.get("content"))


SUMMARY_PROMPT = (
    "Summarise this earlier part of a conversation between a user and a voice "
    "assistant in at most 3 short sentences, in the conversation's language. "
    "Keep every named entity (places, people, lists, folders, dates, times) "
    "that later questions might refer to; drop small talk."
)


async def _summarise(turns: list[Turn], k: int) -> tuple[list[dict], int, str]:
    keep, older = turns[len(turns) - k:], turns[: len(turns) - k]
    if not older:
        return render_text(turns), 0, ""
    t0 = time.monotonic()
    choice = await llm_utils.chat(
        [{"role": "system", "content": SUMMARY_PROMPT},
         {"role": "user", "content": _transcript(render_text(older))}],
        temperature=0.0, max_tokens=200,
    )
    text = llm_utils.extract_text(choice["message"]).strip()
    ms = int((time.monotonic() - t0) * 1000)
    msgs = [U(f"(Summary of our earlier conversation: {text})"),
            {"role": "assistant", "content": "OK."}] + render_text(keep)
    return msgs, ms, text


REWRITE_PROMPT = (
    "Rewrite the user's LAST message so it is self-contained: replace pronouns "
    "and words like 'there', 'it', 'this', 'туда', 'там', 'об этом' with what "
    "they refer to in the conversation.  Keep the language and intent.  If it "
    "is already self-contained, return it unchanged.  Output only the rewritten "
    "message."
)
# Cheap gate for (vi): only pay the extra LLM hop when the message looks referential.
_REFERENTIAL = re.compile(
    r"\b(там|туда|оттуда|тут|сюда|об этом|про это|это|его|её|ее|их|него|неё|"
    r"there|it|this|that|them|dort|dorthin|das|es)\b", re.I)


async def _rewrite(turns: list[Turn], followup: str, focus: str = "") -> tuple[str, int]:
    if not _REFERENTIAL.search(followup):
        return followup, 0
    t0 = time.monotonic()
    convo = _transcript(render_text(turns[-10:]))  # same 10-turn window as ws.py
    if focus:
        convo = focus + "\n" + convo
    choice = await llm_utils.chat(
        [{"role": "system", "content": REWRITE_PROMPT},
         {"role": "user", "content": convo + "\nuser (LAST): " + followup}],
        temperature=0.0, max_tokens=120,
    )
    text = llm_utils.extract_text(choice["message"]).strip() or followup
    return text, int((time.monotonic() - t0) * 1000)


_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZА-ЯЁ][\w-]+|«[^»]+»|\d{1,2}:\d{2}")


def _with_focus(followup: str, turns: list[Turn]) -> str:
    f = build_focus(turns)
    return f"{f}\n{followup}" if f else followup


async def v_baseline(turns, q):
    return render_text(turns), q, 0, ""


async def v_drop_filler(turns, q):
    kept = [t_ for t_ in turns[:-1] if _ENTITY.search(t_.user) or _ENTITY.search(t_.reply)]
    return render_text(kept + turns[-1:]), q, 0, ""


def v_summary(k):
    async def f(turns, q):
        msgs, ms, note = await _summarise(turns, k)
        return msgs, q, ms, note
    return f


async def v_focus(turns, q):
    return render_text(turns), _with_focus(q, turns), 0, build_focus(turns)


def v_focus_summary(k):
    async def f(turns, q):
        msgs, ms, note = await _summarise(turns, k)
        return msgs, _with_focus(q, turns), ms, note
    return f


async def v_replay(turns, q):
    return render_replay(turns), q, 0, ""


async def v_replay_focus(turns, q):
    return render_replay(turns), _with_focus(q, turns), 0, ""


async def v_rewrite(turns, q):
    nq, ms = await _rewrite(turns, q)
    return render_text(turns), nq, ms, nq


async def v_rewrite_focus(turns, q):
    nq, ms = await _rewrite(turns, q, build_focus(turns))
    return render_text(turns), nq, ms, nq


async def v_phase2(turns, q):
    """The shipped mechanism: app.conversation.Conversation, summary awaited
    between turns (it runs in the background in production)."""
    from app.conversation import Conversation

    conv = Conversation()
    fold_ms = 0
    for t_ in turns:
        invs = [ToolInvocation(name=c[0], args=c[1], text=c[2],
                               data=c[3] if len(c) > 3 else None, terminal=True)
                for c in t_.calls]
        conv.record(t_.user, t_.reply, invs)
        t0 = time.monotonic()
        await conv.wait_summary()
        fold_ms = max(fold_ms, int((time.monotonic() - t0) * 1000))
    focus = conv.focus_block()
    # prep_ms = slowest background fold (off the voice path), not added latency.
    return conv.messages(), (f"{focus}\n{q}" if focus else q), fold_ms, conv.summary


def _capped(v: Variant, max_turns: int = 10) -> Variant:
    """Apply production's MAX_HISTORY_TURNS window before the variant runs."""
    async def f(turns, q):
        return await v(turns[-max_turns:], q)
    return f


VARIANTS: dict[str, Variant] = {
    "baseline": v_baseline,
    "baseline_cap10": _capped(v_baseline),
    "drop_filler": v_drop_filler,
    "summary_k2": v_summary(2),
    "summary_k4": v_summary(4),
    "focus": v_focus,
    "focus_cap10": _capped(v_focus),
    "focus_summary_k4": v_focus_summary(4),
    "replay": v_replay,
    "replay_cap10": _capped(v_replay),
    "replay_focus": v_replay_focus,
    "rewrite": v_rewrite,
    "rewrite_focus": v_rewrite_focus,
    "phase2": v_phase2,
}


# ── Runner ───────────────────────────────────────────────────────────────

CTX = AgentContext(client_id="probe", profile_id=1, is_authenticated=True, user_lang="ru")


# A reply that says an action happened although no tool was called.
_CLAIM = re.compile(
    r"(добавил|добавлен|установил|установлен|создал|сохранил|запомнил|напомню|"
    r"added|saved|created|\bset\b|will remind)", re.I)


async def run_one(case: Case, n: int, fill: str, vname: str) -> dict:
    turns = build_turns(case, n, fill)
    msgs, q, prep_ms, note = await VARIANTS[vname](turns, case.followup)
    _CALLS.clear()
    t0 = time.monotonic()
    res = await run_agent(q, ctx=CTX, history=msgs)
    ms = int((time.monotonic() - t0) * 1000)
    return {
        "model": ARGS.model, "case": case.name, "n": n, "fill": fill,
        "variant": vname, "pass": case.check(_CALLS, res.response_text),
        "agent_ms": ms, "prep_ms": prep_ms, "question": q, "note": note,
        "calls": [(c.name, c.args) for c in _CALLS],
        "reply": None if _CALLS else res.response_text[:160],
        "claimed": not _CALLS and bool(_CLAIM.search(res.response_text)),
    }


async def main() -> None:
    cases = [c for c in CASES if not ARGS.cases or c.name in ARGS.cases]
    rows = []
    for vname in ARGS.variants:
        for case in cases:
            for fill in ARGS.fill:
                for n in ARGS.n:
                    batch = [await run_one(case, n, fill, vname) for _ in range(ARGS.runs)]
                    rows.extend(batch)
                    ok = sum(r["pass"] for r in batch)
                    print(
                        f"{ARGS.model[:12]:12} {vname:16} {case.name:13} {fill:9} N={n:<3} "
                        f"pass {ok}/{len(batch)}  agent {statistics.median(r['agent_ms'] for r in batch):>5} ms"
                        f"  prep {statistics.median(r['prep_ms'] for r in batch):>5} ms"
                        f"  claimed {sum(r['claimed'] for r in batch)}"
                        f"  {batch[-1]['calls'] or batch[-1]['reply']!r:.100}",
                        flush=True,
                    )
                    if ARGS.json:  # incremental, so a killed run keeps its data
                        Path(ARGS.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
