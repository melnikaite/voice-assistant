"""Datasets probe (architecture.md §14) — NOT part of the pytest suite.

Drives ``app.agent.run_agent`` against a live OpenAI-compatible LLM (LocalAI
on :1240 by default) with the real ``datasets`` tool on a throwaway SQLite
file.  Each turn is assembled the way ``pipeline.py`` does it: conversation
history, focus block, the focused category's dataset block, and the held
entry.  Other tools are stubbed so nothing leaves the machine.

Cases: create · add · missing → ask → answer · stats · forecast with too
little data · unknown field → schema proposal · no share → no data.

Usage (from orchestrator/):
    uv run python scripts/datasets_probe.py --runs 3 --langs ru en de --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--url", default="http://localhost:1240")
_ap.add_argument("--model", default="qwen3-vl-8b-instruct")
_ap.add_argument("--runs", type=int, default=3)
_ap.add_argument("--langs", nargs="+", default=["ru", "en", "de"])
_ap.add_argument("--cases", nargs="+", default=None)
_ap.add_argument("--json", default=None)
_ap.add_argument("-v", action="store_true")
ARGS = _ap.parse_args()

os.environ["LLM_URL"] = ARGS.url
os.environ["LLM_MODEL"] = ARGS.model
_DB = Path(tempfile.mkdtemp()) / "probe.db"
os.environ["DB_PATH"] = str(_DB)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import agent, llm_utils  # noqa: E402
from app.agent import AgentContext, ToolInvocation, run_agent  # noqa: E402
from app.conversation import Conversation  # noqa: E402
from app.storage import init_schema, list_pending_actions  # noqa: E402
from app.storage.categories import (  # noqa: E402
    create_category, get_last_category, set_last_category,
)
from app.storage.datasets import add_entry, list_entries, resolve_dataset  # noqa: E402
from app.storage.db import _conn, close_thread_conn  # noqa: E402
from app.tools.datasets import context_block, datasets as datasets_tool  # noqa: E402

llm_utils._fire_record_usage = lambda *a, **k: None
_real_execute = agent._execute_one


async def _execute(tc, ctx):  # real datasets tool, every other tool stubbed
    if tc.name in ("datasets", "list_pending"):
        return await _real_execute(tc, ctx)
    return ToolInvocation(name=tc.name, args=tc.args, text="OK.", data={}, terminal=True)


agent._execute_one = _execute

TODAY = dt.date.today()
Y = TODAY.year
L = {
    "ru": dict(cat="Коммуналка", ds="Счета", el="электричество", water="вода", gas="газ"),
    "en": dict(cat="Utilities", ds="Bills", el="electricity", water="water", gas="gas"),
    "de": dict(cat="Nebenkosten", ds="Rechnungen", el="Strom", water="Wasser", gas="Gas"),
}
FIELDS = [{"name": "month", "type": "month"}, {"name": "service", "type": "string"},
          {"name": "amount", "type": "number", "unit": "EUR"},
          {"name": "provider", "type": "string", "required": False}]
_prev_month = TODAY.replace(day=1) - dt.timedelta(days=1)
PM = f"{_prev_month:%Y-%m}"

Q = {
    "create": {
        "ru": "Начни вести мои счета за коммуналку: месяц, услуга, сумма в евро.",
        "en": "Start tracking my utility bills: month, service, amount in euros.",
        "de": "Fang an, meine Nebenkostenrechnungen zu erfassen: Monat, Leistung, Betrag in Euro.",
    },
    "add": {
        "ru": f"Запиши: за сентябрь {Y} электричество 84 евро, поставщик Stadtwerke.",
        "en": f"Log it: electricity for September {Y}, 84 euros, provider Stadtwerke.",
        "de": f"Trag ein: Strom für September {Y}, 84 Euro, Anbieter Stadtwerke.",
    },
    "missing": {
        "ru": ("Запиши счёт за воду, 31 евро.", "За прошлый месяц."),
        "en": ("Log the water bill, 31 euros.", "For last month."),
        "de": ("Trag die Wasserrechnung ein, 31 Euro.", "Für letzten Monat."),
    },
    "stats": {
        "ru": "Сколько я заплатил за электричество в этом году?",
        "en": "How much did I pay for electricity this year?",
        "de": "Wie viel habe ich dieses Jahr für Strom bezahlt?",
    },
    "forecast": {
        "ru": "Сколько я заплачу за электричество в следующем месяце?",
        "en": "How much will I pay for electricity next month?",
        "de": "Wie viel werde ich nächsten Monat für Strom zahlen?",
    },
    "newfield": {
        "ru": f"Запиши: газ за сентябрь {Y}, 40 евро, тариф ночной.",
        "en": f"Log it: gas for September {Y}, 40 euros, tariff night.",
        "de": f"Trag ein: Gas für September {Y}, 40 Euro, Tarif Nacht.",
    },
    "noshare": {
        "ru": "Сколько всего в счетах за коммуналку в этом году?",
        "en": "What's the total of the utility bills this year?",
        "de": "Wie hoch sind die Nebenkostenrechnungen dieses Jahr insgesamt?",
    },
}


# ── Harness ──────────────────────────────────────────────────────────────

def _reset_db() -> None:
    close_thread_conn()
    if _DB.exists():
        _DB.unlink()
    init_schema()
    for pid in (1, 2):
        _conn().execute(
            "INSERT INTO speaker_profiles (id, client_id, name, embedding, sample_count,"
            " created_at) VALUES (?, ?, ?, X'00', 1, ?)",
            (pid, f"probe{pid}", f"P{pid}", time.time()),
        )


async def _setup(lang: str, *, with_dataset: bool = True, profile_id: int = 1) -> int | None:
    v = L[lang]
    cid = await create_category(owner_profile_id=1, name=v["cat"])
    await set_last_category(1, cid)
    if profile_id != 1:
        other = await create_category(owner_profile_id=profile_id, name="Misc")
        await set_last_category(profile_id, other)
    if not with_dataset:
        return None
    ctx = AgentContext(client_id="probe1", profile_id=1, is_authenticated=True, user_lang=lang)
    r = await datasets_tool("create", ctx=ctx, name=v["ds"], category=v["cat"],
                            description="", fields=FIELDS)
    return r.data["dataset_id"]


async def turn(conv: Conversation, text: str, lang: str, *, profile_id: int = 1,
               authenticated: bool = False) -> tuple[str, list[ToolInvocation], int]:
    """One user turn, built like Pipeline.run."""
    fallback = {}
    if "list" not in conv.focus:
        last = await get_last_category(profile_id)
        if last:
            fallback["list"] = last["name"]
    focus = conv.focus_block(fallback)
    focused = conv.focus.get("list")
    block = await context_block(profile_id, focused.value if focused else fallback.get("list"),
                                conv.held)
    if block:
        focus = f"{focus}\n{block}" if focus else block
    ctx = AgentContext(client_id=f"probe{profile_id}", profile_id=profile_id,
                       is_authenticated=authenticated, user_lang=lang, held=conv.held,
                       transcript=text)
    t0 = time.monotonic()
    res = await run_agent(text, ctx=ctx, history=conv.messages(), focus_block=focus)
    ms = int((time.monotonic() - t0) * 1000)
    conv.record(text, res.response_text, res.invocations)
    if ARGS.v:
        print(f"    > {text}\n    {focus!r}")
        for inv in res.invocations:
            print(f"    · {inv.name} {json.dumps(inv.args, ensure_ascii=False)}")
        print(f"    < {res.response_text}  [{ms} ms]")
    return res.response_text, res.invocations, ms


def _ds_calls(invs, action=None):
    return [i for i in invs if i.name == "datasets" and (action is None or i.args.get("action") == action)]


# ── Cases: (lang) -> (passed, latencies, note) ───────────────────────────

async def case_create(lang):
    await _setup(lang, with_dataset=False)
    reply, invs, ms = await turn(Conversation(), Q["create"][lang], lang)
    pend = await list_pending_actions(profile_id=1, client_id="probe1")
    args = pend[0]["tool_args"] if pend else {}
    types = {f.get("type") for f in args.get("fields") or []}
    ok = (args.get("action") == "create" and len(args.get("fields") or []) >= 3
          and "number" in types and types & {"month", "date"})
    return bool(ok), [ms], json.dumps(args.get("fields"), ensure_ascii=False)[:160]


async def case_add(lang):
    ds = await _setup(lang)
    reply, invs, ms = await turn(Conversation(), Q["add"][lang], lang)
    rows = await list_entries(1, ds)
    d = rows[0]["data"] if rows else {}
    ok = (len(rows) == 1 and d.get("month") == f"{Y}-09" and d.get("amount") == 84
          and str(d.get("service", "")).lower()[:4] in (L[lang]["el"].lower()[:4], "elec"))
    return ok, [ms], json.dumps(d, ensure_ascii=False)


async def case_missing(lang):
    ds = await _setup(lang)
    conv = Conversation()
    q1, q2 = Q["missing"][lang]
    r1, i1, ms1 = await turn(conv, q1, lang)
    saved_early = bool(await list_entries(1, ds))
    asked = conv.held is not None
    r2, i2, ms2 = await turn(conv, q2, lang)
    rows = await list_entries(1, ds)
    d = rows[0]["data"] if rows else {}
    ok = (not saved_early and asked and len(rows) == 1 and d.get("month") == PM
          and d.get("amount") == 31)
    return ok, [ms1, ms2], f"asked={asked} reply1={r1[:60]!r} entry={json.dumps(d, ensure_ascii=False)}"


async def _seed_bills(lang, ds):
    v = L[lang]
    for m, svc, amt in [(1, v["el"], 80), (2, v["el"], 95.5), (3, v["el"], 70),
                        (2, v["water"], 30), (3, v["gas"], 44)]:
        await add_entry(1, ds, {"month": f"{Y}-{m:02d}", "service": svc, "amount": amt})
    await add_entry(1, ds, {"month": f"{Y - 1}-12", "service": v["el"], "amount": 99})


async def case_stats(lang):
    ds = await _setup(lang)
    await _seed_bills(lang, ds)
    reply, invs, ms = await turn(Conversation(), Q["stats"][lang], lang)
    calls = _ds_calls(invs, "stats")
    val = calls[-1].data.get("value") if calls and calls[-1].data else None
    ok = val == 245.5 and "245" in reply
    return ok, [ms], json.dumps(calls[-1].args if calls else None, ensure_ascii=False) + f" → {reply[:80]!r}"


async def case_forecast(lang):
    ds = await _setup(lang)
    await _seed_bills(lang, ds)
    reply, invs, ms = await turn(Conversation(), Q["forecast"][lang], lang)
    calls = _ds_calls(invs, "forecast")
    ok = bool(calls) and calls[-1].data and calls[-1].data.get("enough") is False
    return ok, [ms], reply[:100]


async def case_newfield(lang):
    ds = await _setup(lang)
    reply, invs, ms = await turn(Conversation(), Q["newfield"][lang], lang)
    rows = await list_entries(1, ds)
    pend = await list_pending_actions(profile_id=1, client_id="probe1")
    ext = [p for p in pend if p["tool_args"].get("action") == "extend"]
    ok = not rows and any(any(w in f.get("name", "").lower() for w in ("tari", "тариф"))
                          for p in ext for f in p["tool_args"].get("fields") or [])
    return ok, [ms], (json.dumps(ext[0]["tool_args"]["fields"], ensure_ascii=False) if ext
                      else f"rows={len(rows)} reply={reply[:80]!r}")


async def case_noshare(lang):
    ds = await _setup(lang, profile_id=2)
    await _seed_bills(lang, ds)
    reply, invs, ms = await turn(Conversation(), Q["noshare"][lang], lang, profile_id=2)
    leaked = any((i.data or {}).get("value") is not None or (i.data or {}).get("rows")
                 for i in invs) or any(n in reply for n in ("245", "339", "95.5"))
    ok = not leaked and await resolve_dataset(2, L[lang]["ds"]) is None
    return ok, [ms], reply[:100]


CASES = {"create": case_create, "add": case_add, "missing": case_missing,
         "stats": case_stats, "forecast": case_forecast, "newfield": case_newfield,
         "noshare": case_noshare}
DEFAULT_LANGS = {"ru": list(CASES), "en": ["create", "add", "missing", "stats"],
                 "de": ["create", "add", "missing", "stats"]}


async def block_cost(lang="ru") -> int:
    """Prompt-token delta of the datasets block, measured by the server."""
    _reset_db()
    await _setup(lang)
    block = await context_block(1, L[lang]["cat"])
    async with httpx.AsyncClient(timeout=60) as c:
        async def n(content):
            r = await c.post(f"{ARGS.url}/v1/chat/completions", json={
                "model": ARGS.model, "max_tokens": 1,
                "messages": [{"role": "user", "content": content}]})
            return r.json()["usage"]["prompt_tokens"]
        return await n(block + "\nПривет") - await n("Привет")


async def main() -> None:
    results = []
    for lang in ARGS.langs:
        for name in ARGS.cases or DEFAULT_LANGS.get(lang, []):
            for run in range(ARGS.runs):
                _reset_db()
                try:
                    ok, lat, note = await CASES[name](lang)
                except Exception as e:  # LocalAI restarts: count as a failure, keep going
                    ok, lat, note = False, [], f"{e.__class__.__name__}: {e}"
                results.append({"case": name, "lang": lang, "run": run, "ok": ok,
                                "ms": lat, "note": note})
                print(f"{name:9} {lang} #{run}: {'PASS' if ok else 'FAIL'} {lat} {note}", flush=True)
    print("\ncase       lang  pass  median ms/turn")
    keys = sorted({(r["case"], r["lang"]) for r in results},
                  key=lambda k: (list(CASES).index(k[0]), k[1]))
    for case, lang in keys:
        rs = [r for r in results if r["case"] == case and r["lang"] == lang]
        lat = [m for r in rs for m in r["ms"]]
        print(f"{case:10} {lang:4}  {sum(r['ok'] for r in rs)}/{len(rs)}  "
              f"{int(statistics.median(lat)) if lat else '-'}")
    try:
        print(f"\ndatasets block: +{await block_cost()} prompt tokens (ru, one dataset)")
    except Exception as e:
        print(f"\ndatasets block cost: n/a ({e})")
    if ARGS.json:
        Path(ARGS.json).write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
