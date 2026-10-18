"""LLM replacement bench for the voice-assistant, rebuilt from docs/architecture.md "Models evaluated".

Drives the production ``run_agent`` (production SYSTEM_PROMPT + 34-tool catalog, device_kind=macos_agent)
against an OpenAI-compatible endpoint.  Tool dispatch is stubbed (no network/DB/desktop side effects),
except general_answer, whose real handler is a pure LLM call — needed for the prose-turn timing.

Suites: select (20 intents x ru/de/en), args (7), e2e (7 run_agent turns), screen (8 read + 8 click
+ 3 missing-control), latency (TTFT no catalog, prose turn).

    python bench_llm.py --url http://127.0.0.1:18080 --model NAME --runs 3 --json out.json

Run each model on its own llama-server (32K context, ``--jinja``) rather than through LocalAI, so a
candidate never sits resident next to the production model.  Gemma 4 needs ``--reasoning off``, and
Gemma 4 12B also ``--swa-full`` (docs/architecture.md, "Models evaluated").  ``screen.png`` /
``screen.json`` come from ``make_screen.py`` (needs Pillow).
"""
from __future__ import annotations

import argparse, asyncio, base64, datetime, json, os, re, statistics, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--suites", nargs="+", default=["select", "args", "e2e", "screen", "latency"])
ap.add_argument("--json", required=True)
A = ap.parse_args()

os.environ.update(LLM_URL=A.url, LLM_MODEL=A.model, LLM_MAX_TOKENS="4096", LLM_TIMEOUT=os.environ.get("LLM_TIMEOUT", "300"),
                  DB_PATH=str(Path(tempfile.mkdtemp()) / "bench.db"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # orchestrator/
from app import agent, llm_utils  # noqa: E402
from app.agent import AgentContext, ToolInvocation, run_agent  # noqa: E402
from app.tools import TOOL_REGISTRY  # noqa: E402

llm_utils._fire_record_usage = lambda *a, **k: None
_REAL_EXEC = agent._execute_one
CALLS: list[ToolInvocation] = []
STUB: dict[str, tuple[str, dict]] = {}  # per-case canned outputs: tool -> (text, data)


async def fake_exec(tc, ctx):
    if tc.name == "general_answer" and "general_answer" not in STUB:
        inv = await _REAL_EXEC(tc, ctx)
    else:
        text, data = STUB.get(tc.name, ("OK.", {}))
        term = bool(TOOL_REGISTRY.get(tc.name, {}).get("terminal", True))
        inv = ToolInvocation(name=tc.name, args=tc.args, text=text, data=data, terminal=term)
    CALLS.append(inv)
    return inv

agent._execute_one = fake_exec


def ctx(lang):
    return AgentContext(client_id="bench", profile_id=1, is_authenticated=True, user_lang=lang,
                        device_kind="macos_agent")


async def turn(text, lang, history=None):
    CALLS.clear()
    t0 = time.perf_counter()
    r = await run_agent(text, ctx=ctx(lang), history=history or [])
    return r, list(CALLS), (time.perf_counter() - t0) * 1000


# ── 1. Tool selection: 20 intents in ru/de/en; value = acceptable tool names ──
SEL = [
    ({"weather"}, "Какая сейчас погода в Мюнхене?", "Wie ist das Wetter gerade in München?", "What's the weather in Munich right now?"),
    ({"set_timer"}, "Поставь таймер на десять минут.", "Stell einen Timer auf zehn Minuten.", "Set a timer for ten minutes."),
    ({"set_reminder_at"}, "Напомни мне завтра в девять утра позвонить маме.", "Erinnere mich morgen um neun Uhr, Mama anzurufen.", "Remind me tomorrow at nine a.m. to call mom."),
    ({"list_reminders"}, "Какие у меня есть таймеры?", "Welche Timer habe ich gerade?", "What timers do I have?"),
    ({"cancel_reminder"}, "Отмени таймер для яиц.", "Lösch den Eier-Timer.", "Cancel the egg timer."),
    ({"calculator"}, "Сколько будет триста сорок семь умножить на двенадцать?", "Was ist dreihundertsiebenundvierzig mal zwölf?", "What is three hundred forty-seven times twelve?"),
    ({"calculator"}, "Сколько будет сто евро в долларах?", "Wie viel sind hundert Euro in Dollar?", "How much is a hundred euros in dollars?"),
    ({"translate"}, "Переведи на немецкий: я опаздываю на десять минут.", "Übersetz ins Englische: ich komme zehn Minuten später.", "Translate into German: I'm running ten minutes late."),
    ({"web_search"}, "Кто выиграл вчерашний матч Баварии?", "Wer hat gestern das Bayern-Spiel gewonnen?", "Who won yesterday's Bayern match?"),
    ({"news_briefing"}, "Расскажи новости за сегодня.", "Was gibt's heute für Nachrichten?", "Give me today's news briefing."),
    ({"inbox_list"}, "Есть для меня голосовые сообщения?", "Habe ich neue Sprachnachrichten?", "Do I have any voicemail?"),
    ({"remember"}, "Запомни, что у меня аллергия на орехи.", "Merk dir, dass ich gegen Nüsse allergisch bin.", "Remember that I'm allergic to nuts."),
    ({"read_memory"}, "Что ты обо мне помнишь?", "Was weißt du über mich?", "What do you remember about me?"),
    ({"my_history"}, "О чём мы говорили на прошлой неделе про отпуск?", "Worüber haben wir letzte Woche zum Urlaub gesprochen?", "What did we talk about last week regarding the vacation?"),
    ({"update_settings"}, "Говори со мной на немецком.", "Sprich ab jetzt Russisch mit mir.", "From now on speak German with me."),
    ({"summarize_url"}, "Перескажи это видео: https://www.youtube.com/watch?v=dQw4w9WgXcQ", "Fass dieses Video zusammen: https://www.youtube.com/watch?v=dQw4w9WgXcQ", "Summarize this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
    ({"navigate_browser"}, "Открой в браузере ютуб.", "Öffne YouTube im Browser.", "Open YouTube in the browser."),
    ({"look_at_screen"}, "Что сейчас у меня на экране?", "Was ist gerade auf meinem Bildschirm?", "What's on my screen right now?"),
    ({"desktop", "computer_use"}, "Сделай громкость на компьютере потише.", "Mach die Lautstärke am Computer leiser.", "Turn the volume down on my computer."),
    ({"list_browser_tabs"}, "Какие вкладки у меня открыты в хроме?", "Welche Tabs sind in Chrome offen?", "Which tabs are open in Chrome?"),
]
LANGS = ("ru", "de", "en")


async def suite_select():
    out = []
    for exp, *texts in SEL:
        for lang, t in zip(LANGS, texts):
            for k in range(A.runs):
                r, calls, ms = await turn(t, lang)
                first = calls[0].name if calls else None
                out.append(dict(lang=lang, text=t, expect=sorted(exp), got=first, ok=first in exp, ms=ms,
                                free_text=None if calls else r.response_text[:120]))
    return out


# ── 2. Arguments (7) ──
TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()


def _a(c, k):
    return json.dumps(c.args.get(k, ""), ensure_ascii=False).lower()

ARGS = [
    ("Поставь таймер на полтора часа.", "ru", "set_timer", lambda c: c.args.get("duration", "").upper() in ("PT1H30M", "PT90M")),
    ("Поставь таймер на два с половиной часа.", "ru", "set_timer", lambda c: c.args.get("duration", "").upper() in ("PT2H30M", "PT150M")),
    ("Напомни завтра в девять утра выкинуть мусор.", "ru", "set_reminder_at", lambda c: TOMORROW in _a(c, "fire_at") and "09:00" in _a(c, "fire_at")),
    ("Какая погода в Гамбурге?", "ru", "weather", lambda c: any(x in _a(c, "location") for x in ("гамбург", "hamburg"))),
    ("Переведи на немецкий: доброе утро.", "ru", "translate", lambda c: _a(c, "target_lang").strip('"') in ("de", "german", "немецкий", "deutsch")),
    ("Сколько будет семнадцать умножить на двадцать три?", "ru", "calculator", lambda c: "17" in json.dumps(c.args) and "23" in json.dumps(c.args)),
    ("Отмени напоминание про стоматолога.", "ru", "cancel_reminder", lambda c: "стоматолог" in _a(c, "text") or "dentist" in _a(c, "text")),
]


async def suite_args():
    out = []
    for t, lang, tool, chk in ARGS:
        for k in range(A.runs):
            r, calls, ms = await turn(t, lang)
            c = next((c for c in calls if c.name == tool), None)
            out.append(dict(text=t, tool=tool, ok=bool(c and chk(c)), args=c.args if c else None,
                            got=[x.name for x in calls], ms=ms))
    return out


# ── 3. End-to-end run_agent (7): stubbed outputs, judged on the final reply / chain ──
E2E = [
    ("general_unknown_chain", "Кто сейчас тренер сборной Германии по футболу?", "ru",
     {"general_answer": ("Не знаю.", {"unknown": True}), "web_search": ("Тренер сборной Германии — Юлиан Нагельсманн.", {})},
     lambda calls, rep: any(c.name == "web_search" for c in calls)),
    ("weather_reply", "Нужен ли сегодня зонт в Мюнхене?", "ru",
     {"weather": ("В Мюнхене +12, дождь, 90% вероятность осадков.", {})},
     lambda calls, rep: any(c.name == "weather" for c in calls) and bool(rep.strip())),
    ("timer_eggs", "Поставь таймер на восемь минут для яиц.", "ru", {"set_timer": ("Таймер на 8 минут запущен.", {})},
     lambda calls, rep: any(c.name == "set_timer" and c.args.get("duration", "").upper() == "PT8M" for c in calls)),
    ("chatty_no_free_text", "Привет! Как у тебя дела?", "ru", {},
     lambda calls, rep: bool(calls)),
    ("inbox_then_read", "Прочитай последнее голосовое сообщение.", "ru",
     {"inbox_list": ("1 сообщение: id 42 от Анны, 10:15.", {"messages": [{"id": 42, "from": "Анна"}]}),
      "inbox_read": ("(воспроизведено)", {})},
     lambda calls, rep: any(c.name == "inbox_read" and str(c.args.get("message_id")) == "42" for c in calls)
     or (any(c.name == "inbox_list" for c in calls) and "42" in rep)),
    ("de_weather", "Brauche ich heute in Berlin eine Jacke?", "de", {"weather": ("Berlin: 6 Grad, windig.", {})},
     lambda calls, rep: any(c.name == "weather" and "berlin" in json.dumps(c.args).lower() for c in calls)),
    ("en_reminder", "Remind me at 18:30 today to water the plants.", "en", {"set_reminder_at": ("Reminder set.", {})},
     lambda calls, rep: any(c.name == "set_reminder_at" and "18:30" in json.dumps(c.args) for c in calls)),
]


async def suite_e2e():
    out = []
    for name, t, lang, stub, chk in E2E:
        for k in range(A.runs):
            STUB.clear(); STUB.update(stub)
            r, calls, ms = await turn(t, lang)
            out.append(dict(case=name, ok=bool(chk(calls, r.response_text or "")), calls=[(c.name, c.args) for c in calls],
                            reply=(r.response_text or "")[:160], ms=ms))
    STUB.clear()
    return out


# ── 4. Screen: read text (8), click error (8), missing control refusal (3) — direct vision calls ──
HD = Path(__file__).parent
SCR = json.loads((HD / "screen.json").read_text())
IMG = "data:image/png;base64," + base64.b64encode((HD / "screen.png").read_bytes()).decode()
COLORS = ["blue", "green", "red", "purple", "orange", "teal", "pink", "grey"]


async def vision(prompt, max_tokens=200):
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG}}, {"type": "text", "text": prompt}]}]
    t0 = time.perf_counter()
    resp = await llm_utils.chat(msgs, temperature=0.0, max_tokens=max_tokens)
    return llm_utils.extract_text(resp.get("message", resp)) or "", (time.perf_counter() - t0) * 1000


async def suite_screen():
    out = []
    for i, c in enumerate(SCR["controls"]):
        for k in range(A.runs):
            txt, ms = await vision(f"Read the exact text written on the {COLORS[i]} button. Reply with that text only.")
            ok = c["label"].lower() == txt.strip().strip('".\'«»').lower()
            out.append(dict(kind="read", label=c["label"], ok=ok, reply=txt[:80], ms=ms))
        for k in range(A.runs):
            txt, ms = await vision(f'Where should I click to press the "{c["label"]}" button? Reply only JSON '
                                   '{"x": <int>, "y": <int>} in coordinates normalized to 0-1000.')
            m = re.search(r'"?x"?\s*[:=]\s*(-?\d+(?:\.\d+)?).*?"?y"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', txt, re.S)
            err = None
            if m:
                x, y = float(m.group(1)) * SCR["W"] / 1000, float(m.group(2)) * SCR["H"] / 1000
                err = ((x - c["cx"]) ** 2 + (y - c["cy"]) ** 2) ** 0.5
            out.append(dict(kind="click", label=c["label"], err_px=err, reply=txt[:80], ms=ms))
    for missing in ("Delete account", "Abmelden", "Корзина"):
        for k in range(A.runs):
            txt, ms = await vision(f'Where should I click to press the "{missing}" button? Reply only JSON '
                                   '{"x": <int>, "y": <int>} in coordinates normalized to 0-1000, or exactly NOT_VISIBLE if '
                                   'there is no such control on the screen.')
            out.append(dict(kind="missing", label=missing, ok="NOT_VISIBLE" in txt.upper(), reply=txt[:80], ms=ms))
    return out


# ── 5. Latency: TTFT without catalog (voice path), prose-generating turn via run_agent ──
async def ttft(prompt):
    import httpx
    body = {"model": A.model, "stream": True, "temperature": 0.3, "max_tokens": 200,
            "messages": [{"role": "system", "content": agent.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    t0 = time.perf_counter(); first = None
    async with httpx.AsyncClient(timeout=120) as cl:
        async with cl.stream("POST", A.url.rstrip("/") + "/v1/chat/completions", json=body) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and re.search(r'"content":"[^"]', line) and first is None:
                    first = (time.perf_counter() - t0) * 1000
    return first, (time.perf_counter() - t0) * 1000


PROSE = [("Объясни простыми словами, как работает фотосинтез.", "ru"),
         ("Erklär mir kurz, warum der Himmel blau ist.", "de"),
         ("Tell me a short bedtime story about a fox.", "en")]


async def suite_latency():
    out = []
    for p, lang in PROSE:
        for k in range(A.runs):
            f, tot = await ttft(p)
            out.append(dict(kind="ttft", text=p, ttft_ms=f, total_ms=tot))
            r, calls, ms = await turn(p, lang)
            out.append(dict(kind="prose_turn", text=p, ms=ms, calls=[c.name for c in calls], chars=len(r.response_text or "")))
    return out


async def main():
    res = {"model": A.model, "url": A.url, "started": time.time()}
    for s in A.suites:
        t0 = time.time()
        res[s] = await globals()[f"suite_{s}"]()
        print(f"[{A.model}] {s} done in {time.time() - t0:.0f}s", flush=True)
        Path(A.json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

asyncio.run(main())
