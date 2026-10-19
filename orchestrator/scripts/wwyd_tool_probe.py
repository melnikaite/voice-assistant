"""Tool-choice probe for remember / wwyd_remember / how_would_answer — NOT pytest.

Single-turn utterances against a live LLM (LocalAI :1240); tool execution and
usage recording are stubbed like context_probe.py, so nothing touches the DB,
memory.md or a real wwyd.  Cases are split into a `tune` and a `held` half:
tune descriptions on the first, report the second.

Usage (from orchestrator/):
    uv run python scripts/wwyd_tool_probe.py --runs 3 [--split tune|held|all]
        [--services wwyd,wwyd_ask | --services ""] [--json out.json]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import tempfile
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--url", default="http://localhost:1240")
_ap.add_argument("--model", default="qwen3-vl-8b-instruct")
_ap.add_argument("--runs", type=int, default=3)
_ap.add_argument("--split", default="all", choices=["tune", "held", "all"])
_ap.add_argument("--services", default="wwyd,wwyd_ask")
_ap.add_argument("--json", default=None)
ARGS = _ap.parse_args()

os.environ["LLM_URL"] = ARGS.url
os.environ["LLM_MODEL"] = ARGS.model
os.environ.setdefault("DB_PATH", str(Path(tempfile.mkdtemp()) / "probe.db"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import agent, llm_utils  # noqa: E402
from app.agent import AgentContext, ToolInvocation, run_agent  # noqa: E402

llm_utils._fire_record_usage = lambda *a, **k: None

_CALLS: list[ToolInvocation] = []


async def _fake_execute(tc, ctx):
    inv = ToolInvocation(name=tc.name, args=tc.args, text="OK.", data={}, terminal=True)
    _CALLS.append(inv)
    return inv


agent._execute_one = _fake_execute

MEM, WR, HOW, NONE = "remember", "wwyd_remember", "how_would_answer", "none"
WATCHED = {MEM, WR, HOW}

# (utterance, expected) — tune half, then held-out half.
TUNE = [
    ("Запомни, что мои ключи от гаража лежат в верхнем ящике комода.", MEM),
    ("Запомни, у мамы день рождения двенадцатого марта.", MEM),
    ("Запомни, что я не ем глютен.", MEM),
    ("Remember that my dentist is Dr. Weber on Leopoldstraße.", MEM),
    ("Merk dir, dass meine Tochter Anna heißt.", MEM),
    ("Запомни, что я отвечаю клиентам коротко и на вы.", WR),
    ("Запомни, что я никогда не беру звонки до десяти утра.", WR),
    ("Запомни: на скидки я сразу не соглашаюсь, сначала прошу обоснование.", WR),
    ("Remember that I always decline meetings on Friday afternoons.", WR),
    ("Merk dir, dass ich Kollegen immer duze.", WR),
    ("Как бы я ответил клиенту, который просит скидку двадцать процентов?", HOW),
    ("Что бы сказал Женя, если позвать его в пятницу в бар?", HOW),
    ("Согласилась бы мама поехать с нами на дачу в субботу?", HOW),
    ("What would I reply to my boss asking me to work this weekend?", HOW),
    ("Wie würde ich auf eine Einladung zu einem Meeting um acht Uhr morgens antworten?", HOW),
    ("Напомни мне завтра в девять позвонить маме.", NONE),
    ("Отправь Жене сообщение, что я опаздываю.", NONE),
    ("Как ответить клиенту, который просит скидку, вежливо?", NONE),
    ("Привет, как дела?", NONE),
    ("Что ты помнишь обо мне?", NONE),
]
HELD = [
    ("Запомни, что паспорт лежит в синей папке в шкафу.", MEM),
    ("Запомни, мою собаку зовут Бублик.", MEM),
    ("Запомни, что я люблю кофе без сахара.", MEM),
    ("Remember that my wife's birthday is on June 4th.", MEM),
    ("Merk dir, dass das Auto in der Tiefgarage auf Platz 14 steht.", MEM),
    ("Запомни, что на письма от коллег я отвечаю в тот же день, а от незнакомых — никогда.", WR),
    ("Запомни, что я не обсуждаю зарплату по телефону, только письменно.", WR),
    ("Запомни, с семьёй я общаюсь на ты и с эмодзи.", WR),
    ("Remember that I never commit to a deadline without checking with my team first.", WR),
    ("Merk dir, dass ich Anfragen von Recruitern immer höflich ablehne.", WR),
    ("Как бы я ответил, если клиент просит созвониться в воскресенье?", HOW),
    ("Что бы ответила Маша, если пригласить её в кино?", HOW),
    ("Как бы Женя решил: брать отпуск в августе или в сентябре?", HOW),
    ("How would Eugene answer a recruiter offering a job in Berlin?", HOW),
    ("Was würde ich einem Kunden sagen, der die Rechnung nicht bezahlen will?", HOW),
    ("Напомни через час выключить духовку.", NONE),
    ("Напиши Маше, что ужин в семь.", NONE),
    ("Кто такой Сократ?", NONE),
    ("Спасибо, ты молодец.", NONE),
    ("What's the weather like tomorrow?", NONE),
]

SERVICES = frozenset(s for s in ARGS.services.split(",") if s)
CTX = AgentContext(client_id="probe", profile_id=1, is_authenticated=True,
                   user_lang="ru", services=SERVICES)


def picked() -> str:
    names = [c.name for c in _CALLS if c.name in WATCHED]
    return names[0] if names else NONE


async def main() -> None:
    splits = {"tune": TUNE, "held": HELD}
    if ARGS.split != "all":
        splits = {ARGS.split: splits[ARGS.split]}
    rows = []
    for split, cases in splits.items():
        conf: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for utt, want in cases:
            if not SERVICES:  # tools hidden: persona facts still go to remember
                want = {WR: MEM, HOW: NONE}.get(want, want)
            got = []
            for _ in range(ARGS.runs):
                _CALLS.clear()
                await run_agent(utt, ctx=CTX, history=[])
                got.append(picked())
                rows.append({"split": split, "utt": utt, "want": want, "got": got[-1],
                             "calls": [(c.name, c.args) for c in _CALLS]})
            for g in got:
                conf[want][g] += 1
            ok = sum(g == want for g in got)
            mark = "" if ok == len(got) else "  <-- " + ",".join(got)
            print(f"{split:4} {want:16} {ok}/{len(got)}  {utt[:70]}{mark}", flush=True)
            if ARGS.json:
                Path(ARGS.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        total = sum(sum(c.values()) for c in conf.values())
        hit = sum(conf[w][w] for w in conf)
        print(f"== {split} services={sorted(SERVICES)}: {hit}/{total}")
        for w in (MEM, WR, HOW, NONE):
            c = conf.get(w)
            if c:
                print(f"   {w:16} {c[w]}/{sum(c.values())}  {dict(c)}")


if __name__ == "__main__":
    asyncio.run(main())
