"""web_search reply-language probe (architecture.md §13) — NOT part of the pytest suite.

Feeds hand-written results in one language and a question in another through the
same summary prompt `web_search` uses (LocalAI on :1240 by default), then checks the
reply language: Cyrillic share for ru; for de/en, Latin script plus a stopword
majority vote (de vs en).  No DuckDuckGo involved.

Usage (from orchestrator/):
    uv run python scripts/web_search_lang_probe.py --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--url", default="http://localhost:1240")
_ap.add_argument("--model", default="qwen3-vl-8b-instruct")
_ap.add_argument("--runs", type=int, default=3)
_ap.add_argument("-v", action="store_true", help="print failing replies")
ARGS = _ap.parse_args()
os.environ["LLM_URL"] = ARGS.url
os.environ["LLM_MODEL"] = ARGS.model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm_utils import chat, extract_text  # noqa: E402
from app.tools import web_search as ws  # noqa: E402

EN = [
    {"title": "Bitcoin price today", "body": "Bitcoin is trading at $67,412 today, up 2.3% over the last 24 hours, with a market cap of $1.33 trillion."},
    {"title": "BTC live chart", "body": "The current price of Bitcoin is 67,400 US dollars. Trading volume rose to 28 billion dollars."},
]
EN_POPE = [
    {"title": "Pope Leo XIV - Wikipedia", "body": "Pope Leo XIV, born Robert Francis Prevost, is the head of the Catholic Church since 8 May 2025. He is the first pope from the United States."},
    {"title": "Who is the current pope?", "body": "The current pope is Leo XIV, elected in the conclave of May 2025 after the death of Pope Francis."},
]
EN_IPHONE = [
    {"title": "iPhone 17 release date", "body": "Apple released the iPhone 17 on September 19, 2025. Preorders started on September 12. The base model starts at $799."},
]
EN_MOON = [
    {"title": "Moon phase today", "body": "Today the Moon is a waxing gibbous, 78% illuminated. The next full moon is on October 22."},
]
DE_WEATHER = [
    {"title": "Wetter Berlin heute", "body": "In Berlin ist es heute bewölkt bei 14 Grad. Am Nachmittag ist mit leichtem Regen zu rechnen, der Wind weht mäßig aus Westen."},
    {"title": "Wettervorhersage Berlin", "body": "Höchstwerte um 15 Grad, nachts kühlt es auf 8 Grad ab. Regenwahrscheinlichkeit 60 Prozent."},
]
DE_TRAIN = [
    {"title": "Deutsche Bahn Streik", "body": "Die GDL hat für Donnerstag einen bundesweiten Streik im Personenverkehr angekündigt. Der Streik beginnt um 2 Uhr und dauert 24 Stunden."},
]
DE_FUEL = [
    {"title": "Benzinpreis aktuell", "body": "Der durchschnittliche Preis für Super E10 liegt heute bei 1,72 Euro pro Liter, Diesel kostet 1,61 Euro."},
]
RU_NEWS = [
    {"title": "Курс доллара", "body": "Официальный курс доллара на сегодня составляет 81,5 рубля, евро — 94,2 рубля."},
]

# Real pages are a few KB of source-language prose; short snippets alone don't leak.
_FILLER = {
    "en": "Our editors update this page several times a day. Prices and figures are provided for informational purposes only and may differ from those of your local provider. Read our methodology to learn how we collect and verify the data, and sign up for the newsletter to get the latest updates delivered to your inbox every morning. ",
    "de": "Unsere Redaktion aktualisiert diese Seite mehrmals täglich. Alle Angaben sind ohne Gewähr und können von den Werten Ihres lokalen Anbieters abweichen. Lesen Sie unsere Methodik, um zu erfahren, wie wir die Daten erheben und prüfen, und abonnieren Sie den Newsletter für die neuesten Meldungen jeden Morgen. ",
    "ru": "Наша редакция обновляет эту страницу несколько раз в день. Все данные приводятся в информационных целях и могут отличаться от данных вашего местного поставщика. Прочитайте нашу методику, чтобы узнать, как мы собираем и проверяем данные, и подпишитесь на рассылку, чтобы каждое утро получать свежие новости. ",
}


def _pages(results: list[dict], src: str) -> list[dict]:
    out = []
    for r in results:
        text = r["body"] + " "
        while len(text) < 2800:
            text += _FILLER[src] + r["body"] + " "
        out.append({**r, "page_text": text})
    return out + results  # pages first, then the bare snippets


CASES = [  # (name, lang, question, results)
    ("ru<-en btc", "ru", "Сколько сейчас стоит биткоин?", EN),
    ("ru<-en pope", "ru", "Кто сейчас папа римский?", EN_POPE),
    ("ru<-en iphone", "ru", "Когда вышел айфон 17?", EN_IPHONE),
    ("ru<-en moon", "ru", "Какая сегодня фаза луны?", EN_MOON),
    ("ru<-de weather", "ru", "Какая погода в Берлине сегодня?", DE_WEATHER),
    ("ru<-de strike", "ru", "Будет ли забастовка на железной дороге?", DE_TRAIN),
    ("ru<-de fuel", "ru", "Сколько стоит бензин?", DE_FUEL),
    ("de<-en btc", "de", "Wie viel kostet Bitcoin gerade?", EN),
    ("de<-en pope", "de", "Wer ist der aktuelle Papst?", EN_POPE),
    ("de<-en iphone", "de", "Wann kam das iPhone 17 raus?", EN_IPHONE),
    ("en<-de weather", "en", "What's the weather in Berlin today?", DE_WEATHER),
    ("en<-ru rate", "en", "What's the dollar exchange rate in rubles?", RU_NEWS),
    # The agent LLM writes the tool's `query` itself and often in English or in
    # the locale language even when the user spoke Russian/German.
    ("ru,q=en btc", "ru", "bitcoin price today", EN),
    ("ru,q=en pope", "ru", "current pope", EN_POPE),
    ("ru,q=de weather", "ru", "Wetter Berlin heute", DE_WEATHER),
    ("ru,q=de fuel", "ru", "Benzinpreis aktuell", DE_FUEL),
    ("de,q=en iphone", "de", "iPhone 17 release date", EN_IPHONE),
    ("de,q=en moon", "de", "moon phase today", EN_MOON),
]

_STOP = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "bei", "heute", "für", "von", "es", "im", "auf", "wird", "um", "zu", "den", "aktuell"},
    "en": {"the", "and", "is", "of", "a", "an", "with", "today", "for", "in", "it", "on", "will", "at", "to", "currently", "are", "was"},
}


def detect(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "?"
    if sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters) > 0.5:
        return "ru"
    words = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(w in _STOP["de"] for w in words)
    en = sum(w in _STOP["en"] for w in words)
    return "de" if de > en else "en"


def _src(results: list[dict]) -> str:
    return detect(" ".join(r["body"] for r in results))


async def one(lang: str, q: str, results: list[dict]) -> str:
    results = _pages(results, _src(results))
    choice = await chat(
        ws._summary_messages(q, results, lang),
        temperature=0.2, reasoning_effort="low", tool_name="probe",
    )
    return extract_text(choice["message"]).strip()


async def main() -> None:
    total = ok_all = 0
    for name, lang, q, results in CASES:
        ok = 0
        for _ in range(ARGS.runs):
            reply = await one(lang, q, [dict(r) for r in results])
            got = detect(reply)
            ok += got == lang
            if got != lang and ARGS.v:
                print(f"   FAIL {name} -> {got}: {reply[:160]!r}")
        print(f"{name:16} {ok}/{ARGS.runs}", flush=True)
        total += ARGS.runs
        ok_all += ok
    print(f"TOTAL {ok_all}/{total}")


asyncio.run(main())
