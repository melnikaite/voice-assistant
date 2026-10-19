"""Probe the wwyd capture classifier on hand-written utterances.

Run against the real LLM (LLM_URL / LLM_MODEL), one utterance per call:

    LLM_URL=http://127.0.0.1:1240 LLM_MODEL=qwen3-vl-8b-instruct \\
        uv run python scripts/wwyd_capture_probe.py

Expected: the type picked ("message" / "note") or None for requests to the
assistant.  The cheap first-person gate runs first, as in production.
"""
from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("WHISPER_URL", "http://unused")
os.environ.setdefault("WHISPER_MODEL", "unused")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import wwyd_capture  # noqa: E402

CASES: list[tuple[str, str | None]] = [
    ("Tell Anna I'll be late tonight, sorry", "message"),
    ("I never take calls before 10 in the morning", "note"),
    ("Remind me to call the dentist at 5", None),
    ("What's the weather in Berlin tomorrow?", None),
    ("Reply to Mark: sounds good, let's do Thursday", "message"),
    ("Передай Маше, что я задержусь на полчаса", "message"),
    ("Я по пятницам не назначаю встречи", "note"),
    ("Поставь таймер на десять минут", None),
    ("Какая сегодня погода в Мюнхене?", None),
    ("Мне не нравится, когда пишут голосовыми", "note"),
    ("Sag Peter, dass ich morgen nicht kann", "message"),
    ("Ich antworte Kunden immer erst am nächsten Tag", "note"),
    ("Spiel meine Playlist ab", None),
    ("Wie spät ist es in Tokio?", None),
    ("Add milk to my shopping list", None),
]


async def main() -> None:
    from app.storage import init_schema

    init_schema()  # token accounting writes a row per LLM call
    hits = 0
    for text, want in CASES:
        got = None
        if wwyd_capture._FIRST_PERSON.search(text):
            items = await wwyd_capture.classify([text])
            got = items[0]["type"] if items else None
        ok = got == want
        hits += ok
        print(f"{'OK ' if ok else 'BAD'} want={want!s:8} got={got!s:8} {text}")
    print(f"score {hits}/{len(CASES)}")


asyncio.run(main())
