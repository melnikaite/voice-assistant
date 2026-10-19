"""
Automatic wwyd ingest: the owner's own words, picked between turns.

After a turn, ``observe`` queues it when the speaker is one identified
profile with wwyd connected (never guests, never mixed voices).  A
background task — same coalescing pattern as the conversation summary
fold — asks the LLM which parts are the owner's OWN words: messages they
dictate to someone, voicemail replies, statements about themselves.
Requests to the assistant are not.  Picked items go to wwyd's
``ingest_raw`` as episodes, where derived claims land PENDING for the
owner's review.

Nothing here runs on the reply path: the turn has been answered before
``observe`` is called.  Turns are kept in memory only until classified.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass

from . import wwyd
from .storage import get_wwyd_connection, list_wwyd_connected_profiles

log = logging.getLogger(__name__)

# Turns whose wwyd effect is explicit already.
_SKIP_TOOLS = {"wwyd_remember", "how_would_answer"}
_MAX_QUEUE = 20

# Cheap gate before spending an LLM call: the owner speaks in the first
# person, or tells someone something.
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|i've|i'll|me|my|mine|"
    r"я|мне|меня|мой|моя|моё|мои|мной|"
    r"ich|mich|mir|mein|meine|meinen)\b"
    r"|\b(tell|text|write|reply|answer|скажи|передай|напиши|ответь|sag|schreib|antworte)\b",
    re.IGNORECASE,
)

CLASSIFY_PROMPT = (
    "You read what one person said to their voice assistant and pick out the "
    "person's OWN words that show how they talk or behave:\n"
    "  * message: words they dictate to someone else (\"tell Anna I'll be late, "
    "sorry\" -> text \"I'll be late, sorry\", to \"Anna\"), or a reply to a "
    "voicemail;\n"
    "  * note: a statement about themselves — habits, rules, preferences, "
    "boundaries (\"I never take calls before 10\").\n"
    "Requests and questions to the assistant itself (weather, timers, music, "
    "\"what is…\", \"remind me…\", \"add milk\", \"play my playlist\") are NOT picked, "
    "even when they say \"my\".  Keep the "
    "person's wording and language; drop names, phone numbers, emails and "
    "addresses from the text.\n"
    "Answer with JSON only: {\"items\": [{\"type\": \"message\"|\"note\", "
    "\"text\": \"…\", \"to\": \"<recipient name or empty>\"}]}; "
    "{\"items\": []} when nothing qualifies."
)


@dataclass
class _Turn:
    text: str
    ts: str


_queues: dict[int, list[_Turn]] = {}
_tasks: dict[int, asyncio.Task] = {}


def observe(
    profile_id: int | None,
    mixed: bool,
    transcript: str,
    tool_name: str | None = None,
) -> None:
    """Queue a finished turn for classification.  Never blocks, never raises."""
    if profile_id is None or mixed or not transcript or tool_name in _SKIP_TOOLS:
        return
    if not _FIRST_PERSON.search(transcript):
        return
    q = _queues.setdefault(profile_id, [])
    if len(q) >= _MAX_QUEUE:
        return
    q.append(_Turn(transcript.strip(), dt.datetime.now().astimezone().isoformat(timespec="seconds")))
    task = _tasks.get(profile_id)
    if task is not None and not task.done():
        return  # the running drain picks it up
    try:
        _tasks[profile_id] = asyncio.get_running_loop().create_task(_drain(profile_id))
    except RuntimeError:
        pass


async def classify(texts: list[str]) -> list[dict]:
    """LLM pass over the owner's utterances; returns picked items."""
    from .llm_utils import chat, extract_text

    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(texts))
    choice = await chat(
        [{"role": "system", "content": CLASSIFY_PROMPT},
         {"role": "user", "content": numbered}],
        temperature=0.0, max_tokens=400, tool_name="<wwyd_capture>",
    )
    raw = extract_text(choice["message"]).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        items = json.loads(m.group(0)).get("items") if m else []
    except ValueError:
        return []
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        typ, text = it.get("type"), str(it.get("text") or "").strip()
        if typ in ("message", "note") and text:
            out.append({"type": typ, "text": text[:4000], "to": str(it.get("to") or "").strip()})
    return out


async def _episodes(profile_id: int, items: list[dict], ts: str) -> list[dict]:
    conn = await get_wwyd_connection(profile_id)
    amap = (conn or {}).get("audience_map") or {}
    profiles = await list_profiles_by_name()
    episodes = []
    # No category: wwyd places episodes itself, and a path we made up could carry a name.
    for it in items:
        if it["type"] == "note":
            episodes.append({"when": ts, "said": it["text"]})
            continue
        to = it["to"].lower()
        # Recipient name → household profile → owner's audience; names never leave.
        pid = next((p for p, n in profiles if to and (n.lower() == to or n.lower()[:3] == to[:3])), None)
        aud = amap.get(f"profile:{pid}") if pid is not None else None
        # said + audience without situation is an outgoing message; without the
        # audience wwyd would read it as a statement about the owner.
        episodes.append({"when": ts, "said": it["text"], "audience": aud or "default"})
    return episodes


async def list_profiles_by_name() -> list[tuple[int, str]]:
    from .storage import get_all_speaker_profiles

    return [(int(r[0]), r[1]) for r in await get_all_speaker_profiles()]


async def _drain(profile_id: int) -> None:
    while _queues.get(profile_id):
        batch, _queues[profile_id] = _queues[profile_id], []
        try:
            if profile_id not in {p for p, _ in await list_wwyd_connected_profiles()}:
                continue
            items = await classify([t.text for t in batch])
            if not items:
                continue
            episodes = await _episodes(profile_id, items, batch[-1].ts)
            await wwyd.call_tool(profile_id, "ingest_raw", {"episodes": episodes})
            log.info("wwyd: ingested %d episode(s) for profile %d", len(episodes), profile_id)
        except wwyd.WwydError as exc:
            log.warning("wwyd: ingest failed for profile %d: %s", profile_id, exc.code)
        except Exception as exc:  # the batch is dropped; content is never logged
            log.warning("wwyd: capture failed for profile %d: %s", profile_id, exc.__class__.__name__)


async def wait_idle() -> None:
    """Test hook: wait for every running drain."""
    for task in list(_tasks.values()):
        if not task.done():
            await asyncio.shield(task)
