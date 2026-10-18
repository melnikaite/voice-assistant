"""
Telegram channel — linking, gating, turn routing, reminders, splitting.

The Bot API is replaced by ``FakeApi`` (records calls, serves canned
files); ASR, vision and the agent loop are patched at the pipeline's
import sites.  No network.
"""
from __future__ import annotations

import io
import time
import wave
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app import telegram
from app.agent import AgentResult
from app.storage import (
    add_reminder,
    create_telegram_link_token,
    get_telegram_link,
    get_telegram_link_for_profile,
    save_speaker_profile,
)
from app.storage.db import _conn


class FakeApi:
    def __init__(self, files: dict[str, bytes] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.documents: list[tuple[int, str]] = []
        self.files = files or {}

    async def call(self, method, **params):
        self.calls.append((method, params))
        if method == "getMe":
            return {"username": "test_bot"}
        return {}

    async def download_file(self, file_id):
        return self.files[file_id]

    async def send_document(self, chat_id, filename, data):
        self.documents.append((chat_id, filename))

    async def close(self):
        pass

    def sent(self) -> list[str]:
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


USER_ID = 1001


def _msg(**fields) -> dict:
    base = {
        "message_id": 1,
        "chat": {"id": USER_ID, "type": "private"},
        "from": {"id": USER_ID, "is_bot": False, "first_name": "Ann", "language_code": "en"},
    }
    base.update(fields)
    return {"update_id": 1, "message": base}


async def _profile(name="Ann") -> int:
    return await save_speaker_profile("browser-1", name, np.zeros(256, np.float32).tobytes())


async def _linked_bot(api: FakeApi) -> tuple[telegram.TelegramBot, int]:
    pid = await _profile()
    bot = telegram.TelegramBot(api)
    token = await create_telegram_link_token(pid)
    await bot.handle_update(_msg(text=f"/start {token}"))
    return bot, pid


@pytest.fixture(autouse=True)
def _no_memory(monkeypatch):
    monkeypatch.setattr("app.memory.EMBEDDING_ENABLED", False)


# ── Linking ────────────────────────────────────────────────────────────


async def test_link_with_valid_token():
    api = FakeApi()
    bot, pid = await _linked_bot(api)
    link = await get_telegram_link(USER_ID)
    assert link["profile_id"] == pid and link["chat_id"] == USER_ID
    assert "Ann" in api.sent()[-1]


async def test_link_token_is_single_use():
    api = FakeApi()
    pid = await _profile()
    token = await create_telegram_link_token(pid)
    bot = telegram.TelegramBot(api)
    await bot.handle_update(_msg(text=f"/start {token}"))
    # A second account trying the same token is refused.
    other = _msg(text=f"/start {token}")
    other["message"]["from"]["id"] = 2002
    other["message"]["chat"]["id"] = 2002
    await bot.handle_update(other)
    assert await get_telegram_link(2002) is None
    assert (await get_telegram_link(USER_ID))["profile_id"] == pid


async def test_link_token_expired():
    api = FakeApi()
    pid = await _profile()
    token = await create_telegram_link_token(pid)
    _conn().execute("UPDATE telegram_link_tokens SET expires_at=?", (time.time() - 1,))
    await telegram.TelegramBot(api).handle_update(_msg(text=f"/start {token}"))
    assert await get_telegram_link(USER_ID) is None
    assert api.sent()  # told the code is invalid


async def test_relink_replaces_previous_profile():
    api = FakeApi()
    bot, pid = await _linked_bot(api)
    pid2 = await _profile("Bob")
    token = await create_telegram_link_token(pid2)
    await bot.handle_update(_msg(text=f"/start {token}"))
    assert (await get_telegram_link(USER_ID))["profile_id"] == pid2
    assert await get_telegram_link_for_profile(pid) is None


# ── Gating ─────────────────────────────────────────────────────────────


async def test_unlinked_user_ignored_without_llm():
    api = FakeApi()
    with patch("app.pipeline.run_agent", new=AsyncMock()) as agent:
        await telegram.TelegramBot(api).handle_update(_msg(text="hello"))
    agent.assert_not_called()
    assert api.calls == []


async def test_group_chat_ignored():
    api = FakeApi()
    bot, _ = await _linked_bot(api)
    api.calls.clear()
    upd = _msg(text="hi")
    upd["message"]["chat"]["type"] = "group"
    with patch("app.pipeline.run_agent", new=AsyncMock()) as agent:
        await bot.handle_update(upd)
    agent.assert_not_called()
    assert api.calls == []


# ── Turns ──────────────────────────────────────────────────────────────


async def test_text_turn_carries_profile_and_auth():
    api = FakeApi()
    bot, pid = await _linked_bot(api)
    with patch(
        "app.pipeline.run_agent",
        new=AsyncMock(return_value=AgentResult(response_text="Hi 👋 <b>&")),
    ) as agent, patch("app.pipeline.transcribe", new=AsyncMock()) as asr:
        await bot.handle_update(_msg(text="hello 🙂"))
    asr.assert_not_called()
    (transcript,), kw = agent.call_args
    assert transcript == "hello 🙂"
    assert kw["ctx"].profile_id == pid
    assert kw["ctx"].is_authenticated is True
    assert kw["ctx"].step_up_auth is True
    assert kw["ctx"].client_id == f"tg:{USER_ID}"
    # Plain text, no parse_mode — markup characters arrive verbatim.
    sends = [p for m, p in api.calls if m == "sendMessage"]
    assert sends[-1]["text"] == "Hi 👋 <b>&" and "parse_mode" not in sends[-1]


async def test_new_command_clears_history():
    api = FakeApi()
    bot, _ = await _linked_bot(api)
    with patch("app.pipeline.run_agent", new=AsyncMock(return_value=AgentResult(response_text="a"))):
        await bot.handle_update(_msg(text="one"))
    state = bot._chats[USER_ID]
    assert len(state.history) == 2
    await bot.handle_update(_msg(text="/new"))
    assert state.history == []


async def test_photo_goes_to_vision():
    api = FakeApi(files={"big": b"\x89PNG..."})
    bot, _ = await _linked_bot(api)
    photo = [
        {"file_id": "small", "file_size": 10, "width": 90},
        {"file_id": "big", "file_size": 1000, "width": 1280},
    ]
    with patch("app.pipeline.vision.analyze_image_b64", new=AsyncMock(return_value="a cat")) as vis, \
         patch("app.pipeline.run_agent", new=AsyncMock()) as agent:
        await bot.handle_update(_msg(photo=photo, caption="what is it?"))
    agent.assert_not_called()
    b64, question = vis.call_args.args
    assert question == "what is it?" and b64
    assert api.sent()[-1] == "a cat"


async def test_image_document_without_caption_uses_default_prompt():
    api = FakeApi(files={"doc": b"jpeg"})
    bot, _ = await _linked_bot(api)
    with patch("app.pipeline.vision.analyze_image_b64", new=AsyncMock(return_value="ok")) as vis:
        await bot.handle_update(_msg(document={"file_id": "doc", "mime_type": "image/jpeg"}))
    assert vis.call_args.args[1]  # non-empty localized default question


def _wav(seconds=1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros(int(48000 * seconds), np.int16).tobytes())
    return buf.getvalue()


async def test_voice_note_goes_through_asr_without_speaker_id():
    api = FakeApi(files={"v": _wav()})
    bot, pid = await _linked_bot(api)
    with patch("app.pipeline.transcribe", new=AsyncMock(return_value=("what time is it", 50))) as asr, \
         patch("app.pipeline.speaker.encode_audio_full", new=AsyncMock()) as spk, \
         patch("app.pipeline.run_agent", new=AsyncMock(return_value=AgentResult(response_text="noon"))) as agent:
        await bot.handle_update(_msg(voice={"file_id": "v", "duration": 1}))
    pcm = asr.call_args.args[0]
    assert abs(len(pcm) - 32000) < 2000  # 1 s at 16 kHz int16
    spk.assert_not_called()
    assert agent.call_args.args[0] == "what time is it"
    assert agent.call_args.kwargs["ctx"].profile_id == pid
    assert api.sent()[-1] == "noon"


def test_decode_audio_resamples_to_16k_mono():
    pcm = telegram.decode_audio_to_pcm16k(_wav(0.5))
    assert abs(len(pcm) - 16000) < 1000


# ── Reminders ──────────────────────────────────────────────────────────


async def test_reminder_from_web_ui_also_reaches_telegram(monkeypatch):
    from app import scheduler

    api = FakeApi()
    bot, pid = await _linked_bot(api)
    monkeypatch.setattr(telegram, "_bot", bot)
    rid = await add_reminder("browser-1", time.time(), "take the pills", profile_id=pid)
    with patch("app.registry.push", new=AsyncMock(return_value=True)) as ws_push:
        await scheduler._fire(rid, "browser-1", "take the pills")
    ws_push.assert_awaited_once()
    assert api.sent()[-1] == "take the pills"


async def test_reminder_for_unlinked_profile_unchanged(monkeypatch):
    from app import scheduler

    api = FakeApi()
    monkeypatch.setattr(telegram, "_bot", telegram.TelegramBot(api))
    pid = await _profile("Bob")
    rid = await add_reminder("browser-1", time.time(), "x", profile_id=pid)
    with patch("app.registry.push", new=AsyncMock(return_value=False)):
        await scheduler._fire(rid, "browser-1", "x")
    assert api.sent() == []
    row = _conn().execute("SELECT fired, delivered FROM reminders WHERE id=?", (rid,)).fetchone()
    assert tuple(row) == (1, 0)


async def test_reminder_set_in_telegram_delivered_there(monkeypatch):
    from app import scheduler

    api = FakeApi()
    bot, pid = await _linked_bot(api)
    monkeypatch.setattr(telegram, "_bot", bot)
    cid = f"tg:{USER_ID}"
    rid = await add_reminder(cid, time.time(), "call mom", profile_id=pid)
    await scheduler._fire(rid, cid, "call mom")
    assert api.sent().count("call mom") == 1
    row = _conn().execute("SELECT delivered FROM reminders WHERE id=?", (rid,)).fetchone()
    assert row[0] == 1


# ── Splitting ──────────────────────────────────────────────────────────


def test_split_short_message_untouched():
    assert telegram.split_message("hi 👋") == ["hi 👋"]


def test_split_respects_limit_and_prefers_breaks():
    text = ("word " * 2000).strip()
    parts = telegram.split_message(text, limit=4096)
    assert len(parts) > 1
    assert all(telegram._u16len(p) <= 4096 for p in parts)
    assert " ".join(parts) == text


def test_split_counts_emoji_as_two_units():
    text = "😀" * 3000  # 6000 UTF-16 units
    parts = telegram.split_message(text, limit=4096)
    assert all(telegram._u16len(p) <= 4096 for p in parts)
    assert "".join(parts) == text


async def test_private_tool_runs_from_telegram_without_step_up_request():
    from app.agent import _execute_one
    from app.llm_utils import ParsedToolCall
    from app.tools.base import TOOL_REGISTRY, ToolResult, tool

    @tool("test_tg_private", "desc", {"type": "object", "properties": {}, "required": []}, private=True)
    async def _private():
        return ToolResult(text="secret ok")

    api = FakeApi()
    bot, _ = await _linked_bot(api)
    seen = {}

    async def fake_agent(transcript, *, ctx, history, memory_context):
        inv = await _execute_one(ParsedToolCall(id="1", name="test_tg_private", args={}), ctx)
        seen["inv"] = inv
        return AgentResult(response_text=inv.text)

    try:
        with patch("app.pipeline.run_agent", new=fake_agent), \
             patch("app.push.send_to_profile", new=AsyncMock()) as push, \
             patch("app.storage.step_up_grants.create_grant", new=AsyncMock()) as grant:
            await bot.handle_update(_msg(text="show my secret"))
    finally:
        TOOL_REGISTRY.pop("test_tg_private", None)
    push.assert_not_called()
    grant.assert_not_called()
    assert not (seen["inv"].data or {}).get("step_up_pending")
    assert api.sent()[-1] == "secret ok"


async def test_telegram_reminder_reaches_profile_tablet_once(monkeypatch):
    from app import scheduler

    api = FakeApi()
    bot, pid = await _linked_bot(api)  # profile's home client is "browser-1"
    monkeypatch.setattr(telegram, "_bot", bot)

    class _Tab:
        client_id = "browser-1"
        session_id = 7

    from app import ws_registry
    monkeypatch.setitem(ws_registry._SESSIONS_BY_PROFILE, pid, [_Tab()])
    cid = f"tg:{USER_ID}"
    rid = await add_reminder(cid, time.time(), "water plants", profile_id=pid)
    with patch("app.registry.push", new=AsyncMock(return_value=True)) as ws_push:
        await scheduler._fire(rid, cid, "water plants")
    pushed = [c.args[0] for c in ws_push.await_args_list]
    assert pushed.count("browser-1") == 1  # home client + live tab de-duplicated
    assert api.sent().count("water plants") == 1


async def test_web_reminder_not_pushed_twice_to_creating_client(monkeypatch):
    from app import scheduler

    api = FakeApi()
    bot, pid = await _linked_bot(api)
    monkeypatch.setattr(telegram, "_bot", bot)
    rid = await add_reminder("browser-1", time.time(), "y", profile_id=pid)
    with patch("app.registry.push", new=AsyncMock(return_value=True)) as ws_push:
        await scheduler._fire(rid, "browser-1", "y")
    assert [c.args[0] for c in ws_push.await_args_list] == ["browser-1"]
