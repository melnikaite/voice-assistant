"""
Telegram channel — a second front-end to the same turn pipeline the web UI uses.

Off unless ``TELEGRAM_BOT_TOKEN`` is set.  Runs as a background task inside
the orchestrator (started from the FastAPI lifespan) and talks to the Bot
API by long polling, so it needs no inbound port and works behind NAT.

Trust model: Telegram is an identified channel like the logged-in web UI.
A linked Telegram account (``telegram_links``) maps to exactly one
speaker_profile; its turns run as that profile and authenticated, the
same way a ``va_session`` cookie does for a browser tab.  No speaker-ID,
no passphrase.  Unlinked accounts get nothing but the ``/start`` linking
reply — the LLM is never called for them.  Private chats only.

Each chat is its own pipeline client (``client_id = "tg:<chat_id>"``), so
history, reminders and per-client memory stay with the chat.  A reminder
carries its owning profile; when it fires, ``scheduler._fire`` sends it to
every channel that profile has, including its linked chat
(:func:`deliver_to_profile`), whichever channel created it.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

from .conversation import Conversation
from .i18n import SUPPORTED_LANGS, t
from .pipeline import Pipeline
from .storage import (
    consume_telegram_link_token,
    get_missed_reminders_for_profile,
    get_telegram_link,
    get_telegram_link_for_profile,
    list_telegram_links,
    mark_reminder_delivered,
    start_session,
)
from .storage.sessions import get_recent_turns
from .user_files import read_settings

log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_API_URL = os.environ.get("TELEGRAM_API_URL", "https://api.telegram.org").rstrip("/")
# Long-poll window; the HTTP timeout must exceed it.
POLL_TIMEOUT_S = int(os.environ.get("TELEGRAM_POLL_TIMEOUT_S", "30"))
# Voice notes / audio files longer than this are refused before ASR.
MAX_AUDIO_S = int(os.environ.get("TELEGRAM_MAX_AUDIO_S", "600"))

CLIENT_PREFIX = "tg:"
MESSAGE_LIMIT = 4096  # Bot API sendMessage limit, in UTF-16 code units
BACKOFF_MAX_S = 60.0


def client_id_for_chat(chat_id: int) -> str:
    return f"{CLIENT_PREFIX}{chat_id}"


# ── Message splitting ───────────────────────────────────────────────────


def _u16len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` UTF-16 code units.

    Telegram measures length in UTF-16 units, so an emoji counts as two.
    Cuts prefer a paragraph break, then a line break, then a space, as
    long as that keeps the chunk at least half full; otherwise a hard cut
    (always between code points, never inside a surrogate pair).
    """
    chunks: list[str] = []
    rest = text
    while rest:
        if _u16len(rest) <= limit:
            chunks.append(rest)
            break
        # Largest prefix (in code points) that fits.
        used, cut = 0, 0
        for i, ch in enumerate(rest):
            w = 2 if ord(ch) > 0xFFFF else 1
            if used + w > limit:
                break
            used += w
            cut = i + 1
        head = rest[:cut]
        split_at = cut
        for sep in ("\n\n", "\n", " "):
            idx = head.rfind(sep)
            if idx >= cut // 2:
                split_at = idx + len(sep)
                break
        piece = rest[:split_at].rstrip()
        if piece:
            chunks.append(piece)
        rest = rest[split_at:].lstrip("\n ")
    return chunks or [""]


# ── Audio ───────────────────────────────────────────────────────────────


def decode_audio_to_pcm16k(data: bytes) -> bytes:
    """Decode any container/codec PyAV knows (OGG/Opus voice notes, MP3,
    M4A, WAV …) to 16 kHz mono int16 PCM — the pipeline's audio shape.

    PyAV is already installed as an aiortc dependency, so this adds no
    package and needs no ffmpeg binary.
    """
    import av  # heavy import, only needed for audio turns

    out = bytearray()
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                out += f.to_ndarray().tobytes()
        for f in resampler.resample(None):
            out += f.to_ndarray().tobytes()
    return bytes(out)


# ── Bot API client ──────────────────────────────────────────────────────


class TelegramApiError(Exception):
    pass


class TelegramApi:
    """Minimal Bot API client over the httpx the orchestrator already uses."""

    def __init__(self, token: str, base_url: str = TELEGRAM_API_URL):
        self._base = f"{base_url}/bot{token}"
        self._file_base = f"{base_url}/file/bot{token}"
        self._client = httpx.AsyncClient(timeout=POLL_TIMEOUT_S + 15)

    async def close(self) -> None:
        await self._client.aclose()

    async def call(self, method: str, **params) -> object:
        r = await self._client.post(f"{self._base}/{method}", json=params)
        try:
            body = r.json()
        except ValueError:
            raise TelegramApiError(f"{method}: HTTP {r.status_code}")
        if not body.get("ok"):
            raise TelegramApiError(f"{method}: {body.get('description') or r.status_code}")
        return body.get("result")

    async def send_document(self, chat_id: int, filename: str, data: bytes) -> None:
        r = await self._client.post(
            f"{self._base}/sendDocument",
            data={"chat_id": str(chat_id)},
            files={"document": (filename, data)},
        )
        if r.status_code != 200:
            raise TelegramApiError(f"sendDocument: HTTP {r.status_code}")

    async def download_file(self, file_id: str) -> bytes:
        info = await self.call("getFile", file_id=file_id)
        path = (info or {}).get("file_path")  # type: ignore[union-attr]
        if not path:
            raise TelegramApiError("getFile: no file_path")
        r = await self._client.get(f"{self._file_base}/{path}")
        r.raise_for_status()
        return r.content


# ── Per-chat state + pipeline hooks ─────────────────────────────────────


@dataclass
class ChatState:
    client_id: str
    session_id: int
    conv: Conversation = field(default_factory=Conversation)
    last_response_text: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TelegramHooks:
    """PipelineHooks for one Telegram turn: collect output, send it after.

    The linked profile is returned from ``cookie_profile_id`` and the auth
    and step-up windows are always open — the same shape a cookie-authenticated web UI
    session presents to the pipeline.
    """

    def __init__(self, profile_id: int, chat: ChatState):
        self._profile_id = profile_id
        self._chat = chat
        self.responses: list[str] = []
        self.replay = False
        self.error: str | None = None
        self.voicemail_ids: list[int] = []

    async def on_transcript(self, *, text, asr_ms, speaker) -> None: ...
    async def on_speaker_identified(self, *, name, tts_voice) -> None: ...

    async def on_tool_called(self, *, name, args, data=None) -> None:
        play_id = (data or {}).get("voicemail_play_id")
        if play_id:
            self.voicemail_ids.append(int(play_id))

    async def on_response(self, *, text, llm_ms, history_turns, tts_voice_override=None) -> None:
        if text:
            self.responses.append(text)

    async def on_response_chunk(self, *, text) -> None:
        # The full text arrives in on_response; chunks only matter for TTS.
        ...

    async def on_response_end(self) -> None: ...

    async def on_error(self, message: str) -> None:
        self.error = message

    async def on_replay(self, mode: str) -> None:
        self.replay = True

    async def on_history_reset(self) -> None:
        self._chat.conv.reset()

    async def on_progress(self, *, step, detail=None) -> None: ...
    async def on_media_started(self, *, url, source) -> None: ...

    async def current_auth_until(self) -> float | None:
        return time.time() + 3600

    async def on_auth_succeeded(self, *, profile_id, window_s) -> None: ...

    async def cookie_profile_id(self) -> int | None:
        return self._profile_id

    async def current_step_up_auth_until(self) -> float | None:
        # Identified channel: no extra confirmation for private tools.  The
        # Web Push step-up grant is delivered to WS sessions only, so a
        # Telegram turn could never complete it anyway.
        return time.time() + 3600


# ── Bot ─────────────────────────────────────────────────────────────────


async def _lang_for(profile_id: int | None, tg_lang: str | None) -> str:
    if profile_id is not None:
        try:
            s = await read_settings(profile_id)
            if s.language and s.language != "auto":
                return s.language
        except Exception:
            log.debug("telegram: settings read failed", exc_info=True)
    code = (tg_lang or "").split("-")[0].lower()
    return code if code in SUPPORTED_LANGS else "en"


class TelegramBot:
    def __init__(self, api: TelegramApi):
        self.api = api
        self.username: str | None = None
        self._chats: dict[int, ChatState] = {}
        self._offset: int | None = None
        self._tasks: set[asyncio.Task] = set()

    # -- lifecycle --

    async def run(self) -> None:
        """Poll forever; network/API errors back off and retry."""
        backoff = 1.0
        while self.username is None:
            try:
                me = await self.api.call("getMe")
                self.username = (me or {}).get("username")  # type: ignore[union-attr]
                log.info("telegram: bot @%s ready", self.username)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("telegram: getMe failed (%s) — retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)
        await self._deliver_missed_reminders()
        backoff = 1.0
        while True:
            try:
                params: dict = {"timeout": POLL_TIMEOUT_S, "allowed_updates": ["message"]}
                if self._offset is not None:
                    params["offset"] = self._offset
                updates = await self.api.call("getUpdates", **params) or []
                backoff = 1.0
                for upd in updates:  # type: ignore[union-attr]
                    self._offset = int(upd["update_id"]) + 1
                    task = asyncio.create_task(self._safe_handle(upd))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("telegram: getUpdates failed (%s) — retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self.api.close()

    async def _deliver_missed_reminders(self) -> None:
        """Reminders that fired while Telegram was unreachable."""
        try:
            for link in await list_telegram_links():
                for reminder_id, text in await get_missed_reminders_for_profile(
                    link["profile_id"]
                ):
                    if await self.send_text(link["chat_id"], text):
                        await mark_reminder_delivered(reminder_id)
        except Exception:
            log.warning("telegram: missed-reminder delivery failed", exc_info=True)

    # -- sending --

    async def send_text(self, chat_id: int, text: str) -> bool:
        # Plain text, no parse_mode: model output is arbitrary, and any
        # Markdown/HTML mode would need escaping to not be rejected.
        try:
            for chunk in split_message(text):
                await self.api.call("sendMessage", chat_id=chat_id, text=chunk)
            return True
        except Exception as exc:
            log.warning("telegram: sendMessage to %s failed: %s", chat_id, exc)
            return False

    # -- inbound --

    async def _safe_handle(self, update: dict) -> None:
        try:
            await self.handle_update(update)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("telegram: update %s failed", update.get("update_id"))

    async def handle_update(self, update: dict) -> None:
        msg = update.get("message")
        if not isinstance(msg, dict):
            return
        chat = msg.get("chat") or {}
        sender = msg.get("from") or {}
        if chat.get("type") != "private" or not sender.get("id") or sender.get("is_bot"):
            return
        chat_id, user_id = int(chat["id"]), int(sender["id"])
        text = msg.get("text")

        if isinstance(text, str) and _command(text) == "/start":
            await self._handle_start(chat_id, user_id, sender, text)
            return

        link = await get_telegram_link(user_id)
        if link is None:
            log.info("telegram: ignoring message from unlinked user %s", user_id)
            return
        profile_id = link["profile_id"]
        state = await self._chat_state(chat_id)
        async with state.lock:
            await self._handle_linked(msg, chat_id, sender, link, profile_id, state)

    async def _handle_start(self, chat_id: int, user_id: int, sender: dict, text: str) -> None:
        token = text.split(maxsplit=1)[1].strip() if len(text.split(maxsplit=1)) > 1 else ""
        tg_lang = sender.get("language_code")
        if token:
            name = " ".join(
                p for p in (sender.get("first_name"), sender.get("last_name")) if p
            ) or (sender.get("username") or "")
            link = await consume_telegram_link_token(token, user_id, chat_id, name)
            if link is None:
                await self.send_text(chat_id, t("telegram.link_invalid", await _lang_for(None, tg_lang)))
                return
            # A previous profile's conversation must not carry over.
            self._chats.pop(chat_id, None)
            log.info("telegram: user %s linked to profile %d", user_id, link["profile_id"])
            lang = await _lang_for(link["profile_id"], tg_lang)
            await self.send_text(chat_id, t("telegram.linked", lang, name=link["profile_name"]))
            return
        link = await get_telegram_link(user_id)
        if link is None:
            await self.send_text(chat_id, t("telegram.not_linked", await _lang_for(None, tg_lang)))
        else:
            lang = await _lang_for(link["profile_id"], tg_lang)
            await self.send_text(chat_id, t("telegram.already_linked", lang, name=link["profile_name"]))

    async def _chat_state(self, chat_id: int) -> ChatState:
        state = self._chats.get(chat_id)
        if state is None:
            from .ws import MAX_HISTORY_TURNS  # same window as the web UI

            cid = client_id_for_chat(chat_id)
            session_id = await start_session(client="telegram", client_id=cid)
            conv = Conversation.from_rows(
                await get_recent_turns(cid, max_turns=MAX_HISTORY_TURNS)
            )
            state = self._chats.setdefault(
                chat_id, ChatState(client_id=cid, session_id=session_id, conv=conv)
            )
        return state

    async def _handle_linked(
        self, msg: dict, chat_id: int, sender: dict, link: dict, profile_id: int, state: ChatState
    ) -> None:
        lang = await _lang_for(profile_id, sender.get("language_code"))
        text = msg.get("text")
        if isinstance(text, str) and _command(text) == "/new":
            state.conv.reset()
            await self.send_text(chat_id, t("pipeline.new_topic_confirm", lang))
            return

        run_kwargs: dict = {}
        caption = (msg.get("caption") or "").strip()
        document = msg.get("document") or {}
        doc_mime = (document.get("mime_type") or "").lower()
        audio = msg.get("voice") or msg.get("audio") or (
            document if doc_mime.startswith("audio/") else None
        )

        if isinstance(text, str):
            run_kwargs["text"] = text
        elif msg.get("photo") or doc_mime.startswith("image/"):
            file_id = (
                max(msg["photo"], key=lambda p: p.get("file_size") or p.get("width", 0))["file_id"]
                if msg.get("photo") else document["file_id"]
            )
            data = await self.api.download_file(file_id)
            run_kwargs["text"] = caption or t("telegram.image_default_prompt", lang)
            run_kwargs["attached_image_b64"] = base64.b64encode(data).decode("ascii")
        elif audio:
            duration = int(audio.get("duration") or 0)
            if duration > MAX_AUDIO_S:
                await self.send_text(
                    chat_id, t("telegram.audio_too_long", lang, minutes=MAX_AUDIO_S // 60)
                )
                return
            data = await self.api.download_file(audio["file_id"])
            try:
                pcm = await asyncio.to_thread(decode_audio_to_pcm16k, data)
            except Exception:
                log.warning("telegram: audio decode failed", exc_info=True)
                await self.send_text(chat_id, t("telegram.audio_failed", lang))
                return
            run_kwargs["audio"] = pcm
        else:
            await self.send_text(chat_id, t("telegram.unsupported", lang))
            return

        await self._run_turn(chat_id, link, profile_id, state, lang, **run_kwargs)

    async def _run_turn(
        self, chat_id: int, link: dict, profile_id: int, state: ChatState, lang: str,
        *, audio: bytes = b"", text: str | None = None, attached_image_b64: str | None = None,
    ) -> None:
        from .ws import PIPELINE_TIMEOUT_S

        try:
            await self.api.call("sendChatAction", chat_id=chat_id, action="typing")
        except Exception:
            pass
        hooks = TelegramHooks(profile_id, state)
        try:
            outcome = await asyncio.wait_for(
                Pipeline(hooks).run(
                    audio=audio,
                    text=text,
                    session_id=state.session_id,
                    client_id=state.client_id,
                    history=state.conv.messages(),
                    conversation=state.conv,
                    last_response_text=state.last_response_text,
                    attached_image_b64=attached_image_b64,
                    identify_speaker=False,
                    voicemail_client_id=link.get("profile_client_id"),
                ),
                timeout=PIPELINE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            await self.send_text(chat_id, t("pipeline.thinking_too_long", lang))
            return

        # Same bookkeeping as ws.Session._run_pipeline.
        if outcome.history_cleared:
            state.conv.reset()
        elif outcome.history_appended and outcome.transcript and outcome.response_text:
            state.conv.record(outcome.transcript, outcome.response_text, outcome.invocations)

        if hooks.replay:
            if state.last_response_text:
                await self.send_text(chat_id, state.last_response_text)
            return
        if outcome.last_response_text:
            state.last_response_text = outcome.last_response_text

        for reply in hooks.responses:
            await self.send_text(chat_id, reply)
        if not hooks.responses and hooks.error:
            await self.send_text(chat_id, t("telegram.error", lang))
        for vm_id in hooks.voicemail_ids:
            await self._send_voicemail(chat_id, vm_id)

    async def _send_voicemail(self, chat_id: int, message_id: int) -> None:
        from .storage.voice_messages import audio_path

        try:
            data = await asyncio.to_thread(audio_path(message_id).read_bytes)
            await self.api.send_document(chat_id, f"voicemail-{message_id}.wav", data)
        except Exception:
            log.warning("telegram: voicemail %d upload failed", message_id, exc_info=True)


def _command(text: str) -> str:
    """``"/start@MyBot abc"`` → ``"/start"``; non-commands → ``""``."""
    if not text.startswith("/"):
        return ""
    return text.split(maxsplit=1)[0].split("@", 1)[0].lower()


# ── Module-level lifecycle (FastAPI lifespan) ───────────────────────────

_bot: TelegramBot | None = None
_task: asyncio.Task | None = None


def enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN)


def bot_username() -> str | None:
    return _bot.username if _bot else None


class _RedactToken(logging.Filter):
    """The token is part of every Bot API URL, which httpx logs per request."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if TELEGRAM_BOT_TOKEN in msg:
            record.msg, record.args = msg.replace(TELEGRAM_BOT_TOKEN, "<redacted>"), None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = record.exc_text.replace(TELEGRAM_BOT_TOKEN, "<redacted>")
        return True


def start() -> None:
    """Start polling if ``TELEGRAM_BOT_TOKEN`` is set; otherwise do nothing."""
    global _bot, _task
    if not enabled() or _task is not None:
        return
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, _RedactToken) for f in handler.filters):
            handler.addFilter(_RedactToken())
    _bot = TelegramBot(TelegramApi(TELEGRAM_BOT_TOKEN))
    _task = asyncio.create_task(_bot.run(), name="telegram-poll")
    log.info("telegram: polling started")


async def stop() -> None:
    global _bot, _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    if _bot is not None:
        await _bot.shutdown()
    _bot, _task = None, None


async def deliver_to_profile(profile_id: int, text: str) -> bool:
    """Send a server-initiated message (reminder) to the profile's linked chat.

    Returns False when the bot is off or the profile has no Telegram link.
    """
    if _bot is None:
        return False
    link = await get_telegram_link_for_profile(profile_id)
    if link is None:
        return False
    return await _bot.send_text(link["chat_id"], text)
