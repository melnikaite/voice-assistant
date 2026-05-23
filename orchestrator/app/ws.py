"""
WebSocket session — signalling + audio-state FSM.

What changed in the WebRTC migration:
  • Audio I/O moved off the WebSocket.  The mic now arrives via an
    RTCPeerConnection (peer's outbound audio track) and the assistant's
    voice is pushed back through our own outbound track.  This module
    bridges that audio into the existing VAD / pipeline FSM.
  • The WebSocket itself is now SIGNALLING ONLY: SDP offer/answer, ICE
    candidate exchange, plus our existing JSON state events (transcript,
    response, state, wake_ack, etc.).  No more binary PCM frames either
    way.
  • TTS is server-side (Piper).  We synth → push PCM into the outbound
    RTC track → the browser plays it through its WebRTC voice engine,
    which means the built-in AEC3 sees it as reference audio and cancels
    speaker→mic feedback automatically.  No more half-duplex mute.

The audio-state FSM keeps the same shape it had before, plus the
``SPEAKING`` state which is now actually used:

    LISTENING_WAKE → (wake | ptt_start)         → RECORDING
    RECORDING      → (VAD-end | ptt_end | utt)  → PROCESSING
    PROCESSING     → (pipeline done)            → SPEAKING
    SPEAKING       → (TTS drained)              → CONTINUATION
    SPEAKING       → (user barges in)           → RECORDING (with seed audio)
    CONTINUATION   → (speech + VAD-end)         → PROCESSING
    CONTINUATION   → (wake during this window)  → RECORDING (history cleared)
    CONTINUATION   → (10 s of silence)          → LISTENING_WAKE

Voice barge-in is now real full-duplex: during SPEAKING the mic stays
open, a dedicated VAD watches for user speech, and on a confident hit
we cancel the TTS playback and roll into RECORDING with the new audio
already seeded.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

from . import registry, tts
from .i18n import t
from .pipeline import Pipeline, PipelineHooks, PipelineOutcome
from .storage import (
    get_missed_reminders,
    get_recent_history,
    mark_reminder_delivered,
    start_session,
)
from .vad import VadEndpointer
from .webrtc import RtcSession

log = logging.getLogger(__name__)

CONTINUATION_TIMEOUT_S = float(os.environ.get("CONTINUATION_TIMEOUT_S", "10"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "10"))


# profile_id → list of active Session objects.  Sessions register
# themselves whenever a profile becomes known (cookie auth on the WS
# handshake, voice-side passphrase success, or anywhere else we land
# on a profile_id).  Pipeline.py reaches into this to push
# voicemail-arrived events live to the recipient's open tab.
# Module-level — there's exactly one Session table per process, and
# voicemail-save is process-local too.  A list (not set) because a
# user could have two tabs open; each gets its own ping.
_SESSIONS_BY_PROFILE: dict[int, list["Session"]] = {}


def _register_session_profile(profile_id: int, session: "Session") -> None:
    """Add ``session`` to the registry under ``profile_id``.

    Idempotent — a session that calls this twice (e.g. cookie + voice
    both resolve the same profile) only ends up in the list once.
    """
    bucket = _SESSIONS_BY_PROFILE.setdefault(profile_id, [])
    if session not in bucket:
        bucket.append(session)
        log.info(
            "ws registry: session %d registered under profile=%d (n=%d)",
            session.session_id, profile_id, len(bucket),
        )


def _unregister_session_profile(profile_id: int, session: "Session") -> None:
    """Drop ``session`` from the registry.  Safe to call multiple times."""
    bucket = _SESSIONS_BY_PROFILE.get(profile_id)
    if not bucket:
        return
    try:
        bucket.remove(session)
    except ValueError:
        return
    if not bucket:
        _SESSIONS_BY_PROFILE.pop(profile_id, None)
    log.info(
        "ws registry: session %d unregistered from profile=%d (n=%d)",
        session.session_id, profile_id, len(bucket),
    )


async def notify_voicemail_arrived(
    *,
    to_profile_id: int,
    message_id: int,
    from_name: str | None,
    duration_ms: int,
) -> int:
    """Push a ``voicemail_arrived`` event to every active session for this profile.

    Returns the number of sessions notified.  Best-effort — a failed
    ``send_json`` (closed socket etc.) is swallowed because the WS
    cleanup will deregister the stale session shortly.

    Additive Web Push step at the end: fan out a closed-browser ping to
    every PushSubscription the recipient has registered.  Push failures
    are swallowed — they shouldn't impact the live WS notification or
    the rest of the pipeline turn.
    """
    sessions = list(_SESSIONS_BY_PROFILE.get(to_profile_id, ()))
    payload = {
        "type": "voicemail_arrived",
        "message_id": message_id,
        "from_name": from_name,
        "duration_ms": duration_ms,
    }
    n = 0
    for s in sessions:
        try:
            await s._send(payload)
            n += 1
        except Exception:
            log.debug(
                "voicemail_arrived: send to session %d failed (likely closed)",
                s.session_id,
                exc_info=True,
            )
    # Closed-browser delivery via Web Push.  We resolve the recipient's
    # language for the localised body — read_settings is a tiny per-user
    # JSON file, cheap to read on demand.  ``from_name`` falls back to
    # the localised ``inbox.guest_sender`` when the voice didn't match.
    try:
        from . import push  # local import to keep ws.py import-cheap
        from .i18n import t
        from .user_files import read_settings

        settings = await read_settings(to_profile_id)
        # ``settings.language`` may be "auto" — pick_lang would resolve
        # from Whisper's detected lang, but we don't have that here.
        # Default to "en" (catalog fallback) when auto, or honour an
        # explicitly set lang.
        lang = settings.language if settings.language in ("en", "ru", "de") else "en"
        display_name = from_name or t("inbox.guest_sender", lang)
        push_payload = {
            "title": t("push.voicemail_title", lang),
            "body": t("push.voicemail_body", lang, from_name=display_name),
            "voicemail_id": message_id,
            "tag": f"voicemail-{message_id}",
        }
        sent = await push.send_to_profile(to_profile_id, push_payload)
        if sent:
            log.info(
                "voicemail_arrived: push delivered to %d subscription(s) for profile=%d",
                sent, to_profile_id,
            )
    except Exception:
        log.warning(
            "voicemail_arrived: push fan-out failed for profile=%d",
            to_profile_id, exc_info=True,
        )
    return n

# Cheap script-based guess for which TTS voice to use, based on the
# assistant's response text.  Kept here (not in tts.py) so we can log
# the decision per-turn alongside the rest of the session state.
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_UMLAUT_RE = re.compile(r"[äöüÄÖÜß]")


def _guess_response_lang(text: str) -> str:
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _UMLAUT_RE.search(text):
        return "de"
    return "en"


class State(str, Enum):
    LISTENING_WAKE = "listening_wake"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    CONTINUATION = "continuation"


class Session(PipelineHooks):
    """One WebSocket connection + its peer RTC + its state machine."""

    def __init__(
        self,
        ws: WebSocket,
        session_id: int,
        client_id: str | None = None,
        history: list[dict] | None = None,
    ):
        self.ws = ws
        self.client_id = client_id
        self.session_id = session_id
        self.history: list[dict] = list(history) if history else []
        self.last_response_text: str = ""
        self.last_response_lang: str = "en"
        self._pipeline = Pipeline(hooks=self)

        # WebRTC peer for this client.  Created lazily on the first
        # `webrtc_offer` message — we keep the WS connect cheap so a
        # client that never starts media doesn't allocate aiortc state.
        self.rtc: RtcSession | None = None

        # ---- Audio FSM state ----
        self.vad: VadEndpointer | None = None
        self.state = State.LISTENING_WAKE
        self.cmd_audio = bytearray()
        # PTT keeps the mic open across silences (VAD endpoint is ignored).
        self.ptt_holding = False
        # Server-side flag: True between the moment we start pushing TTS
        # into the outbound track and the moment that track has drained.
        # Used to gate barge-VAD lifecycle and continuation timers.
        self.tts_playing = False
        # Barge-VAD: dedicated VAD instance + buffer; alive during
        # PROCESSING (catches utterance extensions) and SPEAKING (catches
        # voice barge-in).  See handle_binary's two interrupt branches.
        self._barge_vad: VadEndpointer | None = None
        self._barge_buf = bytearray()
        # Min new speech to count as an interrupt vs. background noise /
        # short echo blips.  Higher = less twitchy, lower = more responsive.
        self._barge_min_speech_s = 0.4
        # Inflight pipeline audio kept for extension concat (PROCESSING).
        self._inflight_audio: bytes = b""
        self._cancel_reason: str | None = None
        # Background tasks owned by this session.
        self._pipeline_task: asyncio.Task | None = None
        self._continuation_task: asyncio.Task | None = None
        self._tts_synth_task: asyncio.Task | None = None
        self._tts_drain_task: asyncio.Task | None = None
        # True from the first on_response_chunk of a turn until the next
        # PROCESSING entry.  Lets on_response know "audio is already being
        # streamed in chunks — don't re-synth the full text".  Reset to
        # False in _set_state(PROCESSING).
        self._streaming_response: bool = False
        # Current TTS voice for this session.  Set by on_speaker_identified
        # to the recognised speaker's `tts_voice` column; None falls back
        # to the xtts-server default.  Cleared at the start of every turn
        # so a stale value from a previous speaker doesn't bleed in when
        # the new speaker can't be identified.
        self.current_tts_voice: str | None = None
        # One-shot TTS voice override (inbox_summary feature).  When
        # set, applies for the duration of a single on_response →
        # on_response_end pair, then reverts.  See ``on_response``.
        self._voice_override_active: bool = False
        self._voice_override_stash: str | None = None

        # Sprint-2 auth window.  When the speaker says a valid passphrase,
        # pipeline.run() calls on_auth_succeeded() and we stamp the
        # expiry here; tier-2 tool calls in subsequent turns within the
        # window proceed without re-prompting.  None = never authenticated
        # or window expired.  Per-session; doesn't survive WS reconnect
        # (a fresh connect must re-authenticate, which is what we want).
        self.auth_until: float | None = None
        self.auth_profile_id: int | None = None

        # Image attach.  When the user picks/drops an image in the
        # UI BEFORE speaking, the frontend sends an `attach_image` WS
        # message carrying its base64 data URL.  We pin it here; the
        # next utterance's pipeline run picks it up, short-circuits to
        # the local multimodal LLM (transcript + image → answer),
        # then clears the slot.  One image per turn — picking a
        # second one before speaking replaces the first (and the UI
        # shows that).
        self.pending_image_b64: str | None = None
        self.pending_image_name: str | None = None

    # ------------------------------------------------------------------ #
    # Pipeline hooks
    # ------------------------------------------------------------------ #

    async def on_transcript(self, *, text: str, asr_ms: int, speaker: str | None) -> None:
        await self._send({
            "type": "transcript",
            "text": text,
            "asr_ms": asr_ms,
            **({"speaker": speaker} if speaker else {}),
        })

    async def on_speaker_identified(
        self, *, name: str | None, tts_voice: str | None
    ) -> None:
        """Latched by the pipeline once it knows whose voice this turn is.

        We pin the resolved XTTS voice on the session so every TTS call
        for the rest of this turn picks it up (per-sentence streaming
        chunks, full-text fallback, replay, even reminder pushes that
        fire while this person is still in front of the mic).  ``None``
        means "couldn't identify" — fall back to the server default
        rather than re-using a previous speaker's voice.
        """
        self.current_tts_voice = tts_voice
        if name:
            log.info(
                "session %d: speaker=%r voice=%r",
                self.session_id, name, tts_voice or "<default>",
            )

    async def on_tool_called(
        self,
        *,
        name: str | None,
        args: dict | None,
        data: dict | None = None,
    ) -> None:
        # Surface multi-agent routing decision to the UI.  When a tool
        # ran against a specific desktop-agent (computer_use /
        # look_at_screen / desktop tools all stamp ``agent_id`` into
        # their result.data), we forward it as ``target_agent`` so the
        # frontend can:
        #   • show «computer_use · @macbook» on the tool line
        #   • briefly highlight the matching row in the agents panel
        # Anonymous tools (calculator, web_search, …) omit the field.
        target_agent = (data or {}).get("agent_id") if data else None
        payload: dict = {"type": "tool_called", "name": name, "args": args}
        if target_agent:
            payload["target_agent"] = target_agent
        await self._send(payload)
        # Voicemail "play original" signal: when ``inbox_read`` returns
        # the audio is on disk; the frontend pulls it from
        # /api/voicemail/<id>/audio and plays it inline after the
        # spoken intro ("Message from Alice:").  We don't push the
        # bytes over WS — binary frames are mic input only — so this
        # is a content-light control event.
        play_id = (data or {}).get("voicemail_play_id")
        if play_id:
            await self._send({
                "type": "voicemail_play",
                "message_id": int(play_id),
                "from_name": data.get("from_name"),
                "duration_ms": data.get("duration_ms"),
            })

    async def on_response(
        self,
        *,
        text: str,
        llm_ms: int,
        history_turns: int,
        tts_voice_override: str | None = None,
    ) -> None:
        # Notify the UI right away so the transcript card shows the text
        # *before* the audio finishes synthesising — feels more responsive.
        await self._send({
            "type": "response",
            "text": text,
            "llm_ms": llm_ms,
            "history_turns": history_turns,
        })
        self.last_response_text = text
        self.last_response_lang = _guess_response_lang(text)
        # Per-response TTS voice override (e.g. inbox_summary wants to
        # speak the summary in the original sender's voice).  Stash the
        # session's current voice and apply the override for THIS reply
        # only — on_response_end restores it so a subsequent turn from
        # the same speaker doesn't inherit the swap.  We don't reach
        # this branch when `_streaming_response` already kicked TTS
        # via chunks (chunk-streaming tools don't set the override).
        if tts_voice_override and tts_voice_override != self.current_tts_voice:
            self._voice_override_active = True
            self._voice_override_stash = self.current_tts_voice
            self.current_tts_voice = tts_voice_override
            log.info(
                "session %d: TTS voice overridden for this reply: %r → %r",
                self.session_id, self._voice_override_stash, tts_voice_override,
            )
        # If we already streamed sentence-by-sentence via on_response_chunk,
        # the audio is already on its way through the RTC track — don't
        # re-synth the whole thing.  The full text comes here only to
        # update the UI card and feed conversation history.
        if self._streaming_response:
            log.info(
                "session %d: on_response (full text %d chars) — TTS already "
                "streamed via chunks, skipping full-text synth",
                self.session_id, len(text),
            )
            return
        # Non-streaming tool (general_answer / replay): kick off full-text
        # synth now.  _start_speaking_then_continuation polls for first
        # chunk to land before transitioning state.
        self._tts_synth_task = asyncio.create_task(
            self._synth_for_playback(text, self.last_response_lang)
        )

    async def on_response_chunk(self, *, text: str) -> None:
        """Per-sentence text from a streaming tool (web_search).

        Streams TTS for this sentence and pushes the PCM frames into the
        outbound RTC track BEFORE returning, so the caller (web_search's
        LLM-iteration loop) naturally throttles to TTS pace.  The first
        invocation of a turn also transitions the FSM to SPEAKING so the
        UI shows the "speaking" status while the rest of the response is
        still synthesising — much more responsive than waiting for
        pipeline.run to return.
        """
        if not text or not self.rtc:
            return
        if not self._streaming_response:
            # First chunk of this turn: flip state to SPEAKING.  The
            # _start_speaking_then_continuation flow on pipeline exit
            # then just waits for the queue to drain instead of polling
            # for a first chunk.
            self._streaming_response = True
            self.tts_playing = True
            if self.state == State.PROCESSING:
                await self._set_state(State.SPEAKING)
        lang = _guess_response_lang(text)
        log.info(
            "session %d: chunk → tts (%d chars, lang=%s)",
            self.session_id, len(text), lang,
        )
        try:
            async for pcm in tts.stream(
                text,
                lang=lang,
                voice=self.current_tts_voice,
                session_id=self.client_id,
            ):
                if not self.rtc:
                    break
                await self.rtc.push_tts_pcm(pcm, tts.TTS_SAMPLE_RATE)
        except Exception:
            log.exception("session %d: chunk synth failed", self.session_id)

    async def on_response_end(self) -> None:
        # Roll back any one-shot TTS voice override applied in
        # ``on_response``.  Done AFTER the response_end event so any
        # in-flight chunk synth still uses the override voice.
        if self._voice_override_active:
            self.current_tts_voice = self._voice_override_stash
            self._voice_override_stash = None
            self._voice_override_active = False
        await self._send({"type": "response_end"})

    async def on_error(self, message: str) -> None:
        await self._send({"type": "error", "message": message})

    async def on_replay(self, mode: str) -> None:
        # Browser used to replay its cached SpeechSynthesis buffer here.
        # With server-side TTS we no longer have that cache on the
        # client, so we re-synthesise from `last_response_text` and play
        # back through the same RTC track as a fresh response.  `mode`
        # is logged for posterity; resume-from-position isn't supported
        # in v1 — both modes synth the full text.  (Adding position
        # tracking would require keeping the previous synth's sample
        # count and re-synthesising only the tail.)
        await self._send({"type": "replay", "mode": mode})
        if not self.last_response_text:
            return
        self._tts_synth_task = asyncio.create_task(
            self._synth_for_playback(self.last_response_text, self.last_response_lang)
        )

    async def on_history_reset(self) -> None:
        self.history.clear()
        await self._send({"type": "history_reset"})

    async def current_auth_until(self) -> float | None:
        """Pipeline calls this at the start of each turn to inherit any
        live auth window from previous turns.  None == not authenticated."""
        return self.auth_until

    async def cookie_profile_id(self) -> int | None:
        """Pipeline-side fallback for when voice-ID didn't match anyone:
        whatever profile the va_session cookie says is logged in.  Set
        once at WS handshake and stable for the WS lifetime.
        """
        return self.auth_profile_id

    async def on_auth_succeeded(
        self, *, profile_id: int | None, window_s: float
    ) -> None:
        """Pipeline fires this when a valid passphrase was found in this
        turn's transcript.  Stamp the expiry and notify the UI so the
        login banner / Pending tab can react.
        """
        import time as _t
        self.auth_until = _t.time() + window_s
        self._set_known_profile(profile_id)
        log.info(
            "session %d: auth window opened for profile=%s (%.0fs)",
            self.session_id, profile_id, window_s,
        )
        await self._send({
            "type": "auth",
            "status": "authenticated",
            "profile_id": profile_id,
            "expires_at": self.auth_until,
        })

    async def on_progress(self, *, step: str, detail: str | None = None) -> None:
        """Forward pipeline-stage updates to the browser.

        Frontend maps the `step` token onto a human-readable phrase
        (Russian) and shows it in the yellow PROCESSING status,
        replacing the static placeholder.  `detail` lets a tool add a
        small context bit (e.g. target language for `localize`).
        """
        payload: dict = {"type": "progress", "step": step}
        if detail:
            payload["detail"] = detail
        await self._send(payload)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _send(self, payload: dict) -> None:
        await self.ws.send_json(payload)

    def _set_known_profile(self, profile_id: int | None) -> None:
        """Single funnel for «we now know which profile owns this session».

        Both paths land here: cookie auth on WS handshake (read-only
        trust anchor) and voice-side passphrase success (5-minute
        sliding window).  We update :attr:`auth_profile_id` and
        register/deregister with the module-level ``_SESSIONS_BY_PROFILE``
        table so :func:`notify_voicemail_arrived` can push live events
        to every open tab for this profile.  Idempotent — calling with
        the same id twice is a noop; calling with a different id swaps
        the registration.  ``None`` only ever clears (we don't reset to
        anonymous mid-session today, but the path is here so a future
        logout doesn't leak a stale registration).
        """
        old = self.auth_profile_id
        if old == profile_id:
            return
        if old is not None:
            _unregister_session_profile(old, self)
        self.auth_profile_id = profile_id
        if profile_id is not None:
            _register_session_profile(profile_id, self)

    async def _synth_for_playback(self, text: str, lang: str) -> None:
        """Stream PCM chunks from xtts-server straight into the RTC track
        as they arrive — first chunk lands ~700 ms after the call, so the
        SPEAKING transition can fire almost immediately instead of waiting
        for the full synth.

        Doesn't wait for the track to drain — that's the SPEAKING phase's
        job (it polls rtc.tts_busy()).
        """
        if not self.rtc:
            log.warning("session %d: TTS skipped — no RTC", self.session_id)
            return
        sample_rate = tts.TTS_SAMPLE_RATE
        chunks_pushed = 0
        total_samples = 0
        try:
            async for chunk in tts.stream(
                text,
                lang=lang,
                voice=self.current_tts_voice,
                session_id=self.client_id,
            ):
                await self.rtc.push_tts_pcm(chunk, sample_rate)
                chunks_pushed += 1
                total_samples += chunk.size
            log.info(
                "session %d: TTS streamed %d chunks, %d samples @ %d Hz, lang=%s",
                self.session_id, chunks_pushed, total_samples, sample_rate, lang,
            )
        except Exception:
            log.exception("session %d: TTS streaming failed", self.session_id)

    async def _set_state(self, new_state: State) -> None:
        if self.state != new_state:
            log.info(
                "session %d state: %s -> %s", self.session_id, self.state, new_state
            )
            self.state = new_state
            await self._send({"type": "state", "value": new_state})
        # Per-turn streaming flag is scoped to one user turn — reset
        # whenever a fresh turn begins (entry to PROCESSING).
        if new_state == State.PROCESSING:
            self._streaming_response = False
        # Barge-VAD lives during PROCESSING (catches extensions) and
        # SPEAKING (catches voice barge-in).  Recreate it fresh each
        # transition so it doesn't carry over leftover speech samples.
        if new_state in (State.PROCESSING, State.SPEAKING):
            self._barge_vad = VadEndpointer(
                speech_threshold=0.6,
                min_speech_s=self._barge_min_speech_s,
                silence_timeout_s=999.0,
                max_record_s=999.0,
            )
            self._barge_buf = bytearray()
        else:
            self._barge_vad = None
            self._barge_buf = bytearray()

    def _begin_recording(self) -> None:
        self.cmd_audio.clear()
        self.vad = VadEndpointer()

    def _append_history(self, transcript: str, response_text: str) -> None:
        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": response_text})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def _reset_for_next_turn(self) -> None:
        self.cmd_audio.clear()
        if self.vad:
            self.vad.reset()
            self.vad = None

    def _cancel_tts(self) -> None:
        """Drop everything we're about to say; stop the drain monitor."""
        if self.rtc:
            self.rtc.cancel_tts()
        if self._tts_drain_task and not self._tts_drain_task.done():
            self._tts_drain_task.cancel()
        if self._tts_synth_task and not self._tts_synth_task.done():
            self._tts_synth_task.cancel()
        self.tts_playing = False

    async def play_notification(self, text: str) -> None:
        """Server-initiated TTS that doesn't transition the FSM.

        Used by the scheduler for reminder pushes and by the missed-
        reminder catch-up on connect.  Plays through the same outbound
        RTC track as a regular response, but doesn't enter SPEAKING /
        CONTINUATION — the conversational state machine is owned by the
        user's actual turn, not by a one-off notification.
        """
        if not text or not self.rtc:
            return
        lang = _guess_response_lang(text)
        try:
            sample_rate, pcm = await tts.synth(
                text,
                lang=lang,
                voice=self.current_tts_voice,
                session_id=self.client_id,
            )
            await self.rtc.push_tts_pcm(pcm, sample_rate)
            log.info(
                "session %d: notification audio enqueued (%d samples, lang=%s)",
                self.session_id, pcm.size, lang,
            )
        except Exception:
            log.exception("session %d: notification TTS failed", self.session_id)

    # ------------------------------------------------------------------ #
    # Mic PCM ingest (from RtcSession's on_mic_pcm callback)
    # ------------------------------------------------------------------ #

    async def on_mic_pcm(self, data: bytes) -> None:
        """Called per 20 ms frame of 16 kHz mono int16 audio from the peer.

        Same shape and cadence the binary-PCM WS path used to deliver, so
        the FSM below is the same as before — just sourced from WebRTC
        instead of binary websocket frames.
        """
        # ── PROCESSING: utterance extension (cancel + concat + re-run) ──
        if self.state == State.PROCESSING and self._barge_vad is not None:
            self._barge_buf.extend(data)
            self._barge_vad.feed(data)
            min_samples = int(self._barge_min_speech_s * 16000)
            if self._barge_vad.cumulative_speech_samples >= min_samples:
                await self._handle_extension_during_processing()
            return

        # ── SPEAKING: voice barge-in (cancel TTS, become RECORDING) ──
        if self.state == State.SPEAKING and self._barge_vad is not None:
            self._barge_buf.extend(data)
            self._barge_vad.feed(data)
            min_samples = int(self._barge_min_speech_s * 16000)
            if self._barge_vad.cumulative_speech_samples >= min_samples:
                await self._handle_barge_in_during_speaking()
            return

        # ── RECORDING / CONTINUATION: normal endpointing path ──
        if self.state in (State.RECORDING, State.CONTINUATION):
            self.cmd_audio.extend(data)
            assert self.vad is not None
            ended = self.vad.feed(data)
            if ended and not self.ptt_holding:
                log.info(
                    "session %d VAD endpoint reached (state=%s), %d bytes audio",
                    self.session_id, self.state, len(self.cmd_audio),
                )
                if self._continuation_task and not self._continuation_task.done():
                    self._continuation_task.cancel()
                await self._set_state(State.PROCESSING)
                self._pipeline_task = asyncio.create_task(self._run_pipeline())

    async def _become_recording_with_seed(
        self,
        *,
        seed_audio: bytes,
        via_label: str,
        prepend_inflight: bool,
    ) -> None:
        """Common path for "switch to RECORDING with audio already buffered".

        Both extension-during-PROCESSING and barge-in-during-SPEAKING
        end up doing the same five steps: cancel whatever was running
        (TTS + maybe pipeline + continuation timer), seed cmd_audio +
        VAD with the audio that triggered the switch, ack the client,
        flip state.  ``prepend_inflight`` differentiates the two
        cases — extension concatenates the original utterance audio
        in front of the new chunk so the LLM sees the full sentence;
        barge-in starts fresh because the previous turn already
        completed (this is an interrupt of the assistant's reply, not
        a continuation of the user's question).
        """
        # Snapshot inflight before any await — the `finally` block in
        # _run_pipeline clears it on cancel.
        original_audio = self._inflight_audio if prepend_inflight else b""

        if prepend_inflight:
            self._cancel_reason = "extension"
            if self._pipeline_task and not self._pipeline_task.done():
                self._pipeline_task.cancel()
                try:
                    await self._pipeline_task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._continuation_task and not self._continuation_task.done():
            self._continuation_task.cancel()
        self._cancel_tts()

        if original_audio:
            combined = bytearray(original_audio)
            combined.extend(seed_audio)
            self.cmd_audio = combined
        else:
            self.cmd_audio = bytearray(seed_audio)

        self.vad = VadEndpointer()
        if seed_audio:
            self.vad.feed(seed_audio)

        self._barge_buf = bytearray()
        self._barge_vad = None
        if prepend_inflight:
            self._inflight_audio = b""

        await self._send({"type": "wake_ack", "via": via_label})
        await self._set_state(State.RECORDING)

    async def _handle_extension_during_processing(self) -> None:
        """User added to the question while we were processing.

        Cancel the in-flight pipeline, glue original + extension audio
        together, restart the pipeline against the combined utterance.
        """
        log.info(
            "session %d extension: %d ms of new speech during PROCESSING — "
            "cancelling, concatenating, re-running",
            self.session_id,
            int(self._barge_vad.cumulative_speech_samples / 16) if self._barge_vad else 0,
        )
        await self._become_recording_with_seed(
            seed_audio=bytes(self._barge_buf),
            via_label="voice_extension",
            prepend_inflight=True,
        )

    async def _handle_barge_in_during_speaking(self) -> None:
        """User started talking while the assistant was speaking.

        Cut TTS playback, swap to RECORDING with the barge audio
        already in the buffer — feels like a natural interruption,
        the user doesn't need to repeat themselves.
        """
        log.info(
            "session %d barge-in: %d ms of new speech during SPEAKING — "
            "cancelling TTS, switching to RECORDING",
            self.session_id,
            int(self._barge_vad.cumulative_speech_samples / 16) if self._barge_vad else 0,
        )
        await self._become_recording_with_seed(
            seed_audio=bytes(self._barge_buf),
            via_label="voice_barge_in",
            prepend_inflight=False,
        )

    # ------------------------------------------------------------------ #
    # JSON messages
    # ------------------------------------------------------------------ #

    async def handle_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        # ── WebRTC signalling ──
        if mtype == "webrtc_offer":
            await self._handle_webrtc_offer(msg)
            return
        if mtype == "webrtc_ice":
            if self.rtc:
                await self.rtc.add_remote_ice(msg.get("candidate") or {})
            return
        # ── Voice-control messages ──
        if mtype == "wake_detected" and self.state in (
            State.LISTENING_WAKE,
            State.CONTINUATION,
            State.SPEAKING,
        ):
            if self.state == State.CONTINUATION:
                log.info(
                    "session %d: wake during continuation, clearing %d history msgs",
                    self.session_id, len(self.history),
                )
                self.history.clear()
                if self._continuation_task and not self._continuation_task.done():
                    self._continuation_task.cancel()
            elif self.state == State.SPEAKING:
                # Wake-word interrupt during TTS — equivalent to a voice
                # barge-in but explicit.  Don't clear history, the user
                # most likely wants a follow-up (e.g. "repeat" / "continue").
                log.info(
                    "session %d: wake during speaking, cancelling TTS",
                    self.session_id,
                )
                self._cancel_tts()
            score = msg.get("score")
            log.info("session %d wake from client (score=%s)", self.session_id, score)
            self._begin_recording()
            await self._set_state(State.RECORDING)
            await self._send({"type": "wake_ack"})
        elif mtype == "ptt_start":
            log.info(
                "session %d push-to-talk start (from %s)", self.session_id, self.state
            )
            if self.state == State.RECORDING:
                self.ptt_holding = True
            else:
                if self._pipeline_task and not self._pipeline_task.done():
                    self._pipeline_task.cancel()
                    try:
                        await self._pipeline_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if self._continuation_task and not self._continuation_task.done():
                    self._continuation_task.cancel()
                self._cancel_tts()
                self._begin_recording()
                self.ptt_holding = True
                await self._set_state(State.RECORDING)
                await self._send({"type": "wake_ack", "via": "ptt"})
        elif mtype == "ptt_end" and self.state == State.RECORDING:
            log.info(
                "session %d push-to-talk end, %d bytes audio",
                self.session_id, len(self.cmd_audio),
            )
            self.ptt_holding = False
            await self._set_state(State.PROCESSING)
            self._pipeline_task = asyncio.create_task(self._run_pipeline())
        elif mtype == "utterance_end" and self.state in (
            State.RECORDING,
            State.CONTINUATION,
        ):
            log.info("session %d client forced utterance_end", self.session_id)
            if self._continuation_task and not self._continuation_task.done():
                self._continuation_task.cancel()
            await self._set_state(State.PROCESSING)
            self._pipeline_task = asyncio.create_task(self._run_pipeline())
        elif mtype == "cancel":
            log.info("session %d cancel from client", self.session_id)
            if self._pipeline_task and not self._pipeline_task.done():
                self._pipeline_task.cancel()
            if self._continuation_task and not self._continuation_task.done():
                self._continuation_task.cancel()
            self._cancel_tts()
            self._reset_for_next_turn()
            await self._send({"type": "cancelled"})
            await self._set_state(State.LISTENING_WAKE)
        elif mtype == "reset_history":
            log.info(
                "session %d: client requested history reset (%d msgs)",
                self.session_id, len(self.history),
            )
            self.history.clear()
            await self._send({"type": "history_reset"})
        elif mtype == "attach_image":
            # User dropped/picked an image to ask about.  We accept either
            # a raw base64 payload or a full `data:image/...;base64,...`
            # URL — `vision.analyze_image_b64` strips the prefix either
            # way.  Cap at ~6 MB before b64-overhead so a misclick on a
            # camera RAW doesn't blow up the socket; vision quality
            # tops out long before that anyway.
            payload = msg.get("data") or ""
            name = (msg.get("name") or "image").strip() or "image"
            if not isinstance(payload, str) or not payload:
                await self._send({
                    "type": "attach_image_ack",
                    "ok": False,
                    "error": "empty_payload",
                })
            elif len(payload) > 8 * 1024 * 1024:
                # ~8 MB of b64 ≈ ~6 MB binary.
                await self._send({
                    "type": "attach_image_ack",
                    "ok": False,
                    "error": "too_large",
                })
            else:
                self.pending_image_b64 = payload
                self.pending_image_name = name
                log.info(
                    "session %d: attached image %r (%d b64 chars)",
                    self.session_id, name, len(payload),
                )
                await self._send({
                    "type": "attach_image_ack",
                    "ok": True,
                    "name": name,
                    "bytes_b64": len(payload),
                })
        elif mtype == "detach_image":
            had = self.pending_image_b64 is not None
            self.pending_image_b64 = None
            self.pending_image_name = None
            await self._send({"type": "attach_image_ack", "ok": True, "cleared": had})
        else:
            log.debug("ignored msg %r in state %s", mtype, self.state)

    async def _handle_webrtc_offer(self, msg: dict) -> None:
        """Apply the browser's SDP offer; reply with an SDP answer.

        Creates the RtcSession on first call (subsequent offers reuse
        the existing one — supports renegotiation if we ever need it).
        """
        if self.rtc is None:
            self.rtc = RtcSession(
                on_mic_pcm=self.on_mic_pcm,
                send_signal=self._send,
                session_id=self.session_id,
            )
        sdp = msg.get("sdp") or ""
        sdp_type = msg.get("sdp_type") or msg.get("type_sdp") or "offer"
        log.info(
            "session %d: webrtc_offer (%d bytes SDP)", self.session_id, len(sdp)
        )
        answer_sdp, answer_type = await self.rtc.set_remote_offer(sdp, sdp_type)
        await self._send({
            "type": "webrtc_answer",
            "sdp": answer_sdp,
            "sdp_type": answer_type,
        })

    # ------------------------------------------------------------------ #
    # Timers / transitions
    # ------------------------------------------------------------------ #

    async def _start_speaking_then_continuation(self) -> None:
        """After pipeline completes: SPEAKING (play TTS) → CONTINUATION.

        Two paths converge here:

          • Streaming (web_search): on_response_chunk has been pushing
            audio into the RTC track sentence-by-sentence as the LLM
            generated tokens.  By the time we get here, ALL sentences
            have been synthesised and pushed (web_search returned).
            State was already flipped to SPEAKING on the first chunk.
            We just wait for the track to play out.

          • Non-streaming (general_answer, replay): on_response kicked
            off a full-text synth task in the background.  Wait for
            its first chunk to land in the track, transition to
            SPEAKING, then wait for the track to drain.
        """
        # If chunks already flipped us to SPEAKING, skip the
        # first-chunk poll and go straight to draining.
        if self.state != State.SPEAKING and self._tts_synth_task:
            first_chunk_deadline = asyncio.get_event_loop().time() + 30.0
            while asyncio.get_event_loop().time() < first_chunk_deadline:
                if self.rtc and self.rtc.tts_busy():
                    break
                if self._tts_synth_task.done():
                    break
                await asyncio.sleep(0.05)

        has_audio = (
            self.state == State.SPEAKING
            or bool(self.rtc and self.rtc.tts_busy())
        )
        if has_audio:
            if self.state != State.SPEAKING:
                self.tts_playing = True
                await self._set_state(State.SPEAKING)
            self._tts_drain_task = asyncio.create_task(self._wait_for_tts_drain())
            try:
                await self._tts_drain_task
            except asyncio.CancelledError:
                # Barge-in / cancel got us; don't enter continuation.
                return
        self._tts_synth_task = None
        self.tts_playing = False
        await self._enter_continuation()

    async def _wait_for_tts_drain(self) -> None:
        """Resolve once synth has emitted all chunks AND the track is empty.

        Two phases:
          1. Synth still running → chunks may keep arriving; track being
             non-empty doesn't mean playback is finishing, just that we
             have audio queued.  We must wait for the synth task itself.
          2. Synth done → poll until rtc.tts_busy() returns False.
        """
        # Phase 1: let synth finish enqueuing all chunks.
        if self._tts_synth_task and not self._tts_synth_task.done():
            try:
                await self._tts_synth_task
            except Exception:
                log.exception("session %d: TTS synth task failed", self.session_id)
        if not self.rtc:
            return
        # Phase 2: wait for playback to consume the queue.  Poll cheaply.
        while self.rtc.tts_busy():
            await asyncio.sleep(0.05)
        log.info("session %d: TTS drained", self.session_id)

    async def _enter_continuation(self) -> None:
        """Open the follow-up window with a CONTINUATION_TIMEOUT_S countdown."""
        self.cmd_audio.clear()
        self.vad = VadEndpointer()
        self._barge_buf = bytearray()
        self._barge_vad = None
        log.info(
            "session %d: entering continuation, %.0fs window",
            self.session_id, CONTINUATION_TIMEOUT_S,
        )
        await self._set_state(State.CONTINUATION)
        if self._continuation_task and not self._continuation_task.done():
            self._continuation_task.cancel()
        self._continuation_task = asyncio.create_task(self._continuation_timeout())

    async def _continuation_timeout(self) -> None:
        try:
            await asyncio.sleep(CONTINUATION_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        if self.state == State.CONTINUATION:
            log.info(
                "session %d continuation timeout — no follow-up in %.0fs, returning to wake",
                self.session_id, CONTINUATION_TIMEOUT_S,
            )
            self._reset_for_next_turn()
            try:
                await self._set_state(State.LISTENING_WAKE)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Pipeline orchestration
    # ------------------------------------------------------------------ #

    async def _run_pipeline(self) -> None:
        PIPELINE_TIMEOUT_S = 180.0
        outcome: PipelineOutcome | None = None
        self._inflight_audio = bytes(self.cmd_audio)
        self._cancel_reason = None
        # Snapshot the attached image (if any) for THIS turn and clear
        # the slot immediately — even if the pipeline times out we
        # don't want the image to silently apply to the next utterance.
        # Pipeline.run() reads from this snapshot, not from session
        # state, so a concurrent detach won't race the in-flight call.
        attached_image_b64 = self.pending_image_b64
        attached_image_name = self.pending_image_name
        self.pending_image_b64 = None
        self.pending_image_name = None
        if attached_image_b64:
            log.info(
                "session %d: pipeline run consuming attached image %r",
                self.session_id, attached_image_name,
            )
        try:
            outcome = await asyncio.wait_for(
                self._pipeline.run(
                    audio=self._inflight_audio,
                    session_id=self.session_id,
                    client_id=self.client_id,
                    history=self.history,
                    last_response_text=self.last_response_text,
                    attached_image_b64=attached_image_b64,
                ),
                timeout=PIPELINE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "session %d pipeline timeout (%ss)", self.session_id, PIPELINE_TIMEOUT_S
            )
            try:
                # No AgentContext to read lang from on a pipeline
                # timeout (we never got an outcome back).  Fall back to
                # the language guess we ran on the previous reply —
                # English if this is the first turn of the session.
                await self._send({
                    "type": "response",
                    "text": t("pipeline.thinking_too_long", self.last_response_lang),
                })
                await self._send({"type": "response_end"})
            except Exception:
                pass
        except asyncio.CancelledError:
            raise
        finally:
            if outcome is not None:
                if outcome.history_cleared:
                    self.history.clear()
                elif outcome.history_appended and outcome.transcript and outcome.response_text:
                    self._append_history(outcome.transcript, outcome.response_text)
                if outcome.last_response_text:
                    self.last_response_text = outcome.last_response_text
            self._inflight_audio = b""
            self._reset_for_next_turn()
            # In the extension case the caller immediately re-enters
            # RECORDING; skip the speaking-then-continuation transition.
            if self._cancel_reason != "extension":
                try:
                    await self._start_speaking_then_continuation()
                except Exception:
                    log.exception("session %d: post-pipeline transition failed", self.session_id)


# ---------------------------------------------------------------------- #
# Top-level WS endpoint
# ---------------------------------------------------------------------- #


async def _keepalive(ws: WebSocket, interval_s: float = 25.0) -> None:
    try:
        while True:
            await asyncio.sleep(interval_s)
            try:
                await ws.send_json({"type": "ping", "ts": time.time()})
            except Exception:
                return
    except asyncio.CancelledError:
        return


async def handle_ws(ws: WebSocket, client_id: str | None = None) -> None:
    await ws.accept()
    session_id = await start_session(
        client=ws.client.host if ws.client else None,
        client_id=client_id,
    )

    initial_history: list[dict] = []
    if client_id:
        initial_history = await get_recent_history(client_id, max_turns=MAX_HISTORY_TURNS)
        if initial_history:
            log.info(
                "ws session %d: restored %d history messages for client %.8s…",
                session_id, len(initial_history), client_id,
            )

    session = Session(ws, session_id, client_id=client_id, history=initial_history)

    # Sprint-2: if the browser sent the auth cookie on the WS handshake,
    # resolve it server-side and pin the profile_id on the session
    # *before* the first voice turn lands.  This means a UI-logged-in
    # user gets tier-2 capability from the very first utterance, even
    # if resemblyzer hasn't matched their voice yet.  Voice-side auth
    # (passphrase spoken aloud) still works in parallel and bumps the
    # 5-minute auth window — they're not mutually exclusive.
    cookie_header = ws.headers.get("cookie") or ""
    cookie_token: str | None = None
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == "va_session" and value:
            cookie_token = value
            break
    if cookie_token:
        try:
            from .storage import get_session as _get_auth_session
            sess = await _get_auth_session(cookie_token)
            if sess:
                # Open a long auth window for the cookie's lifetime —
                # the cookie is the durable trust anchor; voice-side
                # auth would only ever shorten that, not extend it.
                session.auth_until = sess["expires_at"]
                # Route through _set_known_profile so the session lands
                # in the live-notify registry from the very first
                # handshake — a cookie-authenticated tab receives
                # ``voicemail_arrived`` events even before the user
                # opens the mic.
                session._set_known_profile(sess["profile_id"])
                log.info(
                    "ws session %d: cookie auth → profile=%d (expires %.0f)",
                    session_id, sess["profile_id"], sess["expires_at"],
                )
        except Exception:
            log.exception("ws session %d: cookie resolution failed", session_id)

    if client_id:
        registry.register(client_id, session)

    log.info(
        "ws connected: session_id=%d client_id=%s history=%d",
        session_id, (client_id or "")[:8] or "—", len(initial_history),
    )
    await session._send({
        "type": "ready",
        "state": session.state,
        "session_id": session_id,
        "history_turns": len(initial_history) // 2,
    })

    if client_id:
        try:
            missed = await get_missed_reminders(client_id)
            for reminder_id, push_text in missed:
                # Missed reminders are server-initiated TTS — they
                # play through the RTC track but don't enter SPEAKING /
                # CONTINUATION (no conversational turn behind them).
                await session._send(
                    {"type": "push_tts", "text": push_text, "reason": "missed_reminder"}
                )
                await session.play_notification(push_text)
                await mark_reminder_delivered(reminder_id)
                log.info("delivered missed reminder %d to %.8s…", reminder_id, client_id)
        except Exception:
            log.exception("failed to deliver missed reminders")

    keepalive_task = asyncio.create_task(_keepalive(ws))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            # We don't accept binary frames any more — audio comes via WebRTC.
            if msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    log.warning("non-JSON text frame ignored")
                    continue
                await session.handle_message(payload)
            elif msg.get("bytes") is not None:
                log.debug(
                    "session %d: ignoring unexpected binary frame (%d bytes) — "
                    "audio is on WebRTC now",
                    session_id, len(msg["bytes"]),
                )
    except WebSocketDisconnect:
        pass
    finally:
        if client_id:
            registry.unregister(client_id)
        # Clean up the by-profile registry so a closed tab stops
        # collecting ``voicemail_arrived`` events — see
        # :func:`notify_voicemail_arrived`.  ``_set_known_profile(None)``
        # is the canonical path and is safe when no profile was ever set.
        session._set_known_profile(None)
        keepalive_task.cancel()
        if session._pipeline_task and not session._pipeline_task.done():
            session._pipeline_task.cancel()
        if session._continuation_task and not session._continuation_task.done():
            session._continuation_task.cancel()
        if session._tts_drain_task and not session._tts_drain_task.done():
            session._tts_drain_task.cancel()
        if session.rtc:
            await session.rtc.close()
        log.info(
            "ws closed: session_id=%d (final state=%s, %d history msgs)",
            session_id, session.state, len(session.history),
        )
