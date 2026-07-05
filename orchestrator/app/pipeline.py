"""
Audio-in → text-out turn pipeline, decoupled from the WebSocket layer.

`Pipeline.run()` takes a finished utterance (raw PCM bytes) plus session
state and walks the standard chain:

    ASR + speaker ID (in parallel)
    │
    ├─ local intent? (replay / new_topic) — short-circuit
    │
    └─ build memory_context (speaker tag + semantic search)
       │
       └─ run_agent() — multi-step LLM with tool use
          │
          └─ persist utterance + fire-and-forget embedding

All side effects (sending WS messages, recording into the DB) happen
through `PipelineHooks` so the pipeline itself is testable with mock
hooks and no WebSocket / no client.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import i18n, memory, speaker, vision
from .agent import AgentContext, AgentResult, run_agent
from .asr import transcribe
from .i18n import pick_lang, t
from .intents import match_intent
from .user_files import read_settings
from .storage import (
    get_candidate_utterances,
    get_speaker_profiles,
    list_pending_actions,
    list_unseen_replies_for_sender,
    mark_voicemail_reply_delivered,
    save_utterance,
    save_voice_message,
)
from .user_files import verify_passphrase

log = logging.getLogger(__name__)


SAMPLE_BYTES_PER_SECOND = 16_000 * 2  # 16 kHz × int16
MIN_DURATION_MS = 200

# How long a successful passphrase keeps the auth window open.  Sliding
# — every successful tier-2 action does NOT extend it (so a child can't
# binge through approvals after one accidental disclosure); only saying
# the passphrase again resets the clock.
AUTH_WINDOW_S = 5 * 60

# Passphrase prefix patterns live in every locale's ``intents.passphrase_prefix``
# JSON section.  We try all locales so a Russian-locale user can say
# "password: …" if they're more comfortable in English — and a new
# locale automatically gets passphrase recognition by adding its
# prefixes to the JSON.


def _extract_voicemail(transcript: str) -> tuple[str, str] | None:
    """Match a leading "tell <name> that ..." / "leave a message for <name>" pattern.

    Returns ``(recipient_name, body)`` or ``None`` if no pattern fires.
    Patterns live in every locale's JSON under ``intents.voicemail`` —
    a household member can address a message in any supported language
    regardless of their session locale, so we check all locales (via
    :func:`i18n.patterns_for_intent`).  Adding a new language means
    adding the corresponding voicemail patterns to its locale JSON;
    no edit here.
    """
    for pat in i18n.patterns_for_intent("voicemail"):
        m = pat.match(transcript)
        if m:
            to = m.group("to").strip().strip(".,!?;:")
            body = m.group("body").strip().strip(" .,!?;:")
            if to and body:
                return to, body
    return None


def _resolve_voicemail_recipient(
    name_spoken: str, profiles_raw: list[tuple]
) -> tuple[int, str] | None:
    """Match the spoken recipient name against enrolled household profiles.

    Two tiers:
      1. Case-insensitive exact match.
      2. Bidirectional 3-char prefix match — absorbs minor inflections
         and abbreviations across languages.

    Three chars is the empirical sweet spot: shorter collides on common
    initials, longer misses inflected forms.  Ties at Tier 2 are broken
    by shortest enrolled name (closest to the stem).

    ``profiles_raw`` rows are ``(id, name, embedding, sample_count, tts_voice)``
    — the shape :func:`get_speaker_profiles` returns.
    """
    if not name_spoken:
        return None
    needle = name_spoken.lower()
    # Tier 1 — exact (case-insensitive)
    for row in profiles_raw:
        pid, pname = row[0], row[1]
        if pname.lower() == needle:
            return pid, pname
    # Tier 2 — bidirectional 3-char stem prefix
    if len(needle) >= 3:
        stem_needle = needle[:3]
        candidates: list[tuple[int, str]] = []
        for row in profiles_raw:
            pid, pname = row[0], row[1]
            stem_p = pname.lower()[:3]
            if pname.lower().startswith(stem_needle) or needle.startswith(stem_p):
                candidates.append((pid, pname))
        if candidates:
            candidates.sort(key=lambda x: len(x[1]))
            return candidates[0]
    return None


def _extract_passphrase(transcript: str) -> tuple[str | None, str]:
    """Pull a leading "passphrase <X>" off the transcript.

    Returns ``(passphrase, remainder)``.  ``passphrase`` is None when
    nothing matched — the transcript is returned verbatim then.  When
    a passphrase is found, only the FIRST whitespace-delimited token
    after the keyword is taken as the passphrase; the rest of the
    sentence is the real intent (so "passphrase amber delete record X"
    becomes (passphrase=amber, remainder="delete record X")).

    Patterns come from every locale's ``intents.passphrase_prefix`` —
    a household member can say the passphrase keyword in any supported
    language regardless of session locale.
    """
    for pat in i18n.patterns_for_intent("passphrase_prefix"):
        m = pat.match(transcript)
        if m:
            phrase = m.group("phrase").strip().strip(".,!?;:")
            remainder = transcript[m.end():].strip()
            return phrase, remainder
    return None, transcript


class PipelineHooks(Protocol):
    """
    What the pipeline can tell the outside world. The WS layer implements
    these as `await self._send(...)`. A test can implement them as plain
    list appenders.
    """

    async def on_transcript(self, *, text: str, asr_ms: int, speaker: str | None) -> None: ...
    async def on_speaker_identified(
        self, *, name: str | None, tts_voice: str | None
    ) -> None: ...
    async def on_tool_called(
        self,
        *,
        name: str | None,
        args: dict | None,
        data: dict | None = None,
    ) -> None: ...
    async def on_response(
        self,
        *,
        text: str,
        llm_ms: int,
        history_turns: int,
        tts_voice_override: str | None = None,
    ) -> None: ...
    async def on_response_end(self) -> None: ...
    async def on_error(self, message: str) -> None: ...
    async def on_replay(self, mode: str) -> None: ...
    async def on_history_reset(self) -> None: ...

    # Optional incremental hook — fires per-sentence as a streaming
    # response is generated.  Tools that produce text via chat_stream
    # (currently just web_search) call this so TTS can start mid-LLM-
    # generation, dropping time-to-first-audio from ~5 s to ~1.5 s.
    # Default no-op on hosts that don't override it.
    async def on_response_chunk(self, *, text: str) -> None: ...

    # Optional pipeline-stage progress signal.  Fires throughout the
    # yellow (PROCESSING) phase as the orchestrator moves between
    # stages — ASR → agent → tool → summarisation.  The UI uses this
    # to show the user *what* is happening instead of a static
    # "thinking…" placeholder.  Default no-op.
    async def on_progress(self, *, step: str, detail: str | None = None) -> None: ...

    # Optional media-stream notification.  A live-streaming tool
    # (stream_camera, stream_tab) calls this to tell the WS layer that
    # a MJPEG stream has started so the frontend can render an <img>.
    # ``url`` is the relative orchestrator URL; ``source`` is a short
    # label ("camera" / "tab") for the UI.  Default no-op.
    async def on_media_started(self, *, url: str, source: str) -> None: ...

    # Sprint-2 auth callbacks.  Optional; sessions without a passphrase
    # mechanism (the /dev/respond HTTP path) implement them as no-ops.
    #
    # ``current_auth_until`` is READ at the start of every turn; the
    # pipeline uses it to compute ``is_authenticated`` for AgentContext.
    # ``on_auth_succeeded`` is CALLED when a valid passphrase was found
    # in the transcript; the session implementation refreshes its
    # auth-window expiry and (later) tells the UI.
    # ``cookie_profile_id`` is the UI-logged-in profile id (from the
    # va_session cookie set on the WS handshake) — used as a fallback
    # when voice identification didn't match anyone in this turn.
    async def current_auth_until(self) -> float | None: ...
    async def on_auth_succeeded(
        self, *, profile_id: int | None, window_s: float
    ) -> None: ...
    async def cookie_profile_id(self) -> int | None: ...
    # Step-up auth (#55).  Returns the expiry timestamp of the active
    # step-up grant for this session, or None if no grant is live.
    async def current_step_up_auth_until(self) -> float | None: ...


@dataclass
class PipelineOutcome:
    """What the WS layer needs back from the pipeline to update Session state."""
    transcript: str | None
    response_text: str | None
    history_appended: bool  # whether (user, assistant) was added to history
    history_cleared: bool   # whether we explicitly wiped history
    last_response_text: str | None  # for the "repeat" intent's cached buffer
    # Carries a TTS voice override propagated from the terminal tool
    # (currently only `inbox_summary`, so a voicemail summary plays in
    # the message author's voice).  Plumbed into ``on_response`` via
    # the WS layer's hook; no other branch sets it today.
    tts_voice_override: str | None = None


class Pipeline:
    """
    Stateless turn processor. One instance per Session is fine; instances
    don't hold per-turn state — everything flows through `run()` args.
    """

    def __init__(self, hooks: PipelineHooks):
        self._hooks = hooks

    async def run(
        self,
        *,
        audio: bytes,
        session_id: int,
        client_id: str | None,
        history: list[dict],
        last_response_text: str,
        attached_image_b64: str | None = None,
        device_kind: str | None = None,
    ) -> PipelineOutcome:
        duration_ms = int(len(audio) / SAMPLE_BYTES_PER_SECOND * 1000)
        record: dict = {
            "session_id": session_id,
            "ts": time.time(),
            "audio_duration_ms": duration_ms,
            "transcript": None,
            "asr_ms": None,
            "llm_ms": None,
            "tool_name": None,
            "tool_args": None,
            "response_text": None,
            "error": None,
            "speaker_name": None,
        }
        outcome = PipelineOutcome(
            transcript=None,
            response_text=None,
            history_appended=False,
            history_cleared=False,
            last_response_text=last_response_text,
        )
        record_id: int | None = None

        # User language for THIS turn.  Stays None until we resolve the
        # speaker's profile_id below (after ASR + voice ID).  All t()
        # calls before that point fall back to English via the i18n
        # layer's default, which is intentional — we have no other
        # signal to pick a language from for "too short" / failed-ASR
        # branches.  Updated once profile_id is known.
        user_lang: str | None = None

        try:
            if duration_ms < MIN_DURATION_MS:
                log.info("session %d skipped: too short (%dms)", session_id, duration_ms)
                msg = t("pipeline.not_heard", user_lang)
                await self._hooks.on_response(
                    text=msg, llm_ms=0, history_turns=len(history) // 2
                )
                record["error"] = "too_short"
                record["response_text"] = msg
                return outcome

            # ASR + speaker embedding in parallel — speaker.encode_audio is
            # CPU-bound (~20–60 ms) and overlaps with the Whisper HTTP call
            # (~200 ms–5 s), so total latency stays at max(asr, embed).
            spk_task = (
                asyncio.create_task(speaker.encode_audio_full(audio))
                if speaker.SPEAKER_ENABLED and client_id
                else None
            )

            await self._hooks.on_progress(step="asr")
            # No `language=...` lock — Whisper auto-detects per utterance
            # so a Russian-speaking user living in Germany can switch into
            # German mid-conversation and get the right transcription
            # (and downstream the right TTS voice, since the frontend
            # picks `utterance.lang` from the response text).
            # Noise reduction is handled upstream — the browser's
            # getUserMedia constraints (`noiseSuppression: true` + AEC3)
            # do a better job than a Python pre-filter inside the
            # orchestrator, and don't add a ~120 ms latency hop on the
            # critical ASR path.
            transcript, asr_ms = await transcribe(audio)
            record["transcript"] = transcript
            record["asr_ms"] = asr_ms
            outcome.transcript = transcript

            # Latency-win: kick off the memory query embedding the moment
            # we have the transcript.  The embedding call (LM Studio,
            # ~80-150 ms) then runs in parallel with speaker resolution,
            # passphrase verification, and intent matching — all of
            # which together take 30-100 ms on a typical turn.  By the
            # time _build_memory_context awaits the result, it's
            # usually ready, and we shave 100-200 ms off the critical
            # path on every memory-enabled turn.
            #
            # We don't kick this off when memory is disabled (no
            # embedding model, no client_id) — the task would just
            # raise immediately.
            embed_task: asyncio.Task | None = None
            if memory.EMBEDDING_ENABLED and client_id and transcript:
                embed_task = asyncio.create_task(memory.embed_query(transcript))

            attr = await self._resolve_speaker(spk_task, client_id, session_id)
            speaker_name = attr.name
            speaker_voice = attr.tts_voice
            profile_id = attr.profile_id
            # Cookie fallback: a UI-logged-in user gets tier-2 ability
            # even if resemblyzer hasn't matched their voice yet.  Only
            # used to *fill in* a missing profile_id — never to override
            # a voice-recognised match (which is the stronger signal).
            # Muted on mixed turns (#59): silently attributing a
            # two-voice utterance to whoever owns the browser session is
            # exactly the misattribution this policy exists to kill.
            if profile_id is None and not attr.is_mixed:
                profile_id = await self._hooks.cookie_profile_id()
                if profile_id is not None:
                    log.info(
                        "session %d: profile %d inherited from cookie (no voice match)",
                        session_id, profile_id,
                    )
            record["speaker_name"] = speaker_name

            # Resolve user-facing language for THIS turn now that we
            # know the speaker.  Drives every t() call below — agent
            # context, voicemail confirmations, intent acks, empty
            # transcript replies.  Falls back to English when no
            # profile or settings read fails.
            if profile_id is not None:
                try:
                    settings = await read_settings(profile_id)
                    user_lang = pick_lang(
                        settings_lang=settings.language if settings.language != "auto" else None,
                        detected_lang=None,
                    )
                except Exception:
                    log.warning("pipeline: settings read failed for lang", exc_info=True)

            # Tell the session who's talking BEFORE the transcript event
            # — so the WS layer has the right tts_voice latched in time
            # for the very first streaming TTS chunk of this turn.
            await self._hooks.on_speaker_identified(
                name=speaker_name, tts_voice=speaker_voice
            )
            await self._hooks.on_transcript(
                text=transcript, asr_ms=asr_ms, speaker=speaker_name
            )

            if not transcript:
                msg = t("pipeline.empty_transcript", user_lang)
                await self._hooks.on_response(
                    text=msg, llm_ms=0, history_turns=len(history) // 2
                )
                record["error"] = "empty_transcript"
                record["response_text"] = msg
                return outcome

            # Reply-replay: if the recognised speaker SENT voicemail(s)
            # that the recipient has since replied to, surface those
            # replies on the speaker's NEXT utterance.  Builds the
            # context block here (and stamps the rows delivered so we
            # don't re-surface) so it's ready to splice into
            # ``memory_context`` later — both branches that hit the
            # agent loop (vision short-circuit doesn't, voicemail
            # short-circuit doesn't) pick it up.  Done BEFORE the
            # voicemail-leave branch on purpose: even when the speaker
            # is leaving a fresh voicemail, the previous reply has
            # already been delivered conceptually (they spoke), so we
            # mark it consumed and move on.  No ack is generated here
            # — the LLM is responsible for naturally folding the lines
            # into its next reply.
            unseen_replies_block = ""
            if profile_id is not None:
                try:
                    unseen = await list_unseen_replies_for_sender(profile_id)
                    if unseen:
                        lines = []
                        for r in unseen:
                            lines.append(f"{r['to_name']}: «{r['reply_text']}»")
                            await mark_voicemail_reply_delivered(r['id'])
                        unseen_replies_block = (
                            "\n[Pending replies to messages this speaker sent — "
                            "surface them naturally in your reply, prefixed with "
                            "'by the way' (or its locale equivalent):]\n"
                            + "\n".join(lines)
                        )
                        log.info(
                            "session %d: surfacing %d voicemail reply(ies) to sender",
                            session_id, len(unseen),
                        )
                except Exception:
                    log.warning("voicemail: reply-replay query failed", exc_info=True)

            # 0a. Voicemail leave — pattern-matched before passphrase /
            # intents so a guest saying "tell <Name> that I'll be late"
            # bypasses the rest of the pipeline entirely (no LLM, no
            # tool loop, no history mutation).  We save the FULL audio
            # of the turn (so the recipient hears the actual voice with
            # tone preserved) and the body as the transcript.
            #
            # Resolution requires this client to have at least one
            # enrolled speaker_profile that matches the spoken name —
            # otherwise we fall through to the normal agent loop with
            # an explanation, on the theory that the user might have
            # meant something other than voicemail ("pass me the salt").
            vm = _extract_voicemail(transcript)
            if vm is not None and client_id:
                spoken_to, body = vm
                # Voicemail confirmation language: ``user_lang`` was
                # resolved from the sender's profile.settings above;
                # no per-branch re-read needed here.
                profiles_raw = await get_speaker_profiles(client_id)
                match = _resolve_voicemail_recipient(spoken_to, profiles_raw)
                if match is None:
                    log.info(
                        "session %d: voicemail recipient %r unknown",
                        session_id, spoken_to,
                    )
                    ack = t(
                        "voicemail.unknown_recipient",
                        user_lang,
                        to=spoken_to,
                    )
                    await self._hooks.on_tool_called(
                        name="leave_message",
                        args={"to": spoken_to, "status": "unknown_recipient"},
                    )
                    await self._hooks.on_response(
                        text=ack, llm_ms=0, history_turns=len(history) // 2
                    )
                    record["tool_name"] = "leave_message"
                    record["response_text"] = ack
                    outcome.response_text = ack
                    outcome.last_response_text = ack
                    return outcome
                to_profile_id, to_name = match
                try:
                    vm_id = await save_voice_message(
                        from_profile_id=profile_id,
                        from_name=speaker_name,
                        to_profile_id=to_profile_id,
                        to_name=to_name,
                        transcript=body,
                        duration_ms=duration_ms,
                        audio_pcm=audio,
                    )
                except Exception as exc:
                    log.exception("voicemail: save failed")
                    err = t("voicemail.save_failed", user_lang)
                    await self._hooks.on_error(str(exc))
                    await self._hooks.on_response(
                        text=err, llm_ms=0, history_turns=len(history) // 2
                    )
                    record["error"] = f"voicemail_save_failed:{exc.__class__.__name__}"
                    record["response_text"] = err
                    outcome.response_text = err
                    return outcome
                log.info(
                    "session %d: voicemail saved id=%d from=%s to=%s (%dms)",
                    session_id, vm_id, speaker_name or "guest",
                    to_name, duration_ms,
                )
                # Best-effort live ping to every open tab the recipient
                # has authenticated.  The frontend WS handler plays a
                # chime, refreshes the inbox badge, and (if permission
                # granted + tab hidden) raises a desktop Notification.
                # Imported lazily so the pipeline module doesn't pull in
                # ws.py at import time (would create a circular import).
                try:
                    from .ws import notify_voicemail_arrived
                    await notify_voicemail_arrived(
                        to_profile_id=to_profile_id,
                        message_id=vm_id,
                        from_name=speaker_name,
                        duration_ms=duration_ms,
                    )
                except Exception:
                    log.warning("voicemail: live notify failed", exc_info=True)
                ack = t("voicemail.saved", user_lang, to=to_name)
                await self._hooks.on_tool_called(
                    name="leave_message",
                    args={
                        "to": to_name,
                        "to_profile_id": to_profile_id,
                        "id": vm_id,
                        "duration_ms": duration_ms,
                    },
                )
                await self._hooks.on_response(
                    text=ack, llm_ms=0, history_turns=len(history) // 2
                )
                record["tool_name"] = "leave_message"
                record["tool_args"] = json.dumps(
                    {"to": to_name, "id": vm_id}, ensure_ascii=False
                )
                record["response_text"] = ack
                outcome.response_text = ack
                outcome.last_response_text = ack
                return outcome

            # 0. Passphrase preprocessing — must run BEFORE intent matching
            # so a "passphrase amber new topic" doesn't get eaten by the
            # `new_topic` intent before its passphrase is checked.  If a
            # phrase is present and valid we open an auth window and
            # strip the prefix from the transcript so the rest of the
            # pipeline sees the actual request.
            passphrase, transcript_after = _extract_passphrase(transcript)
            is_authenticated = False
            if passphrase is not None and profile_id is not None:
                ok = await verify_passphrase(profile_id, passphrase)
                if ok:
                    is_authenticated = True
                    await self._hooks.on_auth_succeeded(
                        profile_id=profile_id, window_s=AUTH_WINDOW_S
                    )
                    # Drop the passphrase prefix; the remainder is the real
                    # request.  When the user said ONLY the passphrase,
                    # transcript_after is "" and the empty-transcript ack
                    # branch below handles it.
                    transcript = transcript_after
                    record["transcript"] = transcript
                    log.info(
                        "session %d: passphrase OK for profile=%d", session_id, profile_id
                    )
                else:
                    log.info(
                        "session %d: passphrase mismatch for profile=%d",
                        session_id, profile_id,
                    )
            elif passphrase is not None and profile_id is None:
                log.info(
                    "session %d: passphrase ignored (no recognised profile)",
                    session_id,
                )

            # If we didn't just authenticate, inherit any in-flight auth
            # window the session is carrying from a previous turn.
            # NOT inherited on mixed turns (#59): a second voice inside
            # the utterance must not ride the first speaker's window
            # into high_write tools — those defer to pending_actions and
            # the clarify/re-ask path instead.
            if not is_authenticated and not attr.is_mixed:
                until = await self._hooks.current_auth_until()
                if until is not None and until > time.time():
                    is_authenticated = True

            # Empty transcript AFTER passphrase strip is still fine — the
            # user just said the passphrase and nothing else.  Acknowledge
            # without diving into the agent loop.
            if not transcript:
                ack = (
                    t("auth.passphrase_ok", user_lang)
                    if (passphrase and is_authenticated)
                    else t("pipeline.empty_transcript", user_lang)
                )
                await self._hooks.on_response(
                    text=ack, llm_ms=0, history_turns=len(history) // 2
                )
                record["response_text"] = ack
                record["error"] = None if is_authenticated else "empty_transcript"
                outcome.response_text = ack
                outcome.last_response_text = ack
                return outcome

            # 0b. Image-attached turn — short-circuit to vision.
            # When the user dropped an image into the UI we bypass the
            # tool loop entirely and route transcript + image as a
            # single Q&A through the local multimodal LLM (Gemma 4 via
            # LM Studio).  No tools, no history scoping — vision turns
            # are one-shot by design, and pulling history risks a
            # private context leaking into a screenshot description.
            if attached_image_b64:
                log.info(
                    "session %d: image-attached turn (%d b64 chars)",
                    session_id, len(attached_image_b64),
                )
                # The user's text drives the visual question.  If the
                # transcript is just "what is this", that's fine — the
                # model describes the image; if it's specific ("what's
                # this error"), it focuses on that.  Stripping the
                # passphrase prefix here keeps the question clean for
                # the vision model.
                await self._hooks.on_progress(step="vision")
                t_vision_start = time.time()
                try:
                    answer = await vision.analyze_image_b64(
                        attached_image_b64,
                        transcript,
                        client_id=client_id,
                        tool_name="image_attached",
                    )
                except Exception as exc:
                    log.exception("session %d: vision call failed", session_id)
                    answer = t("vision.failed", user_lang)
                    record["error"] = f"vision_failed:{exc.__class__.__name__}"
                vision_ms = int((time.time() - t_vision_start) * 1000)
                if not answer:
                    answer = t("vision.empty", user_lang)
                record["tool_name"] = "image_attached"
                record["llm_ms"] = vision_ms
                record["response_text"] = answer
                outcome.response_text = answer
                outcome.last_response_text = answer
                outcome.history_appended = True
                await self._hooks.on_tool_called(
                    name="image_attached", args={"transcript": transcript}
                )
                await self._hooks.on_response(
                    text=answer,
                    llm_ms=vision_ms,
                    history_turns=(len(history) // 2) + 1,
                )
                return outcome

            # 1. Local intents — zero-latency, no LLM call.  Patterns
            # come from the user's locale; ``user_lang`` resolved above
            # is the source of truth.
            intent = match_intent(transcript, user_lang)
            if intent in ("replay_full", "replay_resume"):
                mode = "full" if intent == "replay_full" else "resume"
                record["tool_name"] = f"intent:{intent}"
                # No response text — the browser plays its cached TTS buffer.
                await self._hooks.on_replay(mode)
                return outcome
            if intent == "new_topic":
                ack = t("pipeline.new_topic_confirm", user_lang)
                record["tool_name"] = "intent:new_topic"
                record["response_text"] = ack
                outcome.history_cleared = True
                outcome.response_text = ack
                outcome.last_response_text = ack
                await self._hooks.on_tool_called(name="intent:new_topic", args={})
                await self._hooks.on_response(
                    text=ack, llm_ms=0, history_turns=0
                )
                await self._hooks.on_history_reset()
                return outcome

            # 2. Build memory_context + (if applicable) pending-queue
            # block concurrently.  Both are independent reads with no
            # data dependency, so running them in parallel cuts the
            # serial latency on authenticated turns by another ~5-50 ms
            # (the pending fetch is cheap but still a real SQL hit).
            should_replay_pending = (
                passphrase is not None
                and is_authenticated
                and profile_id is not None
            )
            mem_ctx_coro = self._build_memory_context(
                transcript, client_id, speaker_name, session_id,
                embed_task=embed_task,
                participants=attr.participants if attr.is_mixed else (),
            )
            if should_replay_pending:
                pending_coro = list_pending_actions(
                    profile_id=profile_id, client_id=client_id
                )
                memory_context, queue = await asyncio.gather(
                    mem_ctx_coro, pending_coro, return_exceptions=False
                )
            else:
                memory_context = await mem_ctx_coro
                queue = []

            # 2a. Pending-queue replay.  If we JUST authenticated this
            # turn AND the speaker has queued actions, surface them as
            # an extra context block.  The LLM sees a short summary
            # of the top pending entries and can naturally fold a
            # "you have X, Y in the queue — approve?" into its reply.
            # This is read-only context — the user still has to say
            # "approve number N" to execute, which reuses the existing
            # `approve_pending` tool path.
            if should_replay_pending and queue:
                head = queue[:3]
                items_block = "\n".join(
                    f"- id={q['id']}: {q['summary']}" for q in head
                )
                more = (
                    f"\n(more in queue: {len(queue) - len(head)})"
                    if len(queue) > len(head)
                    else ""
                )
                memory_context += (
                    "\n[Pending action queue — speaker just authenticated, "
                    "you MAY mention these and offer to approve/reject "
                    "(do NOT execute without explicit user consent):]\n"
                    + items_block
                    + more
                )
                log.info(
                    "session %d: replaying %d pending action(s) in context",
                    session_id, len(queue),
                )

            # Append the unseen-replies block AFTER the pending-queue block
            # so order is: speaker tag → past exchanges → pending → unseen
            # voicemail replies.  When neither speaker nor profile resolved,
            # the block is empty and this is a no-op.
            if unseen_replies_block:
                memory_context += unseen_replies_block

            # 3. Multi-step agent loop with tool use.  Plumb a stream_sink
            # callback so tools (currently web_search) can emit text
            # sentence-by-sentence into the TTS pipeline as the LLM
            # produces them — first audio in ~1.5 s instead of ~5 s after
            # the model finishes thinking.
            async def _stream_sink(text: str) -> None:
                await self._hooks.on_response_chunk(text=text)

            async def _progress_sink(step: str, detail: str | None = None) -> None:
                await self._hooks.on_progress(step=step, detail=detail)

            async def _media_sink(url: str, source: str) -> None:
                await self._hooks.on_media_started(url=url, source=source)

            await self._hooks.on_progress(step="agent")
            # ``user_lang`` was resolved up-front from the speaker's
            # profile.settings (or defaulted to English for unknown
            # speakers).  Same value used for every t() call in this
            # turn and now passed into the agent context.
            # Step-up auth: check whether the session has an active grant
            # (push-to-device approval).  Checked here (not in the agent
            # loop) so the hook stays close to the auth-window check above.
            step_up_auth = False
            step_up_until = await self._hooks.current_step_up_auth_until()
            if step_up_until is not None and step_up_until > time.time():
                step_up_auth = True

            ctx = AgentContext(
                client_id=client_id,
                profile_id=profile_id,
                is_authenticated=is_authenticated,
                user_lang=user_lang or "en",
                stream_sink=_stream_sink,
                progress_sink=_progress_sink,
                media_sink=_media_sink,
                device_kind=device_kind,
                step_up_auth=step_up_auth,
                mixed=attr.is_mixed,
                participants=attr.participants,
            )
            agent_result: AgentResult = await run_agent(
                transcript,
                ctx=ctx,
                history=history,
                memory_context=memory_context,
            )
            last = agent_result.last_tool
            record["llm_ms"] = agent_result.elapsed_ms
            record["tool_name"] = last.name if last else None
            record["tool_args"] = (
                json.dumps(last.args, ensure_ascii=False) if last and last.args else None
            )
            record["response_text"] = agent_result.response_text
            outcome.response_text = agent_result.response_text
            outcome.last_response_text = agent_result.response_text
            outcome.history_appended = True
            outcome.tts_voice_override = agent_result.tts_voice_override

            await self._hooks.on_tool_called(
                name=last.name if last else None,
                args=last.args if last else None,
                data=last.data if last else None,
            )
            await self._hooks.on_response(
                text=agent_result.response_text,
                llm_ms=agent_result.elapsed_ms,
                # +1 because the WS layer will append this turn after run().
                history_turns=(len(history) // 2) + 1,
                tts_voice_override=agent_result.tts_voice_override,
            )
            return outcome

        except asyncio.CancelledError:
            log.info("session %d pipeline cancelled", session_id)
            record["error"] = "cancelled"
            raise
        except Exception as e:
            log.exception("session %d pipeline failed", session_id)
            record["error"] = f"{e.__class__.__name__}: {e}"
            try:
                await self._hooks.on_error(str(e))
            except Exception:
                pass
            return outcome
        finally:
            try:
                await self._hooks.on_response_end()
            except Exception:
                pass
            try:
                record_id = await save_utterance(**record)
                # Background: compute embedding so it doesn't delay the
                # continuation window opening.
                if (
                    record_id
                    and record.get("transcript")
                    and record.get("response_text")
                    and memory.EMBEDDING_ENABLED
                ):
                    asyncio.create_task(
                        memory.compute_and_save_embedding(
                            record_id,
                            record["transcript"],
                            record["response_text"],
                        )
                    )
            except Exception:
                log.exception("failed to persist utterance")

    async def _resolve_speaker(
        self,
        spk_task: "asyncio.Task | None",
        client_id: str | None,
        session_id: int,
    ) -> speaker.SpeakerAttribution:
        """
        Await the parallel speaker-embedding task, match against enrolled
        profiles, and return a :class:`speaker.SpeakerAttribution`.

        Single-voice utterances resolve exactly as before (name +
        tts_voice + profile_id from the best cosine match).  When the
        windowed partials split into two voice clusters (#59), each
        cluster is identified separately:

          - clusters land on TWO distinct profiles → ``mixed_known``
          - one profile + one stranger             → ``mixed_unknown``
          - both clusters on the SAME profile      → single (prosody
            swing, not two people — guard against false splits)
          - nobody recognised                      → ``none``

        Mixed modes carry no profile_id: attribution, memory writes and
        device routing all stay conservative, and the agent asks instead
        of guessing.
        """
        if not spk_task or not client_id:
            if spk_task and not spk_task.done():
                spk_task.cancel()
            return speaker.SpeakerAttribution()
        try:
            profiles_raw = await get_speaker_profiles(client_id)
            spk_emb, partials = await spk_task
            if spk_emb is None or not profiles_raw:
                return speaker.SpeakerAttribution()
            # Stored as float32 (resemblyzer's native dtype: 256-dim × 4 B
            # = 1024 B). Reading as float64 silently halves the dimension.
            # Row shape: (id, name, embedding_bytes, sample_count, tts_voice).
            decoded = [
                (nm, np.frombuffer(bl, dtype=np.float32))
                for _, nm, bl, _, _ in profiles_raw
            ]

            clusters = (
                speaker.split_speakers(partials) if partials is not None else []
            )
            if len(clusters) >= 2:
                matches = [speaker.identify(c, decoded)[0] for c in clusters]
                known = [n for n in matches if n]
                distinct = list(dict.fromkeys(known))
                if len(distinct) >= 2:
                    log.info(
                        "session %d mixed utterance: known speakers %s",
                        session_id, distinct,
                    )
                    return speaker.SpeakerAttribution(
                        mode="mixed_known", participants=tuple(distinct)
                    )
                if len(distinct) == 1 and len(known) < len(matches):
                    log.info(
                        "session %d mixed utterance: %r + unknown voice",
                        session_id, distinct[0],
                    )
                    return speaker.SpeakerAttribution(
                        mode="mixed_unknown", participants=tuple(distinct)
                    )
                # Both clusters resolved to the same profile (or none):
                # not two people — fall through to whole-utterance match.

            name, score = speaker.identify(spk_emb, decoded)
            if not name:
                return speaker.SpeakerAttribution()
            # Match found — pull the tts_voice + profile_id from the same
            # row.  Linear scan is fine: enrolled profiles per client
            # typically fit on one hand.
            voice: str | None = None
            profile_id: int | None = None
            for pid, nm, _, _, tv in profiles_raw:
                if nm == name:
                    voice = tv
                    profile_id = pid
                    break
            log.info(
                "session %d speaker=%r score=%.3f voice=%r profile=%s",
                session_id, name, score, voice, profile_id,
            )
            return speaker.SpeakerAttribution(
                mode="single", name=name, tts_voice=voice, profile_id=profile_id
            )
        except Exception as exc:
            log.warning("session %d speaker ID error: %s", session_id, exc)
            return speaker.SpeakerAttribution()

    async def _build_memory_context(
        self,
        transcript: str,
        client_id: str | None,
        speaker_name: str | None,
        session_id: int,
        *,
        embed_task: "asyncio.Task | None" = None,
        participants: "tuple[str, ...]" = (),
    ) -> str:
        """
        Stitch together the system-prompt context block: speaker tag plus
        any semantically-similar past turns.

        If ``embed_task`` is supplied, we await it instead of starting
        a fresh embedding call — this is the parallelism-win: the
        task was kicked off right after ASR, so by the time we get
        here it's usually done.  Fall back to a fresh embed when no
        task was passed (e.g. callers that pre-date this change, or
        the embed_task ended up None because memory was disabled).
        """
        if speaker_name:
            ctx = f"[Current speaker: {speaker_name}]\n"
        elif participants:
            # Mixed turn (#59): tell the model several people spoke so it
            # asks instead of guessing whenever the request needs a
            # personal device or personal memory.
            ctx = (
                "[Mixed turn: more than one person spoke. Recognised: "
                + ", ".join(participants)
                + ". Do not attribute the request to one person; if it"
                " needs a personal device or personal data, ask whose.]\n"
            )
        else:
            ctx = ""
        if not (memory.EMBEDDING_ENABLED and client_id):
            # If somebody scheduled a task that's now dead weight, cancel.
            if embed_task is not None and not embed_task.done():
                embed_task.cancel()
            return ctx
        try:
            if embed_task is not None:
                query_vec = await embed_task
            else:
                query_vec = await memory.embed_query(transcript)
            since_ts = time.time() - memory.MEMORY_MAX_AGE_DAYS * 86400
            candidates = await get_candidate_utterances(
                client_id, since_ts, limit=200, speaker_name=speaker_name
            )
            similar = memory.retrieve(
                query_vec,
                candidates,
                top_k=memory.MEMORY_TOP_K,
                threshold=memory.MEMORY_SIMILARITY_THRESHOLD,
            )
            if similar:
                parts = [
                    f"«{u['transcript']}» → «{u['response_text']}»" for u in similar
                ]
                ctx += "[Relevant past exchanges (for context):]\n" + "\n".join(parts)
                log.info(
                    "session %d: injecting %d memory items", session_id, len(similar)
                )
        except Exception as exc:
            log.warning("session %d: memory retrieval failed: %s", session_id, exc)
        return ctx
