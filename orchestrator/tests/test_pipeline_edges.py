"""
Pipeline edge cases — paths that bypass the agent loop.

Three branches in ``Pipeline.run`` decide a turn's flow before the
LLM ever sees the transcript:

  1. Audio shorter than ``MIN_DURATION_MS`` → ``pipeline.not_heard``, no ASR.
  2. Image attached → vision short-circuit, no agent.
  3. Empty transcript after passphrase strip + valid passphrase →
     ``auth.passphrase_ok``, no agent.

Each of these has been wrong in earlier iterations (the
image-attached branch leaked through to the agent before we noticed,
and the passphrase-only branch used to feed the bare "passphrase X"
prefix to the LLM verbatim — Russian "пароль янтарь" is the canonical
fixture).  These tests pin each path.

We resolve expected strings via ``i18n.t`` so the assertions track
whatever the locale file says today.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline import MIN_DURATION_MS, Pipeline, SAMPLE_BYTES_PER_SECOND


# ── Tiny PipelineHooks stub — records callback args for assertion ─────


class StubHooks:
    """Records every hook call; satisfies the PipelineHooks protocol."""

    def __init__(self):
        self.events: list[tuple] = []
        self._auth_until: float | None = None

    async def on_transcript(self, *, text, asr_ms, speaker):
        self.events.append(("transcript", text, asr_ms, speaker))

    async def on_speaker_identified(self, *, name, tts_voice):
        self.events.append(("speaker_identified", name, tts_voice))

    async def on_tool_called(self, *, name, args):
        self.events.append(("tool_called", name, args))

    async def on_response(self, *, text, llm_ms, history_turns):
        self.events.append(("response", text, llm_ms))

    async def on_response_end(self):
        self.events.append(("response_end",))

    async def on_error(self, message):
        self.events.append(("error", message))

    async def on_replay(self, mode):
        self.events.append(("replay", mode))

    async def on_history_reset(self):
        self.events.append(("history_reset",))

    async def on_response_chunk(self, *, text):
        self.events.append(("chunk", text))

    async def on_progress(self, *, step, detail=None):
        self.events.append(("progress", step))

    async def current_auth_until(self):
        return self._auth_until

    async def on_auth_succeeded(self, *, profile_id, window_s):
        import time
        self._auth_until = time.time() + window_s

    async def cookie_profile_id(self):
        return None

    def find(self, kind):
        return [e for e in self.events if e[0] == kind]


@pytest.fixture
def hooks():
    return StubHooks()


async def test_short_audio_returns_not_heard(hooks):
    """Audio under MIN_DURATION_MS skips ASR and replies pipeline.not_heard."""
    from app.i18n import t

    # 100 ms — below the 200 ms minimum.
    short_audio = bytes(SAMPLE_BYTES_PER_SECOND // 10)
    p = Pipeline(hooks=hooks)
    outcome = await p.run(
        audio=short_audio, session_id=1, client_id="cli-1",
        history=[], last_response_text="",
    )
    assert outcome.transcript is None
    responses = hooks.find("response")
    # No profile resolved → user_lang stays None → English fallback.
    assert responses and responses[0][1] == t("pipeline.not_heard", None)
    # ASR should never have run for too-short audio.
    transcripts = hooks.find("transcript")
    assert transcripts == [], "transcript event must not fire on short audio"


async def test_image_attached_bypasses_agent(hooks):
    """attached_image_b64 routes through vision, skipping the agent loop."""
    audio = bytes(SAMPLE_BYTES_PER_SECOND)  # 1 s of silence
    # ASR returns something innocuous; vision returns a known string.
    with patch("app.pipeline.transcribe", new=AsyncMock(return_value=("describe this", 50))), \
         patch("app.pipeline.vision.analyze_image_b64",
               new=AsyncMock(return_value="a small dog")) as vis, \
         patch("app.pipeline.run_agent", new=AsyncMock()) as agent:
        p = Pipeline(hooks=hooks)
        outcome = await p.run(
            audio=audio, session_id=2, client_id="cli-2",
            history=[], last_response_text="",
            attached_image_b64="data:image/png;base64,AAAA",
        )
    vis.assert_awaited_once()
    agent.assert_not_called(), "agent loop must not run on image-attached turns"
    assert outcome.response_text == "a small dog"
    tool_calls = hooks.find("tool_called")
    assert tool_calls and tool_calls[0][1] == "image_attached"


async def test_passphrase_only_short_circuits(hooks, make_agent_ctx):
    """A turn that's just a valid passphrase acks without invoking the agent."""
    from app.i18n import t
    from app.storage import save_speaker_profile
    from app.user_files import set_passphrase

    # Seed a speaker profile and bind a passphrase to it.
    pid = await save_speaker_profile(
        client_id="cli-3", name="Eugene",
        embedding=b"\x00" * (4 * 256),  # 1024-byte float32 vector
    )
    await set_passphrase(pid, "янтарь")

    audio = bytes(SAMPLE_BYTES_PER_SECOND)
    with patch("app.pipeline.transcribe",
               new=AsyncMock(return_value=("пароль янтарь", 50))), \
         patch.object(Pipeline, "_resolve_speaker",
                      new=AsyncMock(return_value=("Eugene", None, pid))), \
         patch("app.pipeline.run_agent", new=AsyncMock()) as agent:
        p = Pipeline(hooks=hooks)
        outcome = await p.run(
            audio=audio, session_id=3, client_id="cli-3",
            history=[], last_response_text="",
        )
    agent.assert_not_called(), "agent must not run for a passphrase-only turn"
    # Default settings.language is "auto" → user_lang falls back to en.
    assert outcome.response_text == t("auth.passphrase_ok", "en")
    assert hooks._auth_until is not None, "on_auth_succeeded must have fired"
