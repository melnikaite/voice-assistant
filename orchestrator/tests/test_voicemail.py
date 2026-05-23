"""
Voicemail feature — storage round-trip + pipeline branch + inbox tools.

What we pin here:

  • The DB row + WAV file are written atomically (a row with no wav
    on disk would be a corruption bug).
  • Recipient-name resolution tolerates Russian inflections (the most
    common natural-speech failure mode: "передай Жене" when the
    profile is enrolled as "Женя" — Russian fixtures are the most
    interesting case because of dative case morphology).
  • The pipeline's voicemail branch never invokes the agent loop.
  • The inbox tools refuse access without a recognised profile.
  • inbox_summary returns a tts_voice_override pointing at the
    sender's voice — the WS layer uses that to speak the summary
    in the original sender's voice.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline import (
    SAMPLE_BYTES_PER_SECOND,
    Pipeline,
    _extract_voicemail,
    _resolve_voicemail_recipient,
)
from app.storage import (
    VOICE_MESSAGES_DIR,
    count_unread_voicemail,
    get_voice_message,
    list_outgoing_voicemail,
    list_unseen_replies_for_sender,
    list_voicemail,
    mark_voicemail_listened,
    mark_voicemail_reply_delivered,
    save_voice_message,
    save_voicemail_reply,
    save_speaker_profile,
)


# ── Pattern matcher ────────────────────────────────────────────────────


@pytest.mark.parametrize("phrase,expected_to,expected_body", [
    ("передай Жене что я опоздаю на 5 минут", "Жене", "что я опоздаю на 5 минут"),
    ("Передай для Жени: я опоздаю", "Жени", "я опоздаю"),
    ("оставь сообщение для Алисы что встреча в 5", "Алисы", "что встреча в 5"),
    ("Оставьте сообщение Жене: всё ок", "Жене", "всё ок"),
    ("leave a message for John: I'll be late", "John", "I'll be late"),
    ("Tell John that the meeting is at 5", "John", "the meeting is at 5"),
    ("Sag Anna, dass ich später komme", "Anna", "ich später komme"),
])
def test_extract_voicemail_matches(phrase, expected_to, expected_body):
    """All shapes the production regex should parse."""
    result = _extract_voicemail(phrase)
    assert result is not None, f"failed to match: {phrase!r}"
    to, body = result
    assert to == expected_to
    assert body == expected_body


@pytest.mark.parametrize("phrase", [
    "привет как дела",
    "поставь будильник на 8 утра",
    "погода в москве",
    "передай мне соль",  # Pattern matches, but no body → still parsed.
])
def test_extract_voicemail_non_matches(phrase):
    """Unambiguous non-voicemail phrases shouldn't trigger the branch.

    Note: "передай мне соль" ("pass me the salt") DOES match the
    pattern (to=мне body=соль) — that's by design.  The recipient
    resolver in the pipeline then fails to find a profile named "мне"
    ("to me") and we fall through to the agent loop with
    'unknown_recipient' surfaced.  So the fall-through is at the
    resolution layer, not the pattern.
    """
    result = _extract_voicemail(phrase)
    if phrase == "передай мне соль":
        assert result == ("мне", "соль")
    else:
        assert result is None


# ── Recipient resolver ─────────────────────────────────────────────────


def _profile(pid, name):
    """Build a profile-row tuple matching get_speaker_profiles' shape."""
    return (pid, name, b"\x00" * (4 * 256), 1, None)


def test_resolve_exact_case_insensitive():
    profiles = [_profile(1, "Женя"), _profile(2, "Алиса")]
    assert _resolve_voicemail_recipient("женя", profiles) == (1, "Женя")
    assert _resolve_voicemail_recipient("АЛИСА", profiles) == (2, "Алиса")


def test_resolve_russian_inflections():
    """The hard case: dative 'Жене' should match nominative 'Женя'."""
    profiles = [_profile(1, "Женя"), _profile(2, "Алиса")]
    assert _resolve_voicemail_recipient("Жене", profiles) == (1, "Женя")
    assert _resolve_voicemail_recipient("Жени", profiles) == (1, "Женя")
    assert _resolve_voicemail_recipient("Алисе", profiles) == (2, "Алиса")


def test_resolve_tie_breaks_shortest():
    """When two profiles share the 3-char stem, the shorter name wins."""
    profiles = [_profile(1, "Жененька"), _profile(2, "Женя")]
    assert _resolve_voicemail_recipient("Жене", profiles) == (2, "Женя")


def test_resolve_unknown_returns_none():
    profiles = [_profile(1, "Женя")]
    assert _resolve_voicemail_recipient("Боб", profiles) is None
    assert _resolve_voicemail_recipient("", profiles) is None


# ── Storage round-trip ─────────────────────────────────────────────────


async def test_save_then_list_round_trip():
    pid_to = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x00" * (4 * 256),
    )
    pid_from = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    audio = bytes(SAMPLE_BYTES_PER_SECOND * 2)  # 2 s of silence
    mid = await save_voice_message(
        from_profile_id=pid_from, from_name="Алиса",
        to_profile_id=pid_to, to_name="Женя",
        transcript="перезвони пожалуйста", duration_ms=2000,
        audio_pcm=audio,
    )
    # The WAV file must actually exist on disk.
    wav = VOICE_MESSAGES_DIR / f"{mid}.wav"
    assert wav.exists(), "wav file must be written next to the row"
    assert wav.stat().st_size > 44, "must contain audio frames, not just header"
    # The row round-trips.
    row = await get_voice_message(mid)
    assert row is not None
    assert row["from_name"] == "Алиса"
    assert row["to_profile_id"] == pid_to
    assert row["transcript"] == "перезвони пожалуйста"
    assert row["listened_at"] is None
    # And the recipient sees it.
    inbox = await list_voicemail(pid_to)
    assert len(inbox) == 1 and inbox[0]["id"] == mid
    # And nobody else does.
    assert await list_voicemail(pid_from) == []


async def test_mark_listened_is_idempotent():
    pid = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x00" * (4 * 256),
    )
    mid = await save_voice_message(
        from_profile_id=None, from_name=None,
        to_profile_id=pid, to_name="Женя",
        transcript="hi", duration_ms=500,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND // 2),
    )
    assert await mark_voicemail_listened(mid, pid) is True
    assert await mark_voicemail_listened(mid, pid) is False, "second time = noop"
    # Wrong recipient — must not flip listened.
    assert await mark_voicemail_listened(mid, pid + 999) is False
    # Unread count reflects the listened flag.
    assert await count_unread_voicemail(pid) == 0


async def test_reply_also_marks_listened():
    pid = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x00" * (4 * 256),
    )
    mid = await save_voice_message(
        from_profile_id=None, from_name="гость",
        to_profile_id=pid, to_name="Женя",
        transcript="перезвони", duration_ms=1000,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND),
    )
    assert await save_voicemail_reply(mid, pid, "хорошо, через час") is True
    row = await get_voice_message(mid)
    assert row["reply_text"] == "хорошо, через час"
    assert row["listened_at"] is not None, "reply implies listened"
    assert row["replied_at"] is not None


# ── Pipeline branch ────────────────────────────────────────────────────


class _VoicemailHooks:
    """Minimal hooks impl that records every callback."""

    def __init__(self):
        self.events: list[tuple] = []
        self._auth_until = None

    async def on_transcript(self, *, text, asr_ms, speaker):
        self.events.append(("transcript", text, speaker))

    async def on_speaker_identified(self, *, name, tts_voice):
        self.events.append(("speaker_identified", name))

    async def on_tool_called(self, *, name, args, data=None):
        self.events.append(("tool_called", name, args))

    async def on_response(
        self, *, text, llm_ms, history_turns, tts_voice_override=None,
    ):
        self.events.append(("response", text, tts_voice_override))

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
        self._auth_until = 1.0  # truthy

    async def cookie_profile_id(self):
        return None

    def find(self, kind):
        return [e for e in self.events if e[0] == kind]


async def test_voicemail_pipeline_branch_saves_and_skips_agent():
    """A 'tell X that …' / 'передай X …' utterance saves the message and never calls the agent."""
    # Two enrolled profiles under the same client.
    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x02" * (4 * 256),
    )
    audio = bytes(SAMPLE_BYTES_PER_SECOND * 2)
    with patch(
        "app.pipeline.transcribe",
        new=AsyncMock(return_value=("передай Жене что я опоздаю", 100)),
    ), patch.object(
        Pipeline, "_resolve_speaker",
        new=AsyncMock(return_value=("Алиса", None, pid_alice)),
    ), patch(
        "app.pipeline.run_agent", new=AsyncMock(),
    ) as agent:
        hooks = _VoicemailHooks()
        p = Pipeline(hooks=hooks)
        outcome = await p.run(
            audio=audio, session_id=1, client_id="cli-vm",
            history=[], last_response_text="",
        )
    agent.assert_not_called(), "agent must NOT run on a voicemail-leave turn"
    # The tool_called event fires with name=leave_message.
    tools = hooks.find("tool_called")
    assert tools and tools[0][1] == "leave_message"
    assert tools[0][2]["to"] == "Женя"
    # The row exists, addressed to Женя, from Алиса, transcript = body.
    rows = await list_voicemail(pid_zhenya)
    assert len(rows) == 1
    row = rows[0]
    assert row["to_profile_id"] == pid_zhenya
    assert row["from_profile_id"] == pid_alice
    assert row["from_name"] == "Алиса"
    # Transcript stored is the body only — the "передай Жене" prefix
    # is stripped by the regex group capture.
    assert row["transcript"] == "что я опоздаю"
    # The outcome carries the i18n ack.
    assert outcome.response_text is not None
    assert "Жен" in outcome.response_text  # ru ack "Записал сообщение для Женя"


async def test_voicemail_unknown_recipient_short_circuits():
    """'передай Бобу' ('tell Bob …') without a Bob profile → unknown_recipient response, no row."""
    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    audio = bytes(SAMPLE_BYTES_PER_SECOND)
    with patch(
        "app.pipeline.transcribe",
        new=AsyncMock(return_value=("передай Бобу что я опоздаю", 100)),
    ), patch.object(
        Pipeline, "_resolve_speaker",
        new=AsyncMock(return_value=("Алиса", None, pid_alice)),
    ), patch(
        "app.pipeline.run_agent", new=AsyncMock(),
    ) as agent:
        hooks = _VoicemailHooks()
        p = Pipeline(hooks=hooks)
        await p.run(
            audio=audio, session_id=2, client_id="cli-vm",
            history=[], last_response_text="",
        )
    agent.assert_not_called(), "agent must not run when recipient unknown"
    tools = hooks.find("tool_called")
    assert tools and tools[0][2].get("status") == "unknown_recipient"
    # And no row was inserted for any profile.
    assert await list_voicemail(pid_alice) == []


# ── Inbox tools ────────────────────────────────────────────────────────


async def test_inbox_list_filters_by_recipient(make_agent_ctx):
    """A speaker only ever sees their own inbox."""
    from app.tools.inbox import inbox_list

    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x02" * (4 * 256),
    )
    # Two messages: one for Zhenya, one for Alice.
    await save_voice_message(
        from_profile_id=None, from_name="гость",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="msg for Zhenya", duration_ms=500,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND // 2),
    )
    await save_voice_message(
        from_profile_id=None, from_name="гость",
        to_profile_id=pid_alice, to_name="Алиса",
        transcript="msg for Alice", duration_ms=500,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND // 2),
    )
    # Alice asks: she sees only her own.
    ctx = make_agent_ctx(profile_id=pid_alice, client_id="cli-vm")
    res = await inbox_list(ctx=ctx)
    assert res.data["count"] == 1
    assert res.data["messages"][0]["transcript"] == "msg for Alice"


async def test_inbox_list_without_profile_refuses(make_agent_ctx):
    """No recognised profile → auth_required, never leaks anyone's inbox."""
    from app.tools.inbox import inbox_list

    ctx = make_agent_ctx(profile_id=None, client_id="cli-vm")
    res = await inbox_list(ctx=ctx)
    assert (res.data or {}).get("error") == "auth_required"


async def test_inbox_read_marks_listened_and_emits_play_signal(make_agent_ctx):
    """inbox_read marks the row listened and includes voicemail_play_id in data."""
    from app.tools.inbox import inbox_read

    pid = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x00" * (4 * 256),
    )
    mid = await save_voice_message(
        from_profile_id=None, from_name="гость",
        to_profile_id=pid, to_name="Женя",
        transcript="тест", duration_ms=500,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND // 2),
    )
    ctx = make_agent_ctx(profile_id=pid, client_id="cli-vm")
    res = await inbox_read(ctx=ctx, message_id=mid)
    assert res.data["voicemail_play_id"] == mid
    row = await get_voice_message(mid)
    assert row["listened_at"] is not None


async def test_inbox_read_wrong_recipient_404s(make_agent_ctx):
    """A different profile gets «not found», not «forbidden» — no existence leak."""
    from app.tools.inbox import inbox_read

    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x00" * (4 * 256),
    )
    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    mid = await save_voice_message(
        from_profile_id=None, from_name="гость",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="секрет", duration_ms=500,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND // 2),
    )
    # Alice (wrong recipient) asks to read Zhenya's message.
    ctx = make_agent_ctx(profile_id=pid_alice, client_id="cli-vm")
    res = await inbox_read(ctx=ctx, message_id=mid)
    assert (res.data or {}).get("error") == "not_found"


async def test_inbox_summary_returns_sender_tts_voice(make_agent_ctx):
    """inbox_summary surfaces the sender's tts_voice as the override.

    The point of the override: the host hears the summary IN the
    original sender's voice (assuming they have a cloned voice
    registered on their speaker_profile).
    """
    from app.tools.inbox import inbox_summary
    from app.storage import set_speaker_tts_voice

    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x00" * (4 * 256),
    )
    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    await set_speaker_tts_voice(pid_alice, "alice-cloned-voice")
    mid = await save_voice_message(
        from_profile_id=pid_alice, from_name="Алиса",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="перезвони пожалуйста в течение часа",
        duration_ms=2000,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND * 2),
    )
    ctx = make_agent_ctx(profile_id=pid_zhenya, client_id="cli-vm")
    # Stub the LLM call — we're not testing summary quality here.
    with patch(
        "app.tools.inbox._llm_summarise",
        new=AsyncMock(return_value="Алиса просит перезвонить в течение часа."),
    ):
        res = await inbox_summary(ctx=ctx, message_id=mid)
    assert res.tts_voice_override == "alice-cloned-voice"
    # And the summary was cached on the row.
    row = await get_voice_message(mid)
    assert row["summary"] == "Алиса просит перезвонить в течение часа."


# ── Reply-replay storage (Item 1) ──────────────────────────────────────


async def test_unseen_replies_round_trip():
    """Voicemail → recipient reply → sender sees it once, then never again.

    Pins the wiring for the "by the way, Zhenya replied …" replay
    branch in pipeline.py: save a voicemail, set a reply, the sender's
    unseen-replies list returns the row exactly once; after
    ``mark_reply_delivered`` the list is empty even though the reply
    itself is still on disk.
    """
    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x02" * (4 * 256),
    )
    mid = await save_voice_message(
        from_profile_id=pid_alice, from_name="Алиса",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="перезвони",
        duration_ms=1000,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND),
    )
    # No reply yet → nothing to surface.
    assert await list_unseen_replies_for_sender(pid_alice) == []
    # Recipient replies.
    assert await save_voicemail_reply(mid, pid_zhenya, "ок, через час") is True
    unseen = await list_unseen_replies_for_sender(pid_alice)
    assert len(unseen) == 1
    assert unseen[0]["id"] == mid
    assert unseen[0]["reply_text"] == "ок, через час"
    assert unseen[0]["to_name"] == "Женя"
    # Stamp delivered.
    assert await mark_voicemail_reply_delivered(mid) is True
    # Now the unseen list is empty (idempotent: a second mark is a noop).
    assert await list_unseen_replies_for_sender(pid_alice) == []
    assert await mark_voicemail_reply_delivered(mid) is False, "second mark = noop"


# ── Outgoing list (Item 4) ─────────────────────────────────────────────


async def test_outgoing_voicemail_lists_own_sent():
    """Sender sees their own outgoing rows, newest first; no leakage."""
    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x02" * (4 * 256),
    )
    audio = bytes(SAMPLE_BYTES_PER_SECOND)
    mid_first = await save_voice_message(
        from_profile_id=pid_alice, from_name="Алиса",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="message one", duration_ms=1000, audio_pcm=audio,
    )
    mid_second = await save_voice_message(
        from_profile_id=pid_alice, from_name="Алиса",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="message two", duration_ms=1000, audio_pcm=audio,
    )
    rows = await list_outgoing_voicemail(pid_alice)
    assert len(rows) == 2
    # Newest first — second insert wins.
    assert rows[0]["id"] == mid_second
    assert rows[1]["id"] == mid_first
    # Recipient sees no outgoing rows under their own id.
    assert await list_outgoing_voicemail(pid_zhenya) == []


# ── Pipeline surfaces unseen reply to sender (Item 1, integration) ─────


async def test_pipeline_replay_unseen_reply_to_sender(make_agent_ctx):
    """Full pipeline run: Alice speaks → memory_context contains Zhenya's reply.

    The first run should inject the reply line; the second run from
    the same speaker should not (we marked it delivered).  Patch
    out everything we don't need: transcribe, _resolve_speaker,
    run_agent (we just want to inspect the call kwargs), and
    memory.EMBEDDING_ENABLED so we don't try to talk to LM Studio.
    """
    from app import memory as pm
    from app.agent import AgentResult

    pid_alice = await save_speaker_profile(
        client_id="cli-vm", name="Алиса", embedding=b"\x01" * (4 * 256),
    )
    pid_zhenya = await save_speaker_profile(
        client_id="cli-vm", name="Женя", embedding=b"\x02" * (4 * 256),
    )
    mid = await save_voice_message(
        from_profile_id=pid_alice, from_name="Алиса",
        to_profile_id=pid_zhenya, to_name="Женя",
        transcript="перезвони", duration_ms=1000,
        audio_pcm=bytes(SAMPLE_BYTES_PER_SECOND),
    )
    await save_voicemail_reply(mid, pid_zhenya, "ок, через час")

    audio = bytes(SAMPLE_BYTES_PER_SECOND)
    fake_result = AgentResult(response_text="хорошо", invocations=[], elapsed_ms=10)
    with patch(
        "app.pipeline.transcribe",
        new=AsyncMock(return_value=("какая погода", 50)),
    ), patch.object(
        Pipeline, "_resolve_speaker",
        new=AsyncMock(return_value=("Алиса", None, pid_alice)),
    ), patch(
        "app.pipeline.run_agent", new=AsyncMock(return_value=fake_result),
    ) as agent, patch.object(pm, "EMBEDDING_ENABLED", False):
        hooks = _VoicemailHooks()
        p = Pipeline(hooks=hooks)
        await p.run(
            audio=audio, session_id=99, client_id="cli-vm",
            history=[], last_response_text="",
        )
    agent.assert_called_once()
    memory_ctx = agent.call_args.kwargs["memory_context"]
    assert "ок, через час" in memory_ctx, (
        f"expected reply text in memory_context, got: {memory_ctx!r}"
    )
    assert "Женя" in memory_ctx, "recipient name should be surfaced"

    # Second run from the same speaker — reply is already delivered, so
    # it must NOT re-surface.  This is the idempotence guarantee that
    # makes the "by the way" injection feel natural instead of nagging.
    # NB: avoid replay-intent phrases like 'ещё раз' ('once more') —
    # those short-circuit before the agent loop and wouldn't exercise
    # the path we care about.
    with patch(
        "app.pipeline.transcribe",
        new=AsyncMock(return_value=("сколько времени", 50)),
    ), patch.object(
        Pipeline, "_resolve_speaker",
        new=AsyncMock(return_value=("Алиса", None, pid_alice)),
    ), patch(
        "app.pipeline.run_agent", new=AsyncMock(return_value=fake_result),
    ) as agent2, patch.object(pm, "EMBEDDING_ENABLED", False):
        await p.run(
            audio=audio, session_id=100, client_id="cli-vm",
            history=[], last_response_text="",
        )
    agent2.assert_called_once()
    memory_ctx2 = agent2.call_args.kwargs["memory_context"]
    assert "ок, через час" not in memory_ctx2, (
        "reply must not re-surface after delivered mark"
    )
