"""Typed web-UI turn: WS ``text_turn`` → Pipeline.run(text=...) without ASR,
speaker-ID or TTS, with the same history bookkeeping as a spoken turn."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.agent import AgentResult
from app.ws import Session, State


class FakeWs:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload):
        self.sent.append(payload)


async def _run_text_turn(session: Session, text: str) -> None:
    await session.handle_message({"type": "text_turn", "text": text})
    await session._pipeline_task


async def test_text_turn_runs_pipeline_without_asr_or_tts(monkeypatch):
    monkeypatch.setattr("app.memory.EMBEDDING_ENABLED", False)
    ws = FakeWs()
    session = Session(ws, session_id=1, client_id="browser-x")
    with patch("app.pipeline.transcribe", new=AsyncMock()) as asr, \
         patch("app.pipeline.speaker.encode_audio_full", new=AsyncMock()) as spk, \
         patch("app.ws.tts.stream") as tts_stream, \
         patch("app.pipeline.run_agent",
               new=AsyncMock(return_value=AgentResult(response_text="hello 👋"))) as agent:
        await _run_text_turn(session, "hi 🙂")
    asr.assert_not_called()
    spk.assert_not_called()
    tts_stream.assert_not_called()
    assert agent.call_args.args[0] == "hi 🙂"
    # Same rules as before: no step-up granted for the web UI.
    assert agent.call_args.kwargs["ctx"].step_up_auth is False
    assert session.history[-2:] == [
        {"role": "user", "content": "hi 🙂"},
        {"role": "assistant", "content": "hello 👋"},
    ]
    types = [m["type"] for m in ws.sent]
    assert {"type": "transcript", "text": "hi 🙂"}.items() <= next(
        m for m in ws.sent if m["type"] == "transcript").items()
    assert any(m["type"] == "response" and m["text"] == "hello 👋" for m in ws.sent)
    assert "text_turn_rejected" not in types
    assert session.state == State.LISTENING_WAKE
    assert session._typed_text is None


async def test_text_turn_rejected_while_busy():
    ws = FakeWs()
    session = Session(ws, session_id=2, client_id="browser-y")
    session.state = State.PROCESSING
    await session.handle_message({"type": "text_turn", "text": "hi"})
    assert ws.sent[-1]["type"] == "text_turn_rejected"
    assert session._typed_text is None
