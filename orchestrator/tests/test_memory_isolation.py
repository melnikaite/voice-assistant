"""Semantic-memory candidates never cross profiles, even for an unidentified speaker."""
from __future__ import annotations

import time
import uuid

from app.storage import get_candidate_utterances, get_speaker_profile_name, save_speaker_profile, save_utterance, start_session


async def _seed() -> str:
    client_id = f"tablet-{uuid.uuid4().hex[:8]}"
    sid = await start_session(client_id=client_id)
    now = time.time()
    for text, speaker, shared in [
        ("alice private", "Alice", 0),
        ("bob private", "Bob", 0),
        ("alice shared", "Alice", 1),
        ("unattributed", None, 0),
    ]:
        await save_utterance(
            session_id=sid, ts=now, transcript=text, response_text="ok",
            embedding=b"\x00" * 16, speaker_name=speaker, is_shared=shared,
        )
    return client_id


async def test_unidentified_speaker_sees_only_shared_and_unattributed():
    client_id = await _seed()
    rows = await get_candidate_utterances(client_id, 0, speaker_name=None)
    assert {r[0] for r in rows} == {"alice shared", "unattributed"}


async def test_identified_speaker_sees_own_plus_shared():
    client_id = await _seed()
    rows = await get_candidate_utterances(client_id, 0, speaker_name="Bob")
    assert {r[0] for r in rows} == {"bob private", "alice shared", "unattributed"}


async def test_profile_name_lookup():
    pid = await save_speaker_profile(f"b-{uuid.uuid4().hex[:6]}", "Carol", b"\x00" * 4 * 192)
    assert await get_speaker_profile_name(pid) == "Carol"
    assert await get_speaker_profile_name(999_999) is None
