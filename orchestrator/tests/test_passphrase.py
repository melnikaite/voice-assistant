"""
Passphrase normalization round-trip.

A bug in normalization either locks legitimate users out (we accept
"Янтарь" at set time, refuse "янтарь" at verify) or accepts wrong
words (looser-than-intended matching).  We check every case the
production ``_normalise_passphrase`` covers:

* case-insensitive
* trailing punctuation stripped (.,!?;: \\t\\n)
* surrounding whitespace stripped
* genuine mismatches still rejected
"""
from __future__ import annotations

import pytest

from app.storage import save_speaker_profile
from app.user_files import set_passphrase, verify_passphrase


@pytest.fixture
async def profile_id() -> int:
    # Speaker profiles are the FK target for settings.json — create one
    # so set/verify_passphrase have a real row to write against.
    return await save_speaker_profile(
        client_id="cli-passphrase", name="Test Speaker", embedding=b"\x00" * 4 * 256
    )


async def test_set_then_verify_exact(profile_id):
    await set_passphrase(profile_id, "Янтарь")
    assert await verify_passphrase(profile_id, "Янтарь") is True


async def test_case_insensitive(profile_id):
    await set_passphrase(profile_id, "Янтарь")
    assert await verify_passphrase(profile_id, "янтарь") is True
    assert await verify_passphrase(profile_id, "ЯНТАРЬ") is True


async def test_trailing_punctuation_stripped(profile_id):
    """Whisper often appends a period or comma — normalize away."""
    await set_passphrase(profile_id, "янтарь")
    assert await verify_passphrase(profile_id, "Янтарь.") is True
    assert await verify_passphrase(profile_id, "янтарь,") is True
    assert await verify_passphrase(profile_id, "янтарь!") is True
    assert await verify_passphrase(profile_id, "янтарь?") is True


async def test_surrounding_whitespace(profile_id):
    await set_passphrase(profile_id, "янтарь")
    assert await verify_passphrase(profile_id, "  янтарь  ") is True
    assert await verify_passphrase(profile_id, "\tянтарь\n") is True


async def test_genuine_mismatch_rejected(profile_id):
    await set_passphrase(profile_id, "янтарь")
    assert await verify_passphrase(profile_id, "jantar") is False
    assert await verify_passphrase(profile_id, "янтарь жёлтый") is False
    assert await verify_passphrase(profile_id, "") is False


async def test_no_passphrase_set_returns_false():
    """Verify against a profile that never had a passphrase = always False."""
    pid = await save_speaker_profile(
        client_id="cli-no-pp", name="Empty", embedding=b"\x00" * 4 * 256,
    )
    assert await verify_passphrase(pid, "anything") is False
