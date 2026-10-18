"""Login picker + code-word bootstrap: GET /api/speakers, setup → login, CLI reset."""
from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from app.storage import create_session, get_session, save_speaker_profile
from app.user_files import _USERS_ROOT, read_settings, set_passphrase

_EMB = b"\x00" * 4 * 256


@pytest.fixture(autouse=True)
def _fresh_users_dir():
    # Profile ids restart at 1 with every fresh DB, so a settings.json left
    # by another test would leak its hash into this one.
    shutil.rmtree(_USERS_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(_USERS_ROOT, ignore_errors=True)


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


async def test_speakers_lists_all_profiles_with_has_passphrase(client):
    a = await save_speaker_profile("browser-a", "Alice", _EMB)
    b = await save_speaker_profile("browser-b", "Bob", _EMB)
    await set_passphrase(a, "amber")

    body = client.get("/api/speakers").json()
    by_id = {s["id"]: s for s in body["speakers"]}
    assert set(by_id) == {a, b}
    assert by_id[a]["has_passphrase"] is True
    assert by_id[b]["has_passphrase"] is False
    assert body["registration_open"] is True
    assert "code_word_hash" not in str(body)


async def test_speakers_client_id_still_scopes_enrollment_panel(client):
    a = await save_speaker_profile("browser-a", "Alice", _EMB)
    await save_speaker_profile("browser-b", "Bob", _EMB)
    body = client.get("/api/speakers", params={"client_id": "browser-a"}).json()
    assert [s["id"] for s in body["speakers"]] == [a]


async def test_setup_then_login_for_profile_without_hash(client):
    pid = await save_speaker_profile("browser-a", "Alice", _EMB)
    r = client.post("/api/auth/setup_passphrase", json={"profile_id": pid, "passphrase": "amber"})
    assert r.status_code == 200
    r = client.post("/api/auth/login", json={"profile_id": pid, "passphrase": "Amber."})
    assert r.status_code == 200
    assert client.get("/api/me").json()["profile_id"] == pid


async def test_setup_refused_when_hash_exists(client):
    pid = await save_speaker_profile("browser-a", "Alice", _EMB)
    await set_passphrase(pid, "amber")
    r = client.post("/api/auth/setup_passphrase", json={"profile_id": pid, "passphrase": "evil"})
    assert r.status_code == 403
    r = client.post("/api/auth/login", json={"profile_id": pid, "passphrase": "amber"})
    assert r.status_code == 200


async def test_cli_reset_clears_hash_and_sessions():
    from app.cli import reset_passphrase

    pid = await save_speaker_profile("browser-a", "Alice", _EMB)
    other = await save_speaker_profile("browser-a", "Bob", _EMB)
    await set_passphrase(pid, "amber")
    token = await create_session(pid)
    other_token = await create_session(other)

    assert await reset_passphrase("alice") == 0
    assert (await read_settings(pid)).code_word_hash is None
    assert await get_session(token) is None
    assert await get_session(other_token) is not None
    assert await reset_passphrase(str(other)) == 0
    assert await reset_passphrase("nobody") == 1
