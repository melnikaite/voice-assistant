"""CAM++ speaker model: front-end, model-id compatibility, re-enrollment, API flag.

The ONNX session is mocked — no model file is needed.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import speaker
from app.pipeline import Pipeline
from app.storage import get_speaker_profile_by_name, save_speaker_profile

DIM = 512


def _unit(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


class _FakeSession:
    """Records input shapes; returns a fixed embedding per batch row."""

    def __init__(self, emb: np.ndarray):
        self.emb = emb
        self.calls: list[tuple[int, ...]] = []

    def run(self, _outputs, feeds):
        x = feeds["feats"]
        self.calls.append(x.shape)
        return [np.tile(self.emb, (x.shape[0], 1))]


@pytest.fixture
def fake_encoder(monkeypatch):
    sess = _FakeSession(_unit(0) * 3.0)
    monkeypatch.setattr(speaker, "_session", sess)
    monkeypatch.setattr(speaker, "_input_name", "feats")
    monkeypatch.setattr(speaker, "SPEAKER_ENABLED", True)
    return sess


def _pcm(seconds: float) -> bytes:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(16000 * seconds)) * 3000).astype(np.int16).tobytes()


# ── front-end ─────────────────────────────────────────────────────────


async def test_short_clip_embeds_without_partials(fake_encoder):
    emb, partials = await speaker.encode_audio_full(_pcm(2.0))
    assert emb.shape == (DIM,) and emb.dtype == np.float32
    assert partials is None
    # One call: (batch=1, ~198 frames, 80 mel bins).
    assert len(fake_encoder.calls) == 1
    b, t, f = fake_encoder.calls[0]
    assert (b, f) == (1, 80) and 190 <= t <= 200


async def test_long_clip_adds_batched_partials(fake_encoder):
    _, partials = await speaker.encode_audio_full(_pcm(6.0))
    assert partials is not None and partials.shape[1] == DIM
    assert fake_encoder.calls[1][0] == partials.shape[0] >= 4
    assert fake_encoder.calls[1][1] == speaker._WIN_FRAMES


def test_fbank_is_mean_normalised_before_the_model(monkeypatch):
    seen = {}

    class _S:
        def run(self, _o, feeds):
            seen["x"] = feeds["feats"]
            return [np.zeros((feeds["feats"].shape[0], DIM), np.float32)]

    monkeypatch.setattr(speaker, "_session", _S())
    speaker._embed_sync(_pcm(2.0))
    assert np.allclose(seen["x"][0].mean(axis=0), 0.0, atol=1e-3)


async def test_too_short_clip_is_rejected(fake_encoder):
    assert await speaker.encode_audio(_pcm(0.5)) is None
    assert fake_encoder.calls == []


# ── threshold / compatibility ─────────────────────────────────────────


def test_default_threshold_is_campplus_scale():
    assert speaker.SPEAKER_THRESHOLD == pytest.approx(0.55)


def test_threshold_applied_on_campplus_scores():
    q = _unit(0)
    near = (0.6 * _unit(0) + 0.8 * _unit(1)).astype(np.float32)  # cos 0.60
    far = (0.5 * _unit(0) + np.sqrt(0.75) * _unit(1)).astype(np.float32)  # cos 0.50
    assert speaker.identify(q, [("alice", near)])[0] == "alice"
    assert speaker.identify(q, [("alice", far)])[0] is None


def test_identify_skips_other_dimension():
    """A 256-dim resemblyzer vector never matches, even if prefixes agree."""
    name, _ = speaker.identify(_unit(0), [("old", _unit(0)[:256].copy())])
    assert name is None


def _task(result):
    async def _coro():
        return result
    return asyncio.ensure_future(_coro())


async def test_profile_from_other_model_is_skipped():
    client = "c-legacy"
    emb = _unit(0)
    await save_speaker_profile(client, "eugene", emb.tobytes(), model="resemblyzer")
    attr = await Pipeline(object())._resolve_speaker(_task((emb, None)), client, 1)
    assert attr.mode == "none" and attr.profile_id is None


async def test_compatible_profile_matches_next_to_legacy_one():
    client = "c-mixed-models"
    emb = _unit(0)
    await save_speaker_profile(client, "legacy", emb.tobytes(), model="resemblyzer")
    pid = await save_speaker_profile(client, "eugene", emb.tobytes())
    attr = await Pipeline(object())._resolve_speaker(_task((emb, None)), client, 1)
    assert (attr.mode, attr.name, attr.profile_id) == ("single", "eugene", pid)


# ── API: flag + re-enrollment ─────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    from app.main import app

    with TestClient(app) as c:
        monkeypatch.setattr(speaker, "SPEAKER_ENABLED", True)
        yield c


def _enroll(client, monkeypatch, emb: np.ndarray, name: str = "Eugene"):
    async def _fake(_audio):
        return emb

    monkeypatch.setattr(speaker, "encode_audio", _fake)
    return client.post(
        "/api/speakers/enroll",
        params={"client_id": "browser", "name": name},
        files={"audio": ("a.pcm", b"\x00" * 64000)},
    )


async def test_list_flags_legacy_profiles(client):
    await save_speaker_profile("browser", "Old", b"\x00" * 1024, model=None)
    await save_speaker_profile("browser", "New", _unit(0).tobytes())
    by_name = {s["name"]: s for s in client.get("/api/speakers?client_id=browser").json()["speakers"]}
    assert by_name["Old"]["needs_reenroll"] is True
    assert by_name["New"]["needs_reenroll"] is False


async def test_reenroll_replaces_legacy_embedding(client, monkeypatch):
    pid = await save_speaker_profile("browser", "Eugene", b"\x01" * 1024, model=None)
    new = _unit(5) * 2.0
    r = _enroll(client, monkeypatch, new).json()
    assert (r["id"], r["samples"]) == (pid, 1)
    _, blob, n, model = await get_speaker_profile_by_name("browser", "Eugene")
    assert model == speaker.MODEL_ID and n == 1
    assert np.array_equal(np.frombuffer(blob, np.float32), new)


async def test_reenroll_same_model_still_averages(client, monkeypatch):
    await save_speaker_profile("browser", "Eugene", _unit(0).tobytes())
    r = _enroll(client, monkeypatch, _unit(1) * 4.0).json()  # unnormalised model output
    assert r["samples"] == 2
    _, blob, _, _ = await get_speaker_profile_by_name("browser", "Eugene")
    v = np.frombuffer(blob, np.float32)
    assert v[0] == pytest.approx(v[1]) and np.linalg.norm(v) == pytest.approx(1.0)
