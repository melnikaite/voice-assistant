"""
Mixed-speaker attribution (#59) — split_speakers + _resolve_speaker policy.

The encoder itself (resemblyzer) is heavy and never loaded here; we feed
synthetic windowed embeddings.  Geometry of the fixtures mirrors real
d-vector behaviour: same-voice window pairs sit at cos ≈ 0.95, different
voices at cos ≈ 0.5 — straddling MIXED_HOMOGENEITY (0.70) the same way
real speakers straddle SPEAKER_THRESHOLD.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.pipeline import Pipeline
from app.speaker import MIXED_HOMOGENEITY, SpeakerAttribution, split_speakers
from app.storage import save_speaker_profile

DIM = 8


def _voice(cos_to_e1: float) -> np.ndarray:
    """Unit vector at a chosen cosine to the first basis vector."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cos_to_e1
    v[1] = (1.0 - cos_to_e1**2) ** 0.5
    return v


def _windows(voice: np.ndarray, n: int, jitter: float = 0.15) -> np.ndarray:
    """n windows of one voice — alternating ±jitter along an unused axis,
    so same-voice pairs stay at cos ≈ 0.96."""
    out = []
    for i in range(n):
        w = voice.copy()
        w[3] += jitter if i % 2 == 0 else -jitter
        out.append(w / np.linalg.norm(w))
    return np.array(out, dtype=np.float32)


VOICE_A = _voice(1.0)
VOICE_B = _voice(0.5)   # cross-voice pairs land at cos ≈ 0.5 < MIXED_HOMOGENEITY
VOICE_C = _voice(0.3)


# ── split_speakers ────────────────────────────────────────────────────


def test_homogeneous_utterance_yields_single_cluster():
    clusters = split_speakers(_windows(VOICE_A, 6))
    assert len(clusters) == 1
    assert float(clusters[0] @ VOICE_A) > 0.95


def test_two_voices_split_into_two_clusters():
    partials = np.vstack([_windows(VOICE_A, 4), _windows(VOICE_B, 4)])
    clusters = split_speakers(partials)
    assert len(clusters) == 2
    # Each centroid lands on its own voice, whichever order they come in.
    best_for_a = max(float(c @ VOICE_A) for c in clusters)
    best_for_b = max(float(c @ VOICE_B) for c in clusters)
    assert best_for_a > 0.95
    assert best_for_b > 0.95


def test_single_window_blip_does_not_split():
    """One stray window (cough, overlap artifact) is a blip, not a speaker."""
    partials = np.vstack([_windows(VOICE_A, 7), _windows(VOICE_B, 1)])
    assert len(split_speakers(partials)) == 1


def test_interleaved_voices_still_split():
    """Cluster membership is by similarity, not contiguity — A B A B works."""
    a = _windows(VOICE_A, 4)
    b = _windows(VOICE_B, 4)
    partials = np.vstack([a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]])
    assert len(split_speakers(partials)) == 2


def test_threshold_is_env_tunable_constant():
    """Pin the discriminative geometry: same-voice pairs sit above the
    floor, cross-voice pairs below — if either stops being true the
    fixtures (or the default) drifted."""
    same = _windows(VOICE_A, 2)
    assert float(same[0] @ same[1]) > MIXED_HOMOGENEITY
    assert float(VOICE_A @ VOICE_B) < MIXED_HOMOGENEITY


# ── _resolve_speaker policy ───────────────────────────────────────────


class _NullHooks:
    """_resolve_speaker never touches hooks; constructor just stores them."""


def _task(result):
    async def _coro():
        return result
    return asyncio.ensure_future(_coro())


async def _resolve(client_id: str, emb, partials) -> SpeakerAttribution:
    pipe = Pipeline(_NullHooks())
    return await pipe._resolve_speaker(_task((emb, partials)), client_id, 1)


async def test_mixed_known_two_profiles():
    client = "c-mixed-known"
    await save_speaker_profile(client, "alice", VOICE_A.tobytes())
    await save_speaker_profile(client, "bob", VOICE_B.tobytes())
    partials = np.vstack([_windows(VOICE_A, 4), _windows(VOICE_B, 4)])
    attr = await _resolve(client, partials.mean(axis=0), partials)
    assert attr.mode == "mixed_known"
    assert set(attr.participants) == {"alice", "bob"}
    assert attr.profile_id is None
    assert attr.is_mixed


async def test_mixed_unknown_one_profile_one_stranger():
    client = "c-mixed-unknown"
    await save_speaker_profile(client, "alice", VOICE_A.tobytes())
    partials = np.vstack([_windows(VOICE_A, 4), _windows(VOICE_C, 4)])
    attr = await _resolve(client, partials.mean(axis=0), partials)
    assert attr.mode == "mixed_unknown"
    assert attr.participants == ("alice",)
    assert attr.profile_id is None


async def test_prosody_clusters_same_profile_stay_single():
    """Two clusters that both match the SAME profile are one person with
    prosody variation — the false-split guard must keep the single path."""
    client = "c-prosody"
    # Enroll the centroid between the two prosody clusters: both clusters
    # match it at cos ≈ 0.89 ≥ SPEAKER_THRESHOLD while the clusters
    # themselves sit at cos 0.5 (split fires).
    p1, p2 = VOICE_A, VOICE_B
    centroid = (p1 + p2) / np.linalg.norm(p1 + p2)
    await save_speaker_profile(client, "alice", centroid.astype(np.float32).tobytes())
    partials = np.vstack([_windows(p1, 4), _windows(p2, 4)])
    attr = await _resolve(client, centroid.astype(np.float32), partials)
    assert attr.mode == "single"
    assert attr.name == "alice"
    assert attr.profile_id is not None


async def test_all_unknown_clusters_fall_back_to_none():
    client = "c-strangers"
    await save_speaker_profile(client, "alice", _voice(-0.5).tobytes())
    partials = np.vstack([_windows(VOICE_A, 4), _windows(VOICE_B, 4)])
    attr = await _resolve(client, partials.mean(axis=0), partials)
    assert attr.mode == "none"
    assert attr.participants == ()


async def test_no_partials_keeps_single_speaker_path():
    """Short clips (partials=None) resolve exactly as before #59."""
    client = "c-short"
    await save_speaker_profile(client, "alice", VOICE_A.tobytes())
    attr = await _resolve(client, VOICE_A, None)
    assert attr.mode == "single"
    assert attr.name == "alice"
