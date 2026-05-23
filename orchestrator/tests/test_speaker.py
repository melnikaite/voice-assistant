"""
Tests for orchestrator/app/speaker.py — the pure-Python parts.

The encoder (resemblyzer VoiceEncoder) requires PyTorch and a heavy
model download, so we skip ``init_encoder`` / ``encode_audio`` and test
:func:`identify` directly with synthetic embeddings.  identify is the
hot path: voicemail routing, voice-message replay attribution, and the
"latch voice" feature in ws.py all rely on it.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.speaker import SPEAKER_THRESHOLD, identify


def _unit(v: list[float]) -> np.ndarray:
    """Build a unit-norm float32 vector — cosine similarity is then dot product."""
    arr = np.array(v, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


def test_identify_empty_profiles_returns_none():
    """No profiles enrolled → never a match, score 0."""
    q = _unit([1.0, 0.0, 0.0])
    name, score = identify(q, [])
    assert name is None
    assert score == 0.0


def test_identify_exact_match_returns_name():
    """Cosine = 1.0 is always above threshold."""
    q = _unit([1.0, 0.0, 0.0])
    profiles = [("alice", q.copy())]
    name, score = identify(q, profiles)
    assert name == "alice"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_identify_above_threshold_returns_name():
    """Score just above SPEAKER_THRESHOLD must still match."""
    q = _unit([1.0, 0.0, 0.0])
    # Build a profile vector at cos ≈ SPEAKER_THRESHOLD + 0.05 by rotating in 2D.
    target_cos = SPEAKER_THRESHOLD + 0.05
    sin = (1 - target_cos**2) ** 0.5
    p = _unit([target_cos, sin, 0.0])
    name, score = identify(q, [("alice", p)])
    assert name == "alice"
    assert score >= SPEAKER_THRESHOLD


def test_identify_below_threshold_returns_none():
    """Score below SPEAKER_THRESHOLD → unknown, but report the best score."""
    q = _unit([1.0, 0.0, 0.0])
    target_cos = SPEAKER_THRESHOLD - 0.05
    sin = (1 - target_cos**2) ** 0.5
    p = _unit([target_cos, sin, 0.0])
    name, score = identify(q, [("alice", p)])
    assert name is None
    assert score < SPEAKER_THRESHOLD
    # Best score is still surfaced so the caller can log "near miss".
    assert score == pytest.approx(target_cos, abs=1e-4)


def test_identify_picks_best_of_multiple_profiles():
    """When several profiles enroll, the one with the highest cosine wins."""
    q = _unit([1.0, 0.0, 0.0])
    profiles = [
        ("alice", _unit([0.4, 0.9, 0.1])),   # far from q
        ("bob",   _unit([0.95, 0.3, 0.05])), # closest to q
        ("carol", _unit([0.7, 0.7, 0.0])),   # mid
    ]
    name, score = identify(q, profiles)
    assert name == "bob"
    # Sanity: bob's cosine with q is the highest of the three.
    assert score > identify(q, [profiles[0]])[1]
    assert score > identify(q, [profiles[2]])[1]


def test_identify_unnormalised_inputs_still_correct():
    """The function divides by norms internally — magnitudes shouldn't matter."""
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32) * 17.5  # arbitrary scale
    p = np.array([1.0, 0.0, 0.0], dtype=np.float32) * 0.001
    name, score = identify(q, [("alice", p)])
    assert name == "alice"
    assert score == pytest.approx(1.0, abs=1e-4)
