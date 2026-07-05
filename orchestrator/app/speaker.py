"""
Speaker identification via resemblyzer (GE2E d-vectors).

Provides:
  init_encoder()             – load model at startup; sets SPEAKER_ENABLED
  encode_audio(bytes)        – async, returns 256-dim np.ndarray | None
  encode_audio_full(bytes)   – async, returns (mean emb, windowed partials)
  identify(emb, profiles)    – cosine similarity against enrolled profiles
  split_speakers(partials)   – 1 or 2 voice-cluster centroids per utterance
  SpeakerAttribution         – per-turn resolution outcome (single / mixed)

Degrades gracefully: if resemblyzer or torch is missing, SPEAKER_ENABLED stays
False and all callers get None / (None, 0.0).
"""

import asyncio
import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SPEAKER_ENABLED: bool = False

# Minimum cosine similarity to accept a match (0–1).
# 0.75 is conservative: same speaker across sessions is typically 0.80+;
# different speakers are usually below 0.70.
SPEAKER_THRESHOLD: float = float(os.environ.get("SPEAKER_THRESHOLD", "0.75"))

# Minimum audio length (samples at 16 kHz) before we try to embed.
# resemblyzer's windowing needs at least ~1.6 s of speech; anything shorter
# tends to produce unstable embeddings.
_MIN_SAMPLES: int = 16_000  # 1 second

# Floor for the minimum PAIRWISE cosine between windowed partials before
# an utterance counts as one voice.  Same-speaker window pairs sit at
# 0.80+, cross-speaker pairs below ~0.70 (same separation SPEAKER_THRESHOLD
# relies on).  Window-to-MEAN cosine is useless here: for a balanced
# two-voice mix it never drops below √((1+c)/2) ≈ 0.92 at c=0.7, so the
# pairwise minimum is the discriminative signal.  False "mixed" merely
# demotes the turn to shared attribution + a clarifying question; the
# same-profile-clusters and centroid-proximity guards absorb most of them.
MIXED_HOMOGENEITY: float = float(os.environ.get("SPEAKER_MIXED_HOMOGENEITY", "0.70"))

# Below this many ~1.6 s windows a 2-way split is statistical noise —
# short utterances stay single-speaker by construction.
_MIN_PARTIALS_FOR_SPLIT: int = 4

_encoder = None  # VoiceEncoder instance, set by init_encoder()


def init_encoder() -> None:
    """
    Load the GE2E speaker encoder.  Call once at startup.
    On failure (torch/resemblyzer missing) leaves SPEAKER_ENABLED=False.
    """
    global _encoder, SPEAKER_ENABLED
    try:
        from resemblyzer import VoiceEncoder  # type: ignore[import]

        _encoder = VoiceEncoder()
        SPEAKER_ENABLED = True
        log.info("speaker encoder ready (threshold=%.2f)", SPEAKER_THRESHOLD)
    except Exception as exc:
        log.warning("speaker encoder unavailable — speaker ID disabled: %s", exc)


def _embed_sync(audio_bytes: bytes) -> "tuple[np.ndarray, np.ndarray | None]":
    """Blocking: raw int16 PCM (16 kHz, mono) → (mean d-vector, partials).

    ``partials`` are the ~1.6 s windowed embeddings resemblyzer averages
    into the utterance-level vector anyway — surfacing them is free and
    they're what mixed-utterance detection clusters.  None when the clip
    yields too few windows to judge.
    """
    from resemblyzer import preprocess_wav  # type: ignore[import]

    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav = preprocess_wav(pcm, source_sr=16000)
    emb, partials, _splits = _encoder.embed_utterance(  # type: ignore[union-attr]
        wav, return_partials=True
    )
    partials = np.asarray(partials, dtype=np.float32)
    if partials.ndim != 2 or partials.shape[0] < _MIN_PARTIALS_FOR_SPLIT:
        partials = None
    # Canonicalise dtype: resemblyzer currently returns float32, but pin it
    # explicitly so storage (emb.tobytes()) and retrieval (np.frombuffer with
    # dtype=np.float32 in ws.py) stay in lock-step if resemblyzer ever changes.
    return np.asarray(emb, dtype=np.float32), partials


async def encode_audio_full(
    audio_bytes: bytes,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """
    Compute (utterance embedding, windowed partials) from raw PCM bytes.

    Returns (None, None) when:
      - encoder not initialised (SPEAKER_ENABLED is False)
      - clip is shorter than _MIN_SAMPLES (results would be unreliable)
      - any exception occurs during inference
    """
    if not SPEAKER_ENABLED or _encoder is None:
        return None, None
    if len(audio_bytes) // 2 < _MIN_SAMPLES:
        return None, None
    try:
        return await asyncio.to_thread(_embed_sync, audio_bytes)
    except Exception as exc:
        log.warning("speaker: encode_audio failed: %s", exc)
        return None, None


async def encode_audio(audio_bytes: bytes) -> "np.ndarray | None":
    """Utterance-level embedding only — enrollment and replay callers."""
    emb, _ = await encode_audio_full(audio_bytes)
    return emb


def identify(
    query_emb: np.ndarray,
    profiles: list[tuple[str, np.ndarray]],
) -> tuple["str | None", float]:
    """
    Return (speaker_name, best_cosine_score).

    speaker_name is None when:
      - profiles is empty
      - the best match score is below SPEAKER_THRESHOLD
    """
    if not profiles:
        return None, 0.0

    best_name: str | None = None
    best_score = 0.0
    for name, emb in profiles:
        denom = np.linalg.norm(query_emb) * np.linalg.norm(emb)
        score = float(np.dot(query_emb, emb) / (denom + 1e-8))
        if score > best_score:
            best_score, best_name = score, name

    if best_score >= SPEAKER_THRESHOLD:
        return best_name, best_score
    return None, best_score


def split_speakers(partials: np.ndarray) -> "list[np.ndarray]":
    """
    Return ``[mean]`` for a homogeneous utterance, ``[c1, c2]`` when the
    windowed embeddings form two distinct voice clusters.

    Deliberately NOT general diarization: exactly-two split, no timing,
    no overlap handling.  Seeded 2-means on L2-normalised windows where
    the seeds are the most mutually dissimilar window pair.  Single-window
    "clusters" are treated as blips (cough, prosody spike), not speakers.
    """
    norms = np.linalg.norm(partials, axis=1, keepdims=True)
    p = partials / (norms + 1e-8)
    mean = p.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-8)

    # Homogeneity = minimum pairwise window similarity (diagonal is 1.0
    # and never the minimum, so no masking needed).
    sims = p @ p.T
    if float(sims.min()) >= MIXED_HOMOGENEITY:
        return [mean]

    # Seed with the most dissimilar pair, then a few assignment rounds.
    i, j = np.unravel_index(int(np.argmin(sims)), sims.shape)
    c1, c2 = p[i], p[j]
    assign = None
    for _ in range(8):
        new_assign = (p @ c1) >= (p @ c2)
        if assign is not None and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        if bool(assign.all()) or not bool(assign.any()):
            return [mean]
        c1 = p[assign].mean(axis=0)
        c1 = c1 / (np.linalg.norm(c1) + 1e-8)
        c2 = p[~assign].mean(axis=0)
        c2 = c2 / (np.linalg.norm(c2) + 1e-8)

    if min(int(assign.sum()), int((~assign).sum())) < 2:
        return [mean]
    # Cluster centroids that are still same-speaker-close are a prosody
    # swing (whisper→loud, laughter), not two people.
    if float(c1 @ c2) >= MIXED_HOMOGENEITY + 0.1:
        return [mean]
    return [c1, c2]


@dataclass(frozen=True)
class SpeakerAttribution:
    """
    Per-utterance speaker resolution (#59).

    mode:
      "none"           – nobody recognised (no profiles, short clip, or
                         all clusters unknown).  Legacy guest path.
      "single"         – exactly today's behaviour: one recognised voice.
      "mixed_known"    – ≥2 distinct enrolled voices in ONE utterance.
                         No profile attribution; shared memory; device
                         tools ask instead of guessing.
      "mixed_unknown"  – an enrolled voice plus an unrecognised one.
                         Same conservative handling as mixed_known.

    ``participants`` carries the recognised names on mixed turns so the
    agent can phrase the clarifying question ("whose device — Anna or
    Boris?").  The answer arrives through the continuation window as a
    fresh single-speaker turn with its own clean speaker-ID.
    """
    mode: str = "none"
    name: "str | None" = None
    tts_voice: "str | None" = None
    profile_id: "int | None" = None
    participants: "tuple[str, ...]" = ()

    @property
    def is_mixed(self) -> bool:
        return self.mode in ("mixed_known", "mixed_unknown")
