"""
Speaker identification via resemblyzer (GE2E d-vectors).

Provides:
  init_encoder()          – load model at startup; sets SPEAKER_ENABLED
  encode_audio(bytes)     – async, returns 256-dim np.ndarray | None
  identify(emb, profiles) – cosine similarity against enrolled profiles

Degrades gracefully: if resemblyzer or torch is missing, SPEAKER_ENABLED stays
False and all callers get None / (None, 0.0).
"""

import asyncio
import logging
import os

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


def _embed_sync(audio_bytes: bytes) -> np.ndarray:
    """Blocking: convert raw int16 PCM (16 kHz, mono) to a 256-dim d-vector."""
    from resemblyzer import preprocess_wav  # type: ignore[import]

    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav = preprocess_wav(pcm, source_sr=16000)
    emb = _encoder.embed_utterance(wav)  # type: ignore[union-attr]
    # Canonicalise dtype: resemblyzer currently returns float32, but pin it
    # explicitly so storage (emb.tobytes()) and retrieval (np.frombuffer with
    # dtype=np.float32 in ws.py) stay in lock-step if resemblyzer ever changes.
    return np.asarray(emb, dtype=np.float32)


async def encode_audio(audio_bytes: bytes) -> "np.ndarray | None":
    """
    Compute a 256-dim speaker embedding from raw PCM bytes.

    Returns None when:
      - encoder not initialised (SPEAKER_ENABLED is False)
      - clip is shorter than _MIN_SAMPLES (results would be unreliable)
      - any exception occurs during inference
    """
    if not SPEAKER_ENABLED or _encoder is None:
        return None
    if len(audio_bytes) // 2 < _MIN_SAMPLES:
        return None
    try:
        return await asyncio.to_thread(_embed_sync, audio_bytes)
    except Exception as exc:
        log.warning("speaker: encode_audio failed: %s", exc)
        return None


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
