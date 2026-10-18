"""
Speaker identification via WeSpeaker CAM++ (VoxCeleb, ONNX on onnxruntime).

Provides:
  init_encoder()             – load model at startup; sets SPEAKER_ENABLED
  encode_audio(bytes)        – async, returns 512-dim np.ndarray | None
  encode_audio_full(bytes)   – async, returns (utterance emb, windowed partials)
  identify(emb, profiles)    – cosine similarity against enrolled profiles
  split_speakers(partials)   – 1 or 2 voice-cluster centroids per utterance
  SpeakerAttribution         – per-turn resolution outcome (single / mixed)

Front-end matches WeSpeaker's: 80-bin Kaldi fbank (25/10 ms, no dither) on
int16-scaled PCM, then per-utterance mean normalisation (CMN).

Degrades gracefully: if the model file or onnxruntime is missing,
SPEAKER_ENABLED stays False and all callers get None / (None, 0.0).
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SPEAKER_ENABLED: bool = False

# Stored next to each profile's embedding; profiles enrolled with any other
# model live in a different vector space and are skipped by identify.
MODEL_ID: str = "wespeaker-campplus-voxceleb"

SPEAKER_MODEL_PATH: Path = Path(os.environ.get(
    "SPEAKER_MODEL_PATH",
    str(Path.home() / ".cache/voice-assistant/speaker/voxceleb_CAM++.onnx"),
))

# Minimum cosine similarity to accept a match.  Calibrated on public
# LibriSpeech/MLS/OpenSLR crops (en/de/ru, 80 speakers): with a 4 s enrollment
# against 1.8 s queries 0.55 gives FAR 0.04 % / FRR 8.7 %; 4 s vs 4 s gives
# FAR 0.1 % / FRR 1.7 %.  Biased towards low FAR — a wrong-profile match is
# the costly error, a miss only falls back to guest.  See docs/architecture.md.
SPEAKER_THRESHOLD: float = float(os.environ.get("SPEAKER_THRESHOLD", "0.55"))

# Minimum audio length (samples at 16 kHz) before we try to embed;
# shorter clips give unstable embeddings.
_MIN_SAMPLES: int = 16_000  # 1 second

# Floor for the minimum PAIRWISE cosine between windowed partials before
# an utterance counts as one voice.  Window-to-MEAN cosine is useless here:
# for a balanced two-voice mix it stays high, so the pairwise minimum is the
# discriminative signal.  CAM++ 1.6 s windows on 8 s bench clips: 0.20 flags
# 3.8 % of one-voice clips as mixed and misses 5.4 % of two-voice ones.  False "mixed" merely demotes the turn to shared
# attribution + a clarifying question; the same-profile-clusters and
# centroid-proximity guards absorb most of them.
MIXED_HOMOGENEITY: float = float(os.environ.get("SPEAKER_MIXED_HOMOGENEITY", "0.20"))

# Cluster centroids at least this close are one voice (prosody swing).
MIXED_CENTROID_SAME: float = MIXED_HOMOGENEITY + 0.1

# Partials: 1.6 s fbank windows at 0.8 s hop (frames are 10 ms).
_WIN_FRAMES: int = 160
_HOP_FRAMES: int = 80

# Below this many windows a 2-way split is statistical noise —
# short utterances stay single-speaker by construction.
_MIN_PARTIALS_FOR_SPLIT: int = 4

_session = None  # onnxruntime.InferenceSession, set by init_encoder()
_input_name: str = "feats"


def init_encoder() -> None:
    """
    Load the CAM++ ONNX model.  Call once at startup.
    On failure (file or onnxruntime missing) leaves SPEAKER_ENABLED=False.
    """
    global _session, _input_name, SPEAKER_ENABLED
    try:
        import onnxruntime as ort  # type: ignore[import]

        if not SPEAKER_MODEL_PATH.is_file():
            raise FileNotFoundError(f"{SPEAKER_MODEL_PATH} (see docs/deployment.md)")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(
            str(SPEAKER_MODEL_PATH), opts, providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
        SPEAKER_ENABLED = True
        log.info(
            "speaker encoder ready (model=%s, threshold=%.2f)",
            MODEL_ID, SPEAKER_THRESHOLD,
        )
    except Exception as exc:
        log.warning("speaker encoder unavailable — speaker ID disabled: %s", exc)


def _fbank(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM → (frames, 80) log-mel fbank, Kaldi defaults as in WeSpeaker."""
    import kaldi_native_fbank as knf  # type: ignore[import]

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, pcm.astype(np.float32).tolist())
    fb.input_finished()
    n = fb.num_frames_ready
    return np.array([fb.get_frame(i) for i in range(n)], dtype=np.float32)


def _embed_sync(audio_bytes: bytes) -> "tuple[np.ndarray, np.ndarray | None]":
    """Blocking: raw int16 PCM (16 kHz, mono) → (utterance embedding, partials).

    ``partials`` are 1.6 s window embeddings for mixed-utterance detection,
    computed in one batched call and only when the clip yields enough windows.
    """
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    feats = _fbank(pcm)
    utt = feats - feats.mean(axis=0)
    emb = _session.run(None, {_input_name: utt[None]})[0][0]  # type: ignore[union-attr]

    partials = None
    starts = range(0, len(feats) - _WIN_FRAMES + 1, _HOP_FRAMES)
    if len(starts) >= _MIN_PARTIALS_FOR_SPLIT:
        wins = np.stack([feats[s:s + _WIN_FRAMES] for s in starts])
        wins = wins - wins.mean(axis=1, keepdims=True)
        partials = np.asarray(
            _session.run(None, {_input_name: wins})[0], dtype=np.float32  # type: ignore[union-attr]
        )
    # Storage writes emb.tobytes() and readers use np.frombuffer(float32).
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
    if not SPEAKER_ENABLED or _session is None:
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


def is_compatible(embedding_model: "str | None") -> bool:
    """True when a stored profile embedding is in the current model's space."""
    return embedding_model == MODEL_ID


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
    best_score = -1.0
    for name, emb in profiles:
        if emb.shape != query_emb.shape:
            continue  # other model's vector space — never a match
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
    if float(c1 @ c2) >= MIXED_CENTROID_SAME:
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
