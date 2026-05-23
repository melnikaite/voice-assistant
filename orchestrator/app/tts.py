"""
TTS — thin HTTP client for the host-side xtts-server.

The orchestrator doesn't synthesise speech itself.  All TTS work
happens in a separate process on the host (see ../xtts-server/), which
runs XTTS-v2 on Apple Silicon MPS / CUDA and streams PCM over HTTP.

This module exposes the same interface (`synth`, `stream`,
`init_voices`) as the previous in-container Piper / coqui-tts
implementations, so the rest of the codebase didn't have to change
when we moved the engine out.

Pattern matches our other host services:
    mlx-whisper  :18000  (ASR)
    lm-studio    :1234   (LLM)
    xtts-server  :9000   (TTS, this module's target)
"""

from __future__ import annotations

import logging
import os
import re
from typing import AsyncIterator

import httpx
import numpy as np

log = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────

# Where xtts-server listens.  Default assumes both processes are on the
# same host with the container in network_mode: host (see compose file).
TTS_URL = os.environ.get("TTS_URL", "http://localhost:9876")

# Host-side absolute path of the bind-mounted data dir.  The container
# sees this as ``DATA_DIR_CONTAINER`` (/data by default) but xtts-server
# runs on the host and needs the real on-host path so it can read the
# reference WAV for voice cloning.  Compose resolves ${PWD}/data at
# startup time — see docker-compose.yml.
DATA_DIR_HOST = os.environ.get("DATA_DIR_HOST", "/data")
DATA_DIR_CONTAINER = os.environ.get("DATA_DIR_CONTAINER", "/data")

# Per-request default speaker.  The host server has its own default but
# we can override per call if we want different voices in different
# situations (e.g. a child-friendly voice for reminders).  Leave None
# to use whatever the server is configured with.
DEFAULT_SPEAKER = os.environ.get("XTTS_SPEAKER", "") or None

# XTTS streaming granularity.  20 is balanced; lower for faster first-
# byte but choppier prosody, higher for smoother output but later start.
STREAM_CHUNK_SIZE = int(os.environ.get("XTTS_STREAM_CHUNK_SIZE", "20"))

# Native XTTS rate — clients downstream resample to whatever audio sink
# wants (48 kHz for WebRTC Opus).  This is the rate of the int16 frames
# yielded by stream() / returned by synth().
TTS_SAMPLE_RATE = 24000

# Per-request timeout.  Synth can take a while for a long response on
# slower hosts, but with MPS we expect ~realtime so a few minutes is
# generous.  Streaming reads start showing data within seconds anyway.
_REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=10.0)


# ─── Cheap fallback language guess ────────────────────────────────────

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_UMLAUT_RE = re.compile(r"[äöüÄÖÜß]")


def _guess_lang(text: str) -> str:
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _UMLAUT_RE.search(text):
        return "de"
    return "en"


def _normalise_lang(lang: str | None, text: str) -> str:
    if lang and len(lang) == 2 and lang.isalpha():
        return lang.lower()
    return _guess_lang(text)


# ─── Public API ───────────────────────────────────────────────────────


_voices_cache: list[str] | None = None


async def voices(force_refresh: bool = False) -> list[str]:
    """
    Return the list of XTTS speaker names the host server advertises.

    Cached in-process for the lifetime of the orchestrator on the first
    *successful* fetch — XTTS's built-in speaker bank is fixed at model
    load time, so once we've got it we're done.  Empty results (server
    still warming up, network blip) are NOT cached, so the next caller
    will retry.  `force_refresh=True` ignores the cache and re-fetches.
    """
    global _voices_cache
    if _voices_cache and not force_refresh:
        return _voices_cache
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{TTS_URL}/v1/speakers")
            r.raise_for_status()
            fetched = list(r.json().get("speakers", []))
    except Exception as exc:
        log.warning("tts: could not fetch /v1/speakers (%s)", exc)
        return []
    if fetched:
        _voices_cache = fetched
    return fetched


async def init_voices() -> None:
    """Probe the host xtts-server.  Non-fatal if it's down — synth calls
    will just log per-request errors so the user can see what's missing.

    Called once at orchestrator startup so the operator gets immediate
    feedback (in the orchestrator log) about TTS reachability instead of
    having to wait for their first voice interaction to discover it.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{TTS_URL}/v1/health")
            r.raise_for_status()
            info = r.json()
        log.info(
            "tts: ready at %s (device=%s, speaker=%r, sr=%d)",
            TTS_URL,
            info.get("device"),
            info.get("default_speaker"),
            info.get("sample_rate"),
        )
    except Exception as exc:
        log.warning(
            "tts: server unreachable at %s (%s) — start xtts-server "
            "on the host (`cd xtts-server && ./start.sh`) to enable audio",
            TTS_URL, exc,
        )


def _to_host_path(container_path: str) -> str:
    """Map a container-side absolute path under DATA_DIR_CONTAINER to its
    host-side equivalent so xtts-server (which runs on the host) can
    read the file.

    If the path doesn't sit under the container data dir, it's returned
    unchanged — xtts-server will still try it as a literal absolute
    path; useful for one-off curl debugging with manually-placed WAVs.
    """
    if container_path.startswith(DATA_DIR_CONTAINER.rstrip("/") + "/"):
        rel = container_path[len(DATA_DIR_CONTAINER.rstrip("/")) + 1:]
        return f"{DATA_DIR_HOST.rstrip('/')}/{rel}"
    if container_path == DATA_DIR_CONTAINER:
        return DATA_DIR_HOST
    return container_path


async def _resolve_voice(voice: str | None) -> tuple[str | None, str | None]:
    """Translate a `voice` string into (xtts_speaker_name, reference_audio_path).

    Cases:
      • ``None``               → (None, None) — server default kicks in.
      • ``"clone:<id>"``       → look up the custom voice row; on miss
        we fall through to the server default (None, None) and log.
      • anything else          → treat as a built-in XTTS speaker name.
    """
    if not voice:
        return None, None
    if voice.startswith("clone:"):
        try:
            voice_id = int(voice[len("clone:"):])
        except ValueError:
            log.warning("tts: malformed clone voice %r", voice)
            return None, None
        # Local import to avoid a circular dependency at module load
        # (storage package re-exports many names but doesn't depend on
        # tts; still, keep the dependency shallow).
        from .storage import get_custom_voice_by_id

        row = await get_custom_voice_by_id(voice_id)
        if row is None:
            log.warning("tts: custom voice id=%d not found, using default", voice_id)
            return None, None
        _id, _name, wav_path = row
        return None, _to_host_path(wav_path)
    return voice, None


async def stream(
    text: str,
    *,
    lang: str | None = None,
    voice: str | None = None,
    session_id: str | None = None,
    reference_audio_path: str | None = None,
) -> AsyncIterator[np.ndarray]:
    """Stream int16 mono PCM chunks (at TTS_SAMPLE_RATE) from xtts-server.

    Each yielded chunk is a 1-D numpy array of int16 samples; concatenate
    them downstream to get the full utterance.  The HTTP response is
    chunked transfer-encoding, so the first chunk arrives ~300-500 ms
    after the request lands (MPS) and we forward to the RTC outbound
    track immediately for a snappy first-audio feel.

    ``voice`` overrides the default XTTS speaker for this call only —
    used so each identified household member can have their own voice
    (see speaker_profiles.tts_voice).  None falls back to the server's
    configured default.  ``voice`` may also be ``"clone:<id>"`` — a
    custom-voice row from the storage layer; we look up its WAV path
    and pass it as ``reference_audio`` to xtts-server.

    ``reference_audio_path`` lets callers pass an explicit host-side
    WAV path; it takes precedence over the voice string when both are
    supplied.

    ``session_id`` keys the cancel-isolation bucket on xtts-server.
    The orchestrator passes the WebRTC client_id here so that a new
    turn from device A pre-empts only its own in-flight TTS, never the
    TTS playing on device B.

    On any HTTP error we log and return early — caller treats it as
    "no audio" and the conversational FSM still progresses.
    """
    if not text or not text.strip():
        return
    lang_code = _normalise_lang(lang, text)
    payload: dict = {
        "text": text,
        "lang": lang_code,
        "stream_chunk_size": STREAM_CHUNK_SIZE,
    }

    # Resolution order: explicit reference_audio_path > voice ("clone:<id>"
    # or built-in name) > server default.
    speaker_name, ref_from_voice = await _resolve_voice(voice)
    ref_audio = reference_audio_path or ref_from_voice
    chosen_voice = speaker_name or DEFAULT_SPEAKER

    if ref_audio:
        payload["reference_audio"] = ref_audio
    if chosen_voice:
        payload["speaker"] = chosen_voice
    if session_id:
        payload["session_id"] = session_id

    log.info(
        "tts: requesting synth lang=%s voice=%s clone=%s session=%s, %d chars",
        lang_code,
        chosen_voice or "<server-default>",
        ref_audio or "—",
        session_id or "<default>",
        len(text),
    )

    # `aiter_bytes` returns variably-sized byte slices — we don't assume
    # they align on sample boundaries.  Buffer the tail until we have at
    # least two bytes (one int16 sample) before yielding a NumPy view.
    tail = b""
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST", f"{TTS_URL}/v1/synthesize", json=payload
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(8192):
                    if not chunk:
                        continue
                    data = tail + chunk
                    n_samples = len(data) // 2
                    if n_samples == 0:
                        tail = data
                        continue
                    even = n_samples * 2
                    arr = np.frombuffer(data[:even], dtype=np.int16).copy()
                    tail = data[even:]
                    yield arr
        if tail:
            log.debug("tts: trailing %d unaligned byte(s) dropped", len(tail))
    except httpx.HTTPError as exc:
        log.warning("tts: synth HTTP error (%s)", exc.__class__.__name__)
    except Exception:
        log.exception("tts: streaming synth crashed")


async def synth(
    text: str,
    *,
    lang: str | None = None,
    voice: str | None = None,
    session_id: str | None = None,
    reference_audio_path: str | None = None,
) -> tuple[int, np.ndarray]:
    """Non-streaming convenience: collect every chunk into one array.

    Same interface the WS layer has been using all along, so swapping
    engines (piper → coqui in-container → host xtts-server) doesn't
    touch ws.py.
    """
    if not text or not text.strip():
        return TTS_SAMPLE_RATE, np.zeros(0, dtype=np.int16)
    chunks: list[np.ndarray] = []
    async for chunk in stream(
        text,
        lang=lang,
        voice=voice,
        session_id=session_id,
        reference_audio_path=reference_audio_path,
    ):
        chunks.append(chunk)
    if not chunks:
        return TTS_SAMPLE_RATE, np.zeros(0, dtype=np.int16)
    return TTS_SAMPLE_RATE, np.concatenate(chunks)


def sample_rate_for(_lang: str | None = None) -> int:
    """Constant — XTTS emits 24 kHz regardless of language."""
    return TTS_SAMPLE_RATE
