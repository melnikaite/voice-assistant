#!/usr/bin/env python3
"""
XTTS-v2 host service.

This runs DIRECTLY ON THE HOST (not in Docker) so it can use Apple
Silicon's MPS backend (or CUDA on Linux/Windows) for ~3-5× faster
inference than CPU-only Docker.  Same architectural pattern as
mlx-whisper on :18000 and LM Studio on :1234 — the voice-assistant
orchestrator just talks HTTP.

Endpoints:
  GET  /v1/health        — liveness + device info
  GET  /v1/speakers      — list of 58 built-in speaker names
  POST /v1/synthesize    — streams int16 mono PCM @ 24 kHz

Model + speaker file (~1.9 GB total) live in
  $XTTS_MODEL_DIR or ~/.cache/voice-assistant/xtts/
and are downloaded once on first startup.  Persists across restarts.

Start:
    python3 xtts-server.py        # uses ./venv if present
or:
    ./start.sh

License (XTTS-v2): CPML — non-commercial OSS.  Auto-accepted via
COQUI_TOS_AGREED=1 below.  Personal use only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import AsyncIterator

# License acceptance — must be set BEFORE importing TTS.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("xtts-server")


# ─── Config (env-overridable) ──────────────────────────────────────────

MODEL_DIR = Path(
    os.environ.get(
        "XTTS_MODEL_DIR",
        str(Path.home() / ".cache" / "voice-assistant" / "xtts"),
    )
).expanduser()

# Default speaker if a request doesn't override.  See /v1/speakers for
# the full list — XTTS ships 58 built-in voices.
DEFAULT_SPEAKER = os.environ.get("XTTS_SPEAKER", "Claribel Dervla")

# Device selection: "auto" picks the best available (mps > cuda > cpu).
# Override with "mps", "cuda", or "cpu" to pin.
DEVICE = os.environ.get("XTTS_DEVICE", "auto")

PORT = int(os.environ.get("XTTS_PORT", "9876"))
HOST_BIND = os.environ.get("XTTS_HOST", "127.0.0.1")

# Native sample rate of XTTS-v2 output — clients should expect this.
SAMPLE_RATE = 24000

# Files we need to fetch from huggingface on first start.
HF_BASE = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
MODEL_FILES = ["config.json", "vocab.json", "model.pth", "speakers_xtts.pth"]


# ─── State ────────────────────────────────────────────────────────────

_model = None
_gpt_cond_latent = None
_speaker_embedding = None
_device_name = "cpu"

# Per-session cancel registry.  Keyed by the caller-supplied `session_id`
# field on /v1/synthesize.  When a new request comes in for a session
# that's already in flight, we .set() the old Event and replace it —
# that's the "user said something new, drop the half-finished response"
# barge-in.  Different sessions are isolated: device A's new turn does
# NOT abort device B's TTS that's playing in parallel.
#
# Without this, orchestrator-side request cancellation can't kill the
# Python thread doing inference — it would keep running for 1-3 s after
# the client disconnected, eating MPS while the new sentence's
# inference is already starting in parallel.
#
# Requests that omit session_id share the bucket "default" — same
# semantics as the old global flag, only used by ad-hoc curl tests.
_cancels: dict[str, threading.Event] = {}
_cancels_lock = threading.Lock()


def _claim_cancel_slot(session_key: str) -> threading.Event:
    """
    Register a fresh cancel-Event for ``session_key``.  If that session
    already has an in-flight producer, signal it to stop within one
    chunk boundary (~150 ms).  Returns the new Event the caller should
    poll inside its own producer loop.
    """
    new_event = threading.Event()
    with _cancels_lock:
        old = _cancels.get(session_key)
        if old is not None and not old.is_set():
            log.info(
                "synth: pre-empting in-flight inference for session=%s",
                session_key,
            )
            old.set()
        _cancels[session_key] = new_event
    return new_event


def _release_cancel_slot(session_key: str, my_event: threading.Event) -> None:
    """
    Drop our slot on the way out, but only if we still own it — a newer
    request may have replaced our Event already, and we mustn't free
    that one.
    """
    with _cancels_lock:
        if _cancels.get(session_key) is my_event:
            _cancels.pop(session_key, None)


# ─── Device picker ────────────────────────────────────────────────────

def _detect_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ─── Model download (one-time on fresh install) ───────────────────────

async def _download(url: str, dest: Path) -> None:
    """Atomic streaming download via .part temp file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    async with httpx.AsyncClient(timeout=900, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                async for chunk in resp.aiter_bytes(256 * 1024):
                    f.write(chunk)
    tmp.replace(dest)
    log.info("downloaded %s (%.0f MB)", dest.name, dest.stat().st_size / 1_048_576)


async def _ensure_model_on_disk() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MODEL_FILES:
        target = MODEL_DIR / fname
        if target.exists() and target.stat().st_size > 1024:
            continue
        url = f"{HF_BASE}/{fname}"
        log.info("downloading %s …", fname)
        await _download(url, target)


# ─── Model load (blocking, runs in a thread) ──────────────────────────

def _load_model_sync() -> None:
    global _model, _gpt_cond_latent, _speaker_embedding, _device_name

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    _device_name = _detect_device()
    log.info("loading XTTS-v2 on device=%s from %s", _device_name, MODEL_DIR)

    config = XttsConfig()
    config.load_json(str(MODEL_DIR / "config.json"))
    _model = Xtts.init_from_config(config)
    _model.load_checkpoint(
        config,
        checkpoint_path=str(MODEL_DIR / "model.pth"),
        vocab_path=str(MODEL_DIR / "vocab.json"),
        speaker_file_path=str(MODEL_DIR / "speakers_xtts.pth"),
        use_deepspeed=False,
    )
    if _device_name == "mps":
        _model.to(torch.device("mps"))
    elif _device_name == "cuda":
        _model.cuda()
    else:
        _model.cpu()
    _model.eval()

    speakers = _model.speaker_manager.speakers
    speaker_name = DEFAULT_SPEAKER
    if speaker_name not in speakers:
        speaker_name = next(iter(speakers))
        log.warning(
            "default speaker %r not found; falling back to %r",
            DEFAULT_SPEAKER, speaker_name,
        )
    s = speakers[speaker_name]
    _gpt_cond_latent = s["gpt_cond_latent"]
    _speaker_embedding = s["speaker_embedding"]
    log.info(
        "XTTS ready: device=%s, default_speaker=%r (out of %d)",
        _device_name, speaker_name, len(speakers),
    )


# ─── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(title="xtts-server", version="1.0")


@app.on_event("startup")
async def startup() -> None:
    await _ensure_model_on_disk()
    await asyncio.to_thread(_load_model_sync)


@app.get("/v1/health")
async def health() -> dict:
    with _cancels_lock:
        inflight = len(_cancels)
    return {
        "ok": _model is not None,
        "device": _device_name,
        "sample_rate": SAMPLE_RATE,
        "default_speaker": DEFAULT_SPEAKER,
        "model_dir": str(MODEL_DIR),
        "inflight_sessions": inflight,
    }


@app.get("/v1/speakers")
async def speakers() -> dict:
    if _model is None:
        raise HTTPException(503, "model not ready")
    return {"speakers": sorted(_model.speaker_manager.speakers.keys())}


class SynthRequest(BaseModel):
    text: str
    lang: str = "en"
    speaker: str | None = None
    # Absolute host path to a 6-12 s mono WAV; when set, overrides
    # `speaker` and uses on-the-fly voice cloning instead.
    reference_audio: str | None = None
    # Streaming chunk granularity.  Smaller → lower first-byte latency,
    # larger → smoother prosody.  20 is a good middle ground.
    stream_chunk_size: int = 20
    # Cancel-isolation key.  Two callers with different session_id values
    # run their syntheses concurrently; a second call with the *same*
    # session_id pre-empts the first one (barge-in within the same
    # voice channel).  Omit for curl tests; the orchestrator passes
    # client_id here.
    session_id: str | None = None


@app.post("/v1/synthesize")
async def synthesize(req: SynthRequest):
    """Stream int16 mono PCM @ SAMPLE_RATE Hz back as the model produces it.

    Response is chunked Transfer-Encoding with `audio/pcm` content type;
    headers carry the format details so the client can wire it into its
    audio pipeline without parsing a WAV header (we don't send one).

    Pre-emption: a new request **on the same session_id** immediately
    signals the previous in-flight inference to abort.  The previous
    Python thread sees the flag, breaks out of its yield loop within
    one chunk-size (~150 ms), and releases MPS.  Requests on different
    session_ids do not pre-empt each other — they serialise naturally
    on MPS (it's single-stream) but neither cancels the other.
    """
    if _model is None:
        raise HTTPException(503, "model not ready")
    if not req.text.strip():
        raise HTTPException(400, "empty text")

    session_key = req.session_id or "default"
    my_cancel = _claim_cancel_slot(session_key)

    # Resolve speaker conditioning: precedence is reference_audio >
    # speaker name > default speaker.
    gpt_cond_latent = _gpt_cond_latent
    speaker_embedding = _speaker_embedding
    chosen_speaker = "<default>"
    if req.reference_audio and Path(req.reference_audio).exists():
        try:
            gpt_cond_latent, speaker_embedding = _model.get_conditioning_latents(
                audio_path=[req.reference_audio]
            )
            chosen_speaker = f"clone:{req.reference_audio}"
        except Exception as e:
            log.warning("voice clone failed (%s) — using default", e)
    elif req.speaker:
        speakers_map = _model.speaker_manager.speakers
        if req.speaker in speakers_map:
            s = speakers_map[req.speaker]
            gpt_cond_latent = s["gpt_cond_latent"]
            speaker_embedding = s["speaker_embedding"]
            chosen_speaker = req.speaker
        else:
            log.warning(
                "unknown speaker %r — using default %r", req.speaker, DEFAULT_SPEAKER,
            )

    log.info(
        "synth lang=%s speaker=%s, %d chars",
        req.lang, chosen_speaker, len(req.text),
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _producer() -> None:
        """XTTS inference loop — runs in a worker thread so it doesn't
        block the asyncio event loop.  Each yielded chunk is converted
        to int16 PCM bytes and pushed onto the queue via a
        thread-safe loop call.

        Checks ``my_cancel`` between chunks so a newer /v1/synthesize
        request can pre-empt this one quickly.
        """
        chunks_emitted = 0
        try:
            for chunk in _model.inference_stream(
                text=req.text,
                language=req.lang.lower(),
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=True,
                stream_chunk_size=req.stream_chunk_size,
            ):
                if my_cancel.is_set():
                    log.info(
                        "synth: cancelled by newer request after %d chunks",
                        chunks_emitted,
                    )
                    break
                arr = chunk.detach().cpu().numpy().reshape(-1)
                pcm = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
                loop.call_soon_threadsafe(queue.put_nowait, pcm.tobytes())
                chunks_emitted += 1
        except Exception:
            log.exception("synth producer crashed")
        finally:
            # Release our slot in the cancel registry so it doesn't pile
            # up with stale entries.  Safe: only frees the slot if it
            # still points at *our* Event (a newer request may have
            # already replaced it, in which case leave that one alone).
            _release_cancel_slot(session_key, my_cancel)
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    asyncio.create_task(asyncio.to_thread(_producer))

    async def body() -> AsyncIterator[bytes]:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            yield item

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Bit-Depth": "16",
            "X-Channels": "1",
            "X-Speaker": chosen_speaker,
            "X-Session-Id": session_key,
        },
    )


# ─── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(
        "xtts-server starting on %s:%d, model dir=%s, device=%s",
        HOST_BIND, PORT, MODEL_DIR, DEVICE,
    )
    uvicorn.run(app, host=HOST_BIND, port=PORT, log_level="info")
