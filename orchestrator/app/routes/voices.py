import logging
import os
import struct
from pathlib import Path

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from .. import tts
from ..storage import (
    delete_custom_voice,
    get_custom_voice_by_id,
    get_custom_voices,
    save_custom_voice,
)

log = logging.getLogger(__name__)

router = APIRouter()


# Where the orchestrator writes user-recorded reference WAVs.  Inside
# the container this is /data/custom_voices/ (mapped from <repo>/data/
# on the host); xtts-server reads them via the equivalent host path
# (see tts.DATA_DIR_HOST / _to_host_path).
CUSTOM_VOICES_DIR = Path(
    os.environ.get("DATA_DIR_CONTAINER", "/data")
) / "custom_voices"


@router.get("/api/voices")
async def list_voices() -> JSONResponse:
    """
    Catalogue of TTS voices the user can pick from.

    Two flavours, returned in one payload so the UI doesn't need two
    round-trips:
      • ``voices``         — XTTS built-in speaker names (~58, fixed at
        model load time, cached for the orchestrator's lifetime).
      • ``custom_voices``  — user-recorded reference samples that
        xtts-server clones on the fly via ``reference_audio``.  Each is
        ``{"id": int, "name": str}``; the UI references them by the
        string ``"clone:<id>"`` in the same place built-in names live.
    """
    voices = await tts.voices()
    custom = await get_custom_voices()
    return JSONResponse(
        {
            "voices": voices,
            "custom_voices": [{"id": v[0], "name": v[1]} for v in custom],
        }
    )


@router.get("/api/custom_voices")
async def list_custom_voices() -> JSONResponse:
    """List user-recorded cloning voices: id + display name only.

    The on-disk WAV path is an internal implementation detail and is
    not exposed to the browser.
    """
    voices = await get_custom_voices()
    return JSONResponse(
        {"custom_voices": [{"id": v[0], "name": v[1]} for v in voices]}
    )


def _wrap_pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM int16 mono samples in a minimal RIFF/WAVE header.

    XTTS accepts any sample rate for the reference audio; we keep the
    16 kHz of the browser worklet rather than resampling — cleaner and
    XTTS's conditioning encoder is sample-rate agnostic.
    """
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,                # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_chunk = struct.pack("<4sI", b"data", data_size) + pcm_bytes
    riff = b"RIFF" + struct.pack("<I", 4 + len(fmt_chunk) + len(data_chunk)) + b"WAVE"
    return riff + fmt_chunk + data_chunk


@router.post("/api/custom_voices/record")
async def record_custom_voice(name: str, audio: UploadFile) -> JSONResponse:
    """
    Save a user-recorded reference WAV for on-the-fly voice cloning.

    ``audio`` body is raw PCM (signed 16-bit, mono, 16 kHz) — same
    format the speaker enrollment endpoint takes, so the frontend can
    reuse its worklet pipeline.  We wrap it in a WAV header and write
    to ``/data/custom_voices/<id>.wav``.  xtts-server reads the host-
    side equivalent of that path (see tts._to_host_path) when the user
    later selects this voice via ``"clone:<id>"``.
    """
    pcm = await audio.read()
    if len(pcm) < 16000 * 2 * 2:  # < 2 s of audio
        return JSONResponse(
            {"error": "audio_too_short (need ≥2 s, ideally 6-12 s)"},
            status_code=400,
        )
    name = name.strip()
    if not name:
        return JSONResponse({"error": "name_required"}, status_code=400)

    CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    # Two-step write: insert row first so we get the autoincrement id,
    # then write the WAV under that id.  If the disk write fails we
    # leave the row in place — UI will surface the broken voice and
    # the user can delete it.  Keeps id = filename invariant.
    placeholder_path = ""
    voice_id = await save_custom_voice(name, placeholder_path)
    wav_path = str(CUSTOM_VOICES_DIR / f"{voice_id}.wav")
    try:
        wav_bytes = _wrap_pcm16_to_wav(pcm, sample_rate=16000)
        Path(wav_path).write_bytes(wav_bytes)
    except Exception as exc:
        await delete_custom_voice(voice_id)
        log.exception("custom_voice: failed to write WAV")
        return JSONResponse(
            {"error": f"write_failed: {exc.__class__.__name__}"}, status_code=500
        )
    # Patch the row with the real path (separate UPDATE to keep the
    # CRUD module's API surface tiny — no upsert helper needed).
    from ..storage.db import _conn, _lock
    with _lock:
        c = _conn()
        try:
            c.execute(
                "UPDATE custom_voices SET wav_path=? WHERE id=?",
                (wav_path, voice_id),
            )
        finally:
            c.close()
    log.info(
        "custom_voice: saved %r as id=%d (%s, %d bytes pcm)",
        name, voice_id, wav_path, len(pcm),
    )
    return JSONResponse({"ok": True, "id": voice_id, "name": name})


@router.delete("/api/custom_voices/{voice_id}")
async def remove_custom_voice(voice_id: int) -> JSONResponse:
    """Delete a custom voice — DB row + WAV file on disk.

    Speakers that referenced ``"clone:<id>"`` will fall back to the
    server default the next time they're identified (see
    tts._resolve_voice on missing row).  We don't bulk-clear stale
    pointers — they're harmless and trivial to spot in the UI.
    """
    row = await get_custom_voice_by_id(voice_id)
    if row is not None:
        _id, _name, wav_path = row
        try:
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)
        except Exception:
            log.warning("custom_voice %d: WAV unlink failed", voice_id, exc_info=True)
    await delete_custom_voice(voice_id)
    return JSONResponse({"ok": True})
