import logging

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import speaker
from ..storage import (
    delete_speaker_profile,
    get_speaker_profile_by_name,
    get_speaker_profiles,
    save_speaker_profile,
    set_speaker_tts_voice,
    update_speaker_profile,
)

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/speakers")
async def list_speakers(client_id: str) -> JSONResponse:
    """List enrolled speaker profiles for a client."""
    profiles = await get_speaker_profiles(client_id)
    # Row: (id, name, embedding_bytes, sample_count, tts_voice).  We
    # surface sample_count as "samples" for the UI, and tts_voice as
    # "voice" — NULL means "use server default" and the UI shows that
    # as the blank/auto option in the dropdown.
    return JSONResponse(
        {
            "speakers": [
                {"id": p[0], "name": p[1], "samples": p[3], "voice": p[4]}
                for p in profiles
            ]
        }
    )


class SpeakerPatchRequest(BaseModel):
    # `tts_voice` may be null/empty to clear the per-speaker override.
    tts_voice: str | None = None


@router.patch("/api/speakers/{profile_id}")
async def patch_speaker(profile_id: int, body: SpeakerPatchRequest) -> JSONResponse:
    """
    Update a speaker profile.  Today only the per-speaker `tts_voice`
    is mutable; other fields (name, embedding) are write-once via the
    enrollment flow to keep the d-vector centroid consistent.
    """
    voice = body.tts_voice if body.tts_voice else None
    await set_speaker_tts_voice(profile_id, voice)
    log.info("speaker %d: tts_voice=%r", profile_id, voice)
    return JSONResponse({"ok": True, "id": profile_id, "tts_voice": voice})


@router.post("/api/speakers/enroll")
async def enroll_speaker(
    client_id: str,
    name: str,
    audio: UploadFile,
) -> JSONResponse:
    """
    Enroll (or update) a speaker profile via running-mean averaging.

    If a profile for (client_id, name) already exists the new embedding is
    blended in: ``merged = (old * n + new) / (n + 1)``, then re-normalised.
    This way enrolling multiple samples under the same name converges to a
    stable centroid rather than creating duplicate profiles.

    ``audio`` must be raw PCM: signed 16-bit, mono, 16 kHz.
    """
    import numpy as np

    if not speaker.SPEAKER_ENABLED:
        return JSONResponse({"error": "speaker_encoder_unavailable"}, status_code=503)
    audio_bytes = await audio.read()
    emb = await speaker.encode_audio(audio_bytes)
    if emb is None:
        return JSONResponse(
            {"error": "audio_too_short_or_encode_failed (need ≥1 s of speech)"},
            status_code=400,
        )

    existing = await get_speaker_profile_by_name(client_id, name)
    if existing:
        profile_id, old_bytes, n = existing
        old_emb = np.frombuffer(old_bytes, dtype=np.float32)
        # Running mean: weighted average of old centroid and new sample.
        merged = (old_emb * n + emb) / (n + 1)
        norm = float(np.linalg.norm(merged))
        if norm > 1e-9:
            merged = merged / norm  # keep as unit d-vector
        merged = merged.astype(np.float32)
        await update_speaker_profile(profile_id, merged.tobytes(), n + 1)
        log.info(
            "updated speaker %r (id=%d, samples=%d) for client %.8s…",
            name, profile_id, n + 1, client_id,
        )
        return JSONResponse({"ok": True, "id": profile_id, "name": name, "samples": n + 1})
    else:
        profile_id = await save_speaker_profile(client_id, name, emb.tobytes())
        log.info(
            "enrolled speaker %r (id=%d) for client %.8s…",
            name, profile_id, client_id,
        )
        return JSONResponse({"ok": True, "id": profile_id, "name": name, "samples": 1})


@router.delete("/api/speakers/{profile_id}")
async def remove_speaker(profile_id: int) -> JSONResponse:
    """Delete a speaker profile by id."""
    await delete_speaker_profile(profile_id)
    return JSONResponse({"ok": True})
