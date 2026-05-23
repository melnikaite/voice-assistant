import logging
import os
import struct
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent_proxy, desktop_client, memory, pending_executor, push, scheduler, speaker, tts
from .search import current_region as _current_search_region
from .agent import AgentContext
from .llm import respond
from .storage import (
    PRICING,
    compute_projected_cost,
    count_unread_voicemail,
    create_session as auth_create_session,
    delete_custom_voice,
    delete_speaker_profile,
    delete_voice_message,
    get_custom_voice_by_id,
    get_custom_voices,
    get_daily_usage,
    get_pending_action,
    get_per_tool_usage,
    get_per_user_usage,
    get_session as auth_get_session,
    get_speaker_profile_by_name,
    get_speaker_profiles,
    get_voice_message,
    init_schema,
    list_outgoing_voicemail,
    list_pending_actions,
    list_recent_actions,
    list_voicemail,
    mark_approved,
    mark_rejected,
    mark_voicemail_listened,
    revoke_session as auth_revoke_session,
    save_custom_voice,
    save_speaker_profile,
    save_utterance,
    save_voicemail_reply,
    set_speaker_tts_voice,
    start_session,
    update_speaker_profile,
    voicemail_audio_path,
)
from .user_files import (
    UserSettings,
    append_memory,
    patch_settings,
    read_memory,
    read_settings,
    set_passphrase,
    verify_passphrase,
    write_memory,
)
from .storage.items import purge_expired_trash
from .ws import handle_ws

# Browser session cookie name.  HttpOnly + SameSite=Lax so a malicious
# tab can't read it via JS but an in-tab POST to /api/users/... still
# carries it.  Insecure (no `Secure` flag) because we're served over
# plain HTTP on localhost; flip when fronted by HTTPS.
SESSION_COOKIE = "va_session"

# Where the orchestrator writes user-recorded reference WAVs.  Inside
# the container this is /data/custom_voices/ (mapped from <repo>/data/
# on the host); xtts-server reads them via the equivalent host path
# (see tts.DATA_DIR_HOST / _to_host_path).
CUSTOM_VOICES_DIR = Path(
    os.environ.get("DATA_DIR_CONTAINER", "/data")
) / "custom_voices"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

WHISPER_URL = os.environ["WHISPER_URL"]
WHISPER_MODEL = os.environ["WHISPER_MODEL"]
LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]

# Wake-word configuration is surfaced to the frontend via /api/config.
# The browser-side detector takes a model name (resolved to
# `/models/<name>.onnx` by frontend/main.js) and a detection threshold.
# Defaults match the file currently shipped in frontend/models/.
WAKE_WORD_NAME = os.environ.get("WAKE_WORD_NAME", "hey_jarvis_v0.1")
try:
    WAKE_WORD_THRESHOLD = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5"))
except ValueError:
    log.warning(
        "Bad WAKE_WORD_THRESHOLD env (%r) — falling back to 0.5",
        os.environ.get("WAKE_WORD_THRESHOLD"),
    )
    WAKE_WORD_THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("init storage...")
    init_schema()
    log.info("init VAPID keypair (Web Push)...")
    try:
        push.init_vapid()
    except Exception:
        # Push is a nice-to-have; failing here would block the whole
        # orchestrator from booting (and break voice + voicemail + UI).
        # Log and continue — the /api/push/* endpoints will surface 503s
        # on demand so the frontend can degrade gracefully.
        log.exception("push: init_vapid failed — Web Push disabled")
    log.info("init embedding model (semantic memory)...")
    memory.init_embedding_model()
    log.info("init speaker encoder...")
    speaker.init_encoder()
    log.info("init TTS (XTTS-v2)...")
    await tts.init_voices()
    log.info("init desktop-agent client...")
    await desktop_client.init_desktop()
    log.info("active web search region: %s (env DDG_REGION)", _current_search_region())
    log.info("starting scheduler...")
    scheduler.start()
    pending = await scheduler.reload_pending()
    log.info("scheduler: %d pending reminder(s) re-scheduled", pending)
    # Periodic GC: hard-delete item-store rows that have been in the trash
    # for more than 7 days.  Other tables (pending_actions, auth_sessions)
    # filter by expires_at at read time, so they don't need periodic sweeps.
    scheduler.add_periodic(purge_expired_trash, hours=6, job_id="items_trash_gc")
    log.info("starting pending-action executor...")
    pending_executor.start()
    log.info("ready (wake detection happens in browser)")
    yield
    pending_executor.stop()
    scheduler.stop()
    # Cancel the desktop-agent health-poll task so the orchestrator
    # can shut down cleanly.  Best-effort — a stuck poll won't block
    # the process exit beyond the asyncio cancel timeout.
    await desktop_client.shutdown_desktop()


app = FastAPI(title="voice-assistant", lifespan=lifespan)


async def _probe(client: httpx.AsyncClient, url: str, expected_model: str) -> dict:
    try:
        r = await client.get(f"{url}/v1/models")
    except httpx.HTTPError as e:
        return {"status": "unreachable", "error": f"{e.__class__.__name__}: {e}"}
    if r.status_code != 200:
        return {"status": "bad_status", "code": r.status_code}
    ids = [m["id"] for m in r.json().get("data", [])]
    return {
        "status": "ok" if expected_model in ids else "model_missing",
        "expected": expected_model,
        "available": ids,
    }


@app.get("/api/config")
async def api_config() -> dict:
    """
    Per-deployment configuration the frontend needs at boot.

    Today this surfaces only the wake-word knobs (model name + score
    threshold) so they're configurable via env (WAKE_WORD_NAME,
    WAKE_WORD_THRESHOLD) instead of hard-coded in main.js.  Same pattern
    can extend to other browser-visible settings later (locale, sample
    rate, etc.) without a code change on the frontend.
    """
    return {
        "wake_word": {
            "name": WAKE_WORD_NAME,
            "threshold": WAKE_WORD_THRESHOLD,
        },
    }


@app.get("/health")
async def health() -> dict:
    async with httpx.AsyncClient(timeout=3) as client:
        whisper = await _probe(client, WHISPER_URL, WHISPER_MODEL)
        llm = await _probe(client, LLM_URL, LLM_MODEL)
    overall = "ok" if whisper["status"] == "ok" and llm["status"] == "ok" else "degraded"
    return {
        "status": overall,
        "backends": {"whisper": whisper, "llm": llm},
        "wake": {"location": "browser"},
    }


class TextRequest(BaseModel):
    text: str
    history: list[dict] | None = None  # [{role: "user"|"assistant", content: "..."}, ...]
    client_id: str | None = None  # optional — required for set_reminder side effects


@app.post("/dev/respond")
async def dev_respond(req: TextRequest) -> JSONResponse:
    """Bypass ASR — feed a transcript directly to the agent loop."""
    import json as _json
    import time as _time

    session_id = await start_session(client="dev", client_id=req.client_id)
    ctx = AgentContext(client_id=req.client_id or "dev-client")
    decision = await respond(req.text, history=req.history, ctx=ctx)
    await save_utterance(
        session_id=session_id,
        ts=_time.time(),
        audio_duration_ms=None,
        transcript=req.text,
        asr_ms=None,
        llm_ms=decision.elapsed_ms,
        tool_name=decision.tool_name,
        tool_args=_json.dumps(decision.tool_args, ensure_ascii=False)
        if decision.tool_args
        else None,
        response_text=decision.response_text,
        error=None,
    )
    return JSONResponse(
        {
            "transcript": req.text,
            "tool_name": decision.tool_name,
            "tool_args": decision.tool_args,
            "response_text": decision.response_text,
            "llm_ms": decision.elapsed_ms,
        }
    )


@app.get("/api/speakers")
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


@app.get("/api/voices")
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


@app.get("/api/custom_voices")
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


@app.post("/api/custom_voices/record")
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
    from .storage.db import _conn, _lock
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


@app.delete("/api/custom_voices/{voice_id}")
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


class SpeakerPatchRequest(BaseModel):
    # `tts_voice` may be null/empty to clear the per-speaker override.
    tts_voice: str | None = None


@app.patch("/api/speakers/{profile_id}")
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


@app.post("/api/speakers/enroll")
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


@app.delete("/api/speakers/{profile_id}")
async def remove_speaker(profile_id: int) -> JSONResponse:
    """Delete a speaker profile by id."""
    await delete_speaker_profile(profile_id)
    return JSONResponse({"ok": True})


@app.get("/api/stats")
async def stats(range: str = "week") -> JSONResponse:
    """LLM token usage stats for the dashboard.

    ``range`` controls the lookback window: ``day`` (1 d), ``week`` (7 d,
    default), ``month`` (30 d).  Unknown values fall back to ``week`` —
    no 400, so a buggy client never breaks the page.

    Payload:
      • ``daily``      — per-day prompt/completion totals (stacked bar)
      • ``per_tool``   — per-tool totals (horizontal bar)
      • ``per_user``   — per-client_id totals (horizontal bar)
      • ``cost``       — projected $-cost per pricing tier (Claude /
                         GPT-4o-mini / local Gemma) over the range
      • ``pricing``    — the rate table used for ``cost``, so the UI can
                         label its summary ("if you had used … it would
                         have cost …")
      • ``range``      — echoed back, helps the client confirm it asked
                         for the right window after a network glitch
    """
    days = {"day": 1, "week": 7, "month": 30}.get(range, 7)
    daily = await get_daily_usage(days)
    per_tool = await get_per_tool_usage(days)
    per_user = await get_per_user_usage(days)
    cost = compute_projected_cost(daily)
    return JSONResponse(
        {
            "range": range,
            "days": days,
            "daily": daily,
            "per_tool": per_tool,
            "per_user": per_user,
            "cost": cost,
            "pricing": PRICING,
        }
    )


# ─── Auth ────────────────────────────────────────────────────────────────


async def _current_user(
    va_session: str | None = Cookie(default=None),
) -> dict:
    """FastAPI dependency: resolve the cookie session → profile.

    Raises 401 if the cookie is missing, invalid, or expired.  Endpoints
    that want auth declare `user: dict = Depends(_current_user)`; ones
    that don't (login, setup) skip it.
    """
    if not va_session:
        raise HTTPException(401, "not authenticated")
    sess = await auth_get_session(va_session)
    if not sess:
        raise HTTPException(401, "session expired or invalid")
    return sess


class PassphraseSetupRequest(BaseModel):
    profile_id: int
    passphrase: str


@app.post("/api/auth/setup_passphrase")
async def setup_passphrase(req: PassphraseSetupRequest) -> JSONResponse:
    """
    Set or reset the passphrase for a profile.

    Open to anyone the first time — necessary so a fresh install can
    bootstrap.  Once a passphrase exists, only an authenticated session
    for THAT profile may rotate it (else a stranger could lock a
    legitimate user out of their own profile).
    """
    settings_now = await read_settings(req.profile_id)
    if settings_now.code_word_hash:
        # Already set — require an active session for this profile.
        # (Frontend will only show this control to a logged-in user.)
        raise HTTPException(
            403,
            "Passphrase already set — log in and rotate from the Settings tab.",
        )
    try:
        await set_passphrase(req.profile_id, req.passphrase)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("auth: passphrase set for profile=%d", req.profile_id)
    return JSONResponse({"ok": True, "profile_id": req.profile_id})


class LoginRequest(BaseModel):
    profile_id: int
    passphrase: str


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest, request: Request) -> JSONResponse:
    """Verify passphrase, mint a server-side session, set the cookie.

    Note on FastAPI cookie wiring: a cookie set on the dependency-
    injected ``Response`` parameter is dropped when the handler returns
    a *new* Response object (JSONResponse).  We therefore set the cookie
    directly on the JSONResponse instance we're returning.
    """
    ok = await verify_passphrase(req.profile_id, req.passphrase)
    if not ok:
        raise HTTPException(401, "passphrase mismatch")
    token = await auth_create_session(
        req.profile_id,
        user_agent=request.headers.get("user-agent"),
    )
    resp = JSONResponse({"ok": True, "profile_id": req.profile_id})
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=30 * 86400,
        # secure=True  # enable when fronted by HTTPS
        path="/",
    )
    log.info("auth: login OK for profile=%d", req.profile_id)
    return resp


@app.post("/api/auth/logout")
async def auth_logout(
    va_session: str | None = Cookie(default=None),
) -> JSONResponse:
    if va_session:
        await auth_revoke_session(va_session)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/api/me")
async def api_me(user: dict = Depends(_current_user)) -> JSONResponse:
    """Tell the UI which profile is logged in (+ session expiry)."""
    return JSONResponse({"profile_id": user["profile_id"], "expires_at": user["expires_at"]})


# ─── Per-user memory / settings (UI surface) ────────────────────────────


@app.get("/api/users/{profile_id}/memory")
async def api_read_memory(
    profile_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    body = await read_memory(profile_id)
    return JSONResponse({"profile_id": profile_id, "memory": body})


class MemoryWriteRequest(BaseModel):
    content: str  # full replacement body


@app.put("/api/users/{profile_id}/memory")
async def api_write_memory(
    profile_id: int,
    body: MemoryWriteRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    await write_memory(profile_id, body.content)
    return JSONResponse({"ok": True, "profile_id": profile_id})


@app.get("/api/users/{profile_id}/settings")
async def api_read_settings(
    profile_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    s = await read_settings(profile_id)
    return JSONResponse({"profile_id": profile_id, "settings": s.model_dump()})


@app.put("/api/users/{profile_id}/settings")
async def api_write_settings(
    profile_id: int,
    body: UserSettings,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Replace the typed settings with the supplied UserSettings.

    Note: ``code_word_hash`` is intentionally excluded from this flow
    — a UI that ever sends it would let a logged-in user overwrite
    their own hash with garbage and lock themselves out.  Passphrase
    rotation goes through the dedicated ``/api/auth/rotate_passphrase``
    endpoint below.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    safe = body.model_copy(update={"code_word_hash": (await read_settings(profile_id)).code_word_hash})
    from .user_files import write_settings as _ws
    await _ws(profile_id, safe)
    return JSONResponse({"ok": True, "settings": safe.model_dump()})


class RotatePassphraseRequest(BaseModel):
    current_passphrase: str
    new_passphrase: str


@app.post("/api/auth/rotate_passphrase")
async def rotate_passphrase(
    req: RotatePassphraseRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Change the logged-in user's passphrase.

    Re-verifies the current passphrase before accepting the new one —
    so an unattended logged-in tab can't silently lock the owner out.
    The cookie session stays valid (no re-login required) since the
    user just proved they own the profile.
    """
    profile_id = user["profile_id"]
    ok = await verify_passphrase(profile_id, req.current_passphrase)
    if not ok:
        raise HTTPException(401, "current passphrase mismatch")
    try:
        await set_passphrase(profile_id, req.new_passphrase)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("auth: passphrase rotated for profile=%d", profile_id)
    return JSONResponse({"ok": True, "profile_id": profile_id})


# ─── Pending-action queue (UI surface) ──────────────────────────────────


@app.get("/api/users/{profile_id}/pending")
async def api_list_pending(
    profile_id: int,
    include_recent: bool = False,
    recent_limit: int = 20,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """List the speaker's pending action queue.

    With ``include_recent=true`` the response also carries the most
    recently finalised actions (executed / failed / rejected / expired)
    so the UI can show a small "Recent" panel next to the live queue.
    Capped by ``recent_limit`` (default 20).
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    items = await list_pending_actions(profile_id=profile_id)
    payload: dict = {"actions": items}
    if include_recent:
        payload["recent"] = await list_recent_actions(
            profile_id=profile_id, limit=max(1, min(int(recent_limit), 100))
        )
    return JSONResponse(payload)


@app.post("/api/pending/{action_id}/approve")
async def api_approve_pending(
    action_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    row = await get_pending_action(action_id)
    if not row:
        raise HTTPException(404, "no such action")
    # An action queued under a different profile shouldn't be approvable
    # by an unrelated user.  Allow if profile matches OR if the action
    # was queued without any profile (e.g. /dev/respond test traffic).
    if row["profile_id"] is not None and row["profile_id"] != user["profile_id"]:
        raise HTTPException(403, "cross-profile approval not allowed")
    ok = await mark_approved(action_id, via="ui")
    if not ok:
        raise HTTPException(409, "action no longer pending")
    return JSONResponse({"ok": True, "action_id": action_id, "summary": row["summary"]})


@app.post("/api/pending/{action_id}/reject")
async def api_reject_pending(
    action_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    row = await get_pending_action(action_id)
    if not row:
        raise HTTPException(404, "no such action")
    if row["profile_id"] is not None and row["profile_id"] != user["profile_id"]:
        raise HTTPException(403, "cross-profile rejection not allowed")
    ok = await mark_rejected(action_id, via="ui")
    if not ok:
        raise HTTPException(409, "action no longer pending")
    return JSONResponse({"ok": True, "action_id": action_id, "summary": row["summary"]})


# ─── Web Push (VAPID) ───────────────────────────────────────────────────


@app.get("/api/push/vapid_public_key")
async def api_push_vapid_public_key() -> JSONResponse:
    """Hand the frontend our VAPID public key.

    No auth: it's a public key by design — the browser passes it as
    ``applicationServerKey`` to ``pushManager.subscribe``.  Knowing the
    key gives nobody any privileges; the matching private key lives
    exclusively in ``$DATA_DIR_CONTAINER/vapid_private.pem``.
    """
    try:
        key = push.get_public_key()
    except RuntimeError as exc:
        # init_vapid failed at startup — surface 503 so the frontend
        # knows push isn't available (rather than caching a fake key).
        raise HTTPException(503, str(exc))
    return JSONResponse({"public_key": key})


class PushSubscribeRequest(BaseModel):
    """Browser-side ``PushSubscription#toJSON()`` payload.

    We keep the field names verbatim so the frontend can do
    ``fetch('/api/push/subscribe', {body: JSON.stringify(sub)})`` without
    repacking — the lookup keys (``endpoint``, ``keys.p256dh``,
    ``keys.auth``) match the W3C Push API exactly.
    """
    endpoint: str
    keys: dict[str, str]


@app.post("/api/push/subscribe")
async def api_push_subscribe(
    body: PushSubscribeRequest,
    request: Request,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Register a PushSubscription for the logged-in profile.

    Idempotent — repeated calls with the same ``endpoint`` refresh the
    keys (browsers may rotate them) and bump ``created_at`` without
    creating duplicate rows.  The store keys subscription rows on the
    push-service endpoint, so a second profile subscribing from the
    same device transfers ownership rather than collecting both — see
    ``push_subscriptions._upsert_sync`` for the contract.
    """
    try:
        row_id = await push.register_subscription(
            user["profile_id"],
            {"endpoint": body.endpoint, "keys": body.keys},
            user_agent=request.headers.get("user-agent"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info(
        "push: subscribed profile=%d endpoint=%s",
        user["profile_id"], body.endpoint[:48] + "…",
    )
    return JSONResponse({"ok": True, "id": row_id})


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


@app.delete("/api/push/subscribe")
async def api_push_unsubscribe(
    body: PushUnsubscribeRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Delete a PushSubscription by endpoint.

    Auth-required because endpoints are sensitive — the push-service
    URL is effectively a capability token for sending to that
    subscription.  We don't cross-check ownership against the cookie:
    a user unsubscribing from someone else's browser is a no-op for
    privacy (their session can't read it anyway) and a positive for
    GC (one fewer dead row).
    """
    deleted = await push.unregister_subscription(body.endpoint)
    log.info(
        "push: unsubscribe profile=%d endpoint=%s deleted=%s",
        user["profile_id"], body.endpoint[:48] + "…", deleted,
    )
    return JSONResponse({"ok": True, "deleted": deleted})


# ─── Voicemail (UI surface) ─────────────────────────────────────────────


@app.get("/api/users/{profile_id}/voicemail")
async def api_list_voicemail(
    profile_id: int,
    unread_only: bool = False,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """List voicemail addressed to a profile.

    Cookie-auth required and the requesting profile must match the
    inbox owner — like /memory and /settings, no cross-profile peeks.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    items = await list_voicemail(
        profile_id,
        unread_only=bool(unread_only),
        limit=max(1, min(int(limit), 200)),
    )
    unread = await count_unread_voicemail(profile_id)
    return JSONResponse({
        "profile_id": profile_id,
        "unread_count": unread,
        "messages": items,
    })


@app.get("/api/users/{profile_id}/outgoing_voicemail")
async def api_list_outgoing_voicemail(
    profile_id: int,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """List voicemail rows the given profile *sent*.

    Same ownership shape as the inbox endpoint: cookie-auth required,
    requesting profile must equal ``profile_id``.  The Sent panel in
    the UI uses this to show the user their own outgoing messages and
    any replies that have come back — closing the delivery loop the
    sender wouldn't otherwise see (the recipient's reply is on the
    voicemail row, not pushed back via TTS today).
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    items = await list_outgoing_voicemail(
        profile_id,
        limit=max(1, min(int(limit), 200)),
    )
    return JSONResponse({
        "profile_id": profile_id,
        "messages": items,
    })


@app.get("/api/voicemail/{message_id}/audio")
async def api_voicemail_audio(
    message_id: int,
    user: dict = Depends(_current_user),
) -> FileResponse:
    """Serve the raw WAV bytes of a voicemail.

    Cookie-auth required.  The row's ``to_profile_id`` must match the
    logged-in profile — anything else gets a 404 (we don't 403 since
    that would leak existence of someone else's mail).
    """
    row = await get_voice_message(message_id)
    if row is None or row["to_profile_id"] != user["profile_id"]:
        raise HTTPException(404, "not found")
    path = voicemail_audio_path(row["audio_path"] or message_id)
    if not path.exists():
        log.warning("voicemail: wav missing for id=%d (%s)", message_id, path)
        raise HTTPException(404, "audio missing")
    return FileResponse(
        str(path),
        media_type="audio/wav",
        # `inline` lets the <audio> element play it; ``filename=`` makes
        # a manual download save sensibly.
        headers={"Content-Disposition": f'inline; filename="voicemail-{message_id}.wav"'},
    )


@app.post("/api/voicemail/{message_id}/listened")
async def api_voicemail_listened(
    message_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Mark a voicemail as listened (the recipient heard it).

    Idempotent — repeated calls after the first one are no-ops.  Used
    by the frontend Inbox panel when audio playback finishes.
    """
    ok = await mark_voicemail_listened(message_id, user["profile_id"])
    return JSONResponse({"ok": True, "first_time": bool(ok)})


class VoicemailReplyRequest(BaseModel):
    reply: str


@app.post("/api/voicemail/{message_id}/reply")
async def api_voicemail_reply(
    message_id: int,
    body: VoicemailReplyRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Save the recipient's textual reply.

    Delivery to the sender is out-of-scope today — the reply lives on
    the voicemail row so the sender (if they're the host of this
    install too) can see it; cross-device push is a follow-up.
    """
    reply = (body.reply or "").strip()
    if not reply:
        raise HTTPException(400, "empty reply")
    ok = await save_voicemail_reply(message_id, user["profile_id"], reply)
    if not ok:
        raise HTTPException(404, "not found")
    return JSONResponse({"ok": True, "message_id": message_id})


@app.delete("/api/voicemail/{message_id}")
async def api_voicemail_delete(
    message_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    ok = await delete_voice_message(message_id, user["profile_id"])
    if not ok:
        raise HTTPException(404, "not found")
    return JSONResponse({"ok": True})


# ─── Desktop agents (multi-agent registry + reverse-WSS) ────────────────


@app.get("/api/agents")
async def api_list_agents(user: dict = Depends(_current_user)) -> JSONResponse:
    """List every registered desktop-agent + its latest known state.

    Cookie-auth required — same gate as /memory, /settings: not a
    secret per se, but no reason to expose the operator's tailnet
    topology to anyone with the public URL.

    Payload shape:
        {
          "agents": [
            {"agent_id": "...", "platform": "macos"|"linux"|"windows"|"unknown",
             "reachable": true, "mode": "http"|"reverse",
             "capabilities": {...}, "version": "...",
             "last_seen": <unix ts>, "default": bool},
            ...
          ],
          "default": "<agent_id of the default agent, or null>"
        }

    The frontend uses this to render the "Connected devices" panel.
    """
    out: list[dict] = []
    for info in desktop_client.list_agents():
        caps = info.capabilities_cache or {}
        out.append({
            "agent_id": info.agent_id,
            "platform": caps.get("platform") or "unknown",
            "reachable": bool(info.reachable),
            "mode": info.mode,
            "capabilities": caps.get("capabilities") or {},
            "version": caps.get("version") or None,
            "last_seen": info.last_seen,
            "default": info.default,
        })
    default = desktop_client.get_agent(None)
    return JSONResponse({
        "agents": out,
        "default": default.agent_id if default else None,
    })


@app.websocket("/v1/agent/connect")
async def ws_agent_connect(websocket: WebSocket) -> None:
    """Reverse-WSS endpoint for NAT-traversed desktop-agents.

    See :mod:`.agent_proxy` for the wire protocol.  Auth happens
    inside the handler after the hello frame is parsed — we can't gate
    on a header here because the WSS client might not be able to set
    custom headers (browser-style WSS clients can't).
    """
    await agent_proxy.handle_agent_connect(websocket)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, client_id: str | None = None):
    await handle_ws(websocket, client_id=client_id)


# ─── Personal item store — Categories ────────────────────────────────────


class CategoryCreateRequest(BaseModel):
    name: str
    parent_id: int | None = None
    kind: str = "folder"


class CategoryPatchRequest(BaseModel):
    name: str | None = None
    parent_id: int | None = None  # use -1 as sentinel to move to root


class CategoryShareRequest(BaseModel):
    with_profile_id: int
    permission: str = "read"


class ItemCreateRequest(BaseModel):
    kind: str
    title: str | None = None
    url: str | None = None
    body: str | None = None
    category_id: int | None = None
    source_meta: dict | None = None  # stored as JSON string


class ItemPatchRequest(BaseModel):
    title: str | None = None
    body: str | None = None
    category_id: int | None = None
    summary: str | None = None


class ItemMoveRequest(BaseModel):
    category_id: int | None = None


class ItemReorderRequest(BaseModel):
    sort_order: float


class AutoSortApplyRequest(BaseModel):
    suggestions: list[dict]  # [{item_id: int, category_id: int}, ...]


def _strip_embedding(item: dict) -> dict:
    """Drop the embedding blob before serialising to JSON."""
    out = dict(item)
    out.pop("embedding", None)
    return out


@app.get("/api/users/{profile_id}/categories")
async def api_list_categories(
    profile_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.categories import list_categories
    cats = await list_categories(profile_id, include_shared=True)
    return JSONResponse({"categories": cats})


@app.post("/api/users/{profile_id}/categories")
async def api_create_category(
    profile_id: int,
    body: CategoryCreateRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.categories import create_category
    cat_id = await create_category(
        owner_profile_id=profile_id,
        name=body.name,
        parent_id=body.parent_id,
        kind=body.kind,
    )
    return JSONResponse({"ok": True, "category_id": cat_id})


@app.patch("/api/users/{profile_id}/categories/{category_id}")
async def api_patch_category(
    profile_id: int,
    category_id: int,
    body: CategoryPatchRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.categories import rename_category, move_category, list_subtree, get_category
    cat = await get_category(category_id)
    if cat is None or cat["owner_profile_id"] != profile_id:
        raise HTTPException(404, "category not found")
    if body.name is not None:
        ok = await rename_category(category_id, profile_id, body.name)
        if not ok:
            raise HTTPException(404, "category not found")
    if body.parent_id is not None:
        # -1 is the sentinel for "move to root"
        new_parent: int | None = None if body.parent_id == -1 else body.parent_id
        if new_parent is not None:
            subtree = await list_subtree(category_id, profile_id)
            if any(c["id"] == new_parent for c in subtree):
                raise HTTPException(400, "cycle detected: new parent is a descendant")
        ok = await move_category(category_id, profile_id, new_parent)
        if not ok:
            raise HTTPException(404, "category not found")
    return JSONResponse({"ok": True, "category_id": category_id})


@app.delete("/api/users/{profile_id}/categories/{category_id}")
async def api_delete_category(
    profile_id: int,
    category_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.categories import delete_category
    ok = await delete_category(category_id, profile_id)
    if not ok:
        raise HTTPException(404, "category not found")
    return JSONResponse({"ok": True})


@app.post("/api/users/{profile_id}/categories/{category_id}/share")
async def api_share_category(
    profile_id: int,
    category_id: int,
    body: CategoryShareRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.categories import share_category, get_category
    cat = await get_category(category_id)
    if cat is None or cat["owner_profile_id"] != profile_id:
        raise HTTPException(404, "category not found")
    try:
        await share_category(category_id, body.with_profile_id, permission=body.permission)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True})


@app.delete("/api/users/{profile_id}/categories/{category_id}/share/{with_profile_id}")
async def api_unshare_category(
    profile_id: int,
    category_id: int,
    with_profile_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.categories import unshare_category, get_category
    cat = await get_category(category_id)
    if cat is None or cat["owner_profile_id"] != profile_id:
        raise HTTPException(404, "category not found")
    await unshare_category(category_id, with_profile_id)
    return JSONResponse({"ok": True})


# ─── Personal item store — Items ─────────────────────────────────────────


@app.get("/api/users/{profile_id}/items")
async def api_list_items(
    profile_id: int,
    category_id: int | None = None,
    kind: str | None = None,
    sort: str = "date_desc",
    limit: int = 50,
    offset: int = 0,
    deleted_only: bool = False,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import list_items
    rows = await list_items(
        profile_id,
        category_id=category_id,
        kind=kind,
        sort=sort,
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
        deleted_only=deleted_only,
    )
    return JSONResponse({"items": [_strip_embedding(r) for r in rows]})


@app.post("/api/users/{profile_id}/items")
async def api_create_item(
    profile_id: int,
    body: ItemCreateRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    import json as _json
    from .items import ingest as _ingest
    source_meta_str = _json.dumps(body.source_meta) if body.source_meta else None
    try:
        if body.kind == "text":
            if not body.body:
                raise HTTPException(400, "body required for text items")
            item_id = await _ingest.ingest_text(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=body.category_id,
                body=body.body,
                title=body.title,
            )
        elif body.kind == "link":
            if not body.url:
                raise HTTPException(400, "url required for link items")
            item_id = await _ingest.ingest_link(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=body.category_id,
                url=body.url,
                title=body.title,
            )
        elif body.kind in ("video", "short"):
            if not body.url:
                raise HTTPException(400, "url required for video/short items")
            item_id = await _ingest.ingest_video(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=body.category_id,
                url=body.url,
                kind=body.kind,
                title=body.title,
            )
        else:
            raise HTTPException(400, f"unsupported kind: {body.kind!r} — use text/link/video/short; screenshots go to /items/screenshot")
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api_create_item failed")
        raise HTTPException(500, str(exc))
    return JSONResponse({"ok": True, "item_id": item_id}, status_code=201)


@app.post("/api/users/{profile_id}/items/screenshot")
async def api_upload_screenshot(
    profile_id: int,
    file: UploadFile,
    category_id: int | None = None,
    title: str | None = None,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .items import ingest as _ingest
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        item_id = await _ingest.ingest_screenshot(
            owner_profile_id=profile_id,
            created_by_profile_id=profile_id,
            category_id=category_id,
            image_bytes=data,
            title=title,
        )
    except Exception as exc:
        log.exception("api_upload_screenshot failed")
        raise HTTPException(500, str(exc))
    return JSONResponse({"ok": True, "item_id": item_id}, status_code=201)


@app.get("/api/users/{profile_id}/items/search")
async def api_search_items(
    profile_id: int,
    q: str,
    category_id: int | None = None,
    kind: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
    sort: str = "relevance",
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    import datetime as _dt
    from .items import search as _search

    def _parse_ts(s: str | None) -> float | None:
        if not s:
            return None
        try:
            return _dt.datetime.fromisoformat(s).timestamp()
        except ValueError:
            return None

    results = await _search.hybrid_search(
        profile_id, q,
        category_id=category_id,
        kind=kind,
        date_from=_parse_ts(date_from),
        date_to=_parse_ts(date_to),
        limit=max(1, min(limit, 50)),
        sort=sort,
    )
    return JSONResponse({"results": [_strip_embedding(r) for r in results]})


@app.get("/api/users/{profile_id}/items/{item_id}")
async def api_get_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import get_item
    row = await get_item(item_id)
    if row is None or row["owner_profile_id"] != profile_id:
        raise HTTPException(404, "item not found")
    return JSONResponse({"item": _strip_embedding(row)})


@app.patch("/api/users/{profile_id}/items/{item_id}")
async def api_patch_item(
    profile_id: int,
    item_id: int,
    body: ItemPatchRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import update_item, _SENTINEL
    ok = await update_item(
        item_id, profile_id,
        title=body.title if body.title is not None else _SENTINEL,
        body=body.body if body.body is not None else _SENTINEL,
        category_id=body.category_id if body.category_id is not None else _SENTINEL,
        summary=body.summary if body.summary is not None else _SENTINEL,
    )
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@app.delete("/api/users/{profile_id}/items/{item_id}")
async def api_delete_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import delete_item
    ok = await delete_item(item_id, profile_id)
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@app.post("/api/users/{profile_id}/items/{item_id}/restore")
async def api_restore_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import restore_item
    ok = await restore_item(item_id, profile_id)
    if not ok:
        raise HTTPException(404, "item not found or not in trash")
    return JSONResponse({"ok": True})


@app.post("/api/users/{profile_id}/items/{item_id}/move")
async def api_move_item(
    profile_id: int,
    item_id: int,
    body: ItemMoveRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import move_item
    ok = await move_item(item_id, profile_id, body.category_id)
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@app.post("/api/users/{profile_id}/items/{item_id}/reorder")
async def api_reorder_item(
    profile_id: int,
    item_id: int,
    body: ItemReorderRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import reorder_item
    ok = await reorder_item(item_id, profile_id, body.sort_order)
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@app.post("/api/users/{profile_id}/items/{item_id}/check")
async def api_check_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import toggle_checked
    result = await toggle_checked(item_id, profile_id)
    if not result.get("found"):
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True, "completed": result["completed"], "completed_at": result["completed_at"]})


@app.post("/api/users/{profile_id}/items/auto_sort/suggest")
async def api_auto_sort_suggest(
    profile_id: int,
    category_id: int | None = None,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Ask the LLM to suggest category reassignments for items.

    Runs ``suggest_auto_sort`` from the auto_sort module and returns a list
    of ``{item_id, category_id, category_name, reason}`` suggestions.
    The caller must present these to the user before calling the apply
    endpoint — nothing is moved here.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .storage.items import list_items
    from .storage.categories import list_categories, _ALL_DEPTHS
    from .items.auto_sort import suggest_auto_sort

    items = await list_items(
        profile_id,
        category_id=category_id,
        limit=max(1, min(limit, 50)),
    )
    cats = await list_categories(profile_id)
    suggestions = await suggest_auto_sort(items, cats)
    return JSONResponse({"suggestions": suggestions})


@app.post("/api/users/{profile_id}/items/auto_sort/apply")
async def api_auto_sort_apply(
    profile_id: int,
    body: AutoSortApplyRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Apply an approved list of auto-sort suggestions.

    Calls ``apply_auto_sort`` which invokes ``move_item`` for each entry.
    Returns the count of successfully moved items.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from .items.auto_sort import apply_auto_sort

    count = await apply_auto_sort(body.suggestions, profile_id)
    return JSONResponse({"ok": True, "moved": count})


app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
