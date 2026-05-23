import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent_proxy, desktop_client, memory, pending_executor, push, scheduler, speaker, tts
from .search import current_region as _current_search_region
from .agent import AgentContext
from .llm import respond
from .routes import (
    agents as agents_routes,
    auth as auth_routes,
    items as items_routes,
    memory as memory_routes,
    pending as pending_routes,
    push as push_routes,
    speakers as speakers_routes,
    voicemail as voicemail_routes,
    voices as voices_routes,
)
from .storage import (
    PRICING,
    compute_projected_cost,
    get_daily_usage,
    get_per_tool_usage,
    get_per_user_usage,
    init_schema,
    save_utterance,
    start_session,
)
from .storage.items import purge_expired_trash
from .ws import handle_ws

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
app.include_router(auth_routes.router)
app.include_router(speakers_routes.router)
app.include_router(voices_routes.router)
app.include_router(memory_routes.router)
app.include_router(pending_routes.router)
app.include_router(push_routes.router)
app.include_router(voicemail_routes.router)
app.include_router(agents_routes.router)
app.include_router(items_routes.router)


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


app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
