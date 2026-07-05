import base64
import logging
import os
import time as _time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from . import agent_proxy, desktop_client, instance_settings, memory, pending_executor, push, registry, scheduler, speaker, tts
from .search import current_region as _current_search_region
from .agent import AgentContext
from .llm import respond
from .routes import (
    agents as agents_routes,
    auth as auth_routes,
    devices as devices_routes,
    hotkey as hotkey_routes,
    instance as instance_routes,
    items as items_routes,
    memory as memory_routes,
    pending as pending_routes,
    push as push_routes,
    speakers as speakers_routes,
    step_up as step_up_routes,
    stream as stream_routes,
    voicemail as voicemail_routes,
    voices as voices_routes,
)
from .storage import (
    PRICING,
    compute_projected_cost,
    get_daily_usage,
    get_per_tool_usage,
    get_per_user_usage,
    get_tool_perf,
    get_voice_turns_today,
    init_schema,
    save_utterance,
    start_session,
)
from .routes._deps import _current_user
from .storage.items import purge_expired_trash
from .ws import handle_ws

# Process start time for uptime reporting in /api/stats.
_START_TIME: float = _time.time()

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


# ── Basic Auth middleware (optional outer layer, #43) ─────────────────────
#
# When instance_settings has basic_auth_user + basic_auth_password_hash,
# all HTTP routes except /health and WS upgrades require HTTP Basic Auth.
# Goal: anti-scanner / anti-DDoS protection so random internet crawlers
# can't trigger expensive LLM / ASR calls just by hitting the public URL.
# This is NOT a substitute for profile auth — it's the "door before the
# house".  Browser WebSocket upgrades are excluded because the browser
# can't send Basic Auth credentials on a WS upgrade; those are guarded by
# the va_session cookie and (optionally) the allow_guest_voice flag.

_BASIC_AUTH_SKIP = frozenset({"/health", "/ws", "/v1/agent/connect"})


class _BasicAuthMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces HTTP Basic Auth when configured.

    Skips paths in ``_BASIC_AUTH_SKIP`` and any path starting with ``/ws``
    (WebSocket upgrades).  All other paths get a 401 challenge on missing
    or wrong credentials.
    """

    def __init__(self, app, username: str, password_hash: str) -> None:
        super().__init__(app)
        self._username = username
        self._password_hash = password_hash

    async def dispatch(
        self, request: StarletteRequest, call_next
    ) -> StarletteResponse:
        path = request.url.path
        # Exclude health probe + WebSocket upgrade paths.
        if path in _BASIC_AUTH_SKIP or path.startswith("/ws"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.lower().startswith("basic "):
            return Response(
                "Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="voice-assistant"'},
            )
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            sent_user, _, sent_pass = decoded.partition(":")
        except Exception:
            return Response("Unauthorized", status_code=401)

        # Constant-time username compare + always-run bcrypt so the
        # response timing doesn't reveal whether the username was correct
        # (no short-circuit before the hash check → no username oracle).
        import hmac
        import bcrypt  # local import keeps bcrypt optional
        user_ok = hmac.compare_digest(sent_user, self._username)
        try:
            pass_ok = bcrypt.checkpw(
                sent_pass.encode("utf-8"),
                self._password_hash.encode("ascii"),
            )
        except Exception:
            pass_ok = False
        if not (user_ok and pass_ok):
            return Response("Unauthorized", status_code=401)

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("init storage...")
    init_schema()
    # NOTE: Basic Auth middleware is installed at app-CONSTRUCTION time
    # (see below ``app = FastAPI(...)``), NOT here.  Starlette builds its
    # middleware stack on the first request, which happens AFTER lifespan
    # startup — so ``add_middleware`` from inside lifespan either raises
    # "Cannot add middleware after an application has started" or silently
    # no-ops, leaving the door wide open.  Reading the settings file
    # synchronously at import is fine: Basic Auth already requires a
    # restart to take effect.
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

# ── Outer Basic Auth (#43) — installed at construction, NOT in lifespan ──
# Starlette builds its middleware stack on the first request (after lifespan
# startup), so add_middleware() must run here, before the app starts serving.
# We read instance settings synchronously at import; enabling/disabling Basic
# Auth already requires a restart, so a one-time read at boot is correct.
_instance_cfg = instance_settings._read_sync()
if _instance_cfg.basic_auth_user and _instance_cfg.basic_auth_password_hash:
    app.add_middleware(
        _BasicAuthMiddleware,
        username=_instance_cfg.basic_auth_user,
        password_hash=_instance_cfg.basic_auth_password_hash,
    )
    log.info("instance: Basic Auth enabled for user %r", _instance_cfg.basic_auth_user)

app.include_router(auth_routes.router)
app.include_router(speakers_routes.router)
app.include_router(voices_routes.router)
app.include_router(memory_routes.router)
app.include_router(pending_routes.router)
app.include_router(push_routes.router)
app.include_router(voicemail_routes.router)
app.include_router(agents_routes.router)
app.include_router(devices_routes.router)
app.include_router(hotkey_routes.router)
app.include_router(instance_routes.router)
app.include_router(items_routes.router)
app.include_router(step_up_routes.router)
app.include_router(stream_routes.router)


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

    Surfaces wake-word knobs (configurable via env) and instance-level
    flags the UI needs to decide which controls to show.
    """
    cfg = await instance_settings.read()
    return {
        "wake_word": {
            "name": WAKE_WORD_NAME,
            "threshold": WAKE_WORD_THRESHOLD,
        },
        "instance": {
            "registration_open": cfg.registration_open,
            "allow_guest_voice": cfg.allow_guest_voice,
            "basic_auth_configured": bool(
                cfg.basic_auth_user and cfg.basic_auth_password_hash
            ),
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
async def stats(
    range: str = "week",
    _user: dict = Depends(_current_user),
) -> JSONResponse:
    """Token usage + operational stats for the observability dashboard.

    Requires a logged-in session (``va_session`` cookie): the payload
    exposes per-user token totals, per-client breakdown, live session
    counts and cost — owner-level operational data, not public.

    ``range`` controls the lookback window: ``day`` (1 d), ``week`` (7 d,
    default), ``month`` (30 d).  Unknown values fall back to ``week`` —
    no 400, so a buggy client never breaks the page.

    Payload:
      • ``daily``      — per-day prompt/completion totals (stacked bar)
      • ``per_tool``   — per-tool token totals (horizontal bar)
      • ``per_user``   — per-client_id token totals (horizontal bar)
      • ``cost``       — projected $-cost per pricing tier
      • ``pricing``    — rate table used for ``cost``
      • ``range``      — echoed back for client validation
      • ``tool_perf``  — per-tool call count, avg latency, error rate
                         (from utterances, not token_usage — captures
                         every tool turn including zero-token fast-paths)
      • ``system``     — live health snapshot: active WS sessions,
                         reachable desktop agents, process uptime,
                         voice turns completed today
    """
    days = {"day": 1, "week": 7, "month": 30}.get(range, 7)

    # Run all DB queries concurrently — each holds the SQLite lock
    # briefly; parallelising them via gather shaves ~50 ms vs serial.
    import asyncio as _asyncio
    (
        daily,
        per_tool,
        per_user,
        tool_perf,
        turns_today,
    ) = await _asyncio.gather(
        get_daily_usage(days),
        get_per_tool_usage(days),
        get_per_user_usage(days),
        get_tool_perf(days),
        get_voice_turns_today(),
    )
    cost = compute_projected_cost(daily)

    # System snapshot — read from in-process state, no I/O.
    agents = desktop_client.list_agents()
    system = {
        "active_sessions": len(registry._sessions),
        "agents_total": len(agents),
        "agents_reachable": sum(1 for a in agents if a.reachable),
        "uptime_s": int(_time.time() - _START_TIME),
        "turns_today": turns_today,
    }

    return JSONResponse(
        {
            "range": range,
            "days": days,
            "daily": daily,
            "per_tool": per_tool,
            "per_user": per_user,
            "cost": cost,
            "pricing": PRICING,
            "tool_perf": tool_perf,
            "system": system,
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
async def ws_endpoint(
    websocket: WebSocket,
    client_id: str | None = None,
    device_kind: str | None = None,
):
    """Main voice-assistant WebSocket.

    Query params:
      ``client_id``   — stable browser/agent identifier (persisted to DB).
      ``device_kind`` — type of this client: ``"web"`` (browser PWA, default),
                        ``"macos_agent"`` or ``"linux_agent"`` (desktop-agent
                        process).  Controls which tool tier is visible for this
                        session — device-tier tools are hidden when
                        ``device_kind`` doesn't match.
    """
    await handle_ws(websocket, client_id=client_id, device_kind=device_kind)


app.mount(
    "/",
    # Native service runs from orchestrator/ with the PWA one level up
    # (STATIC_DIR=../frontend); the docker-compose path mounts ../frontend
    # at /app/static, which stays the default.
    StaticFiles(directory=os.environ.get("STATIC_DIR", "/app/static"), html=True),
    name="static",
)
