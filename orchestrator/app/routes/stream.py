"""Live MJPEG stream proxy (#53).

Proxies the desktop-agent's ``/v1/stream/{source}`` endpoint so family
devices can view the camera or a browser tab using only their
``va_session`` cookie — they never need to know the agent URL or
DESKTOP_TOKEN.

Endpoints
─────────
GET /api/stream/camera                         — camera MJPEG stream
GET /api/stream/tab                            — Chrome tab MJPEG stream
GET /api/stream/camera?fps=15&agent_id=macbook
GET /api/stream/tab?tab_id=A1&fps=5

All endpoints require the ``va_session`` cookie (same auth as /api/me).
``allow_guest`` sessions also work — family devices are often guests.

The orchestrator acts as the relay: it adds the per-agent
``X-Desktop-Token`` header and forwards the response body verbatim.
"""
from __future__ import annotations

import logging
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Cookie, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import desktop_client
from ..storage import get_session as auth_get_session

router = APIRouter()
log = logging.getLogger(__name__)

_STREAM_BOUNDARY = "va_frame"
_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)


async def _require_session(va_session: str | None) -> dict:
    """Resolve va_session cookie → profile dict.  Raises 401 on failure."""
    if not va_session:
        raise HTTPException(401, "not authenticated")
    sess = await auth_get_session(va_session)
    if not sess:
        raise HTTPException(401, "session expired or invalid")
    return sess


async def _proxy_stream(agent_path: str, agent_id: str | None) -> StreamingResponse:
    """Open a streaming HTTP connection to the desktop-agent and relay it.

    ``agent_path`` is the path on the agent, e.g. ``/v1/stream/camera?fps=10``.
    Returns a ``StreamingResponse`` that yields multipart JPEG frames as
    they arrive from the agent.

    Raises 503 if the agent is not reachable, 404 if CDP says the tab is
    gone, 502 on any other upstream error.
    """
    info = desktop_client.get_agent(agent_id)
    if info is None or not info.url:
        raise HTTPException(503, "desktop agent not configured")
    if not info.reachable:
        raise HTTPException(503, "desktop agent not reachable")

    url = info.url.rstrip("/") + agent_path
    headers = {"X-Desktop-Token": info.token}

    # Open the upstream connection and read its status BEFORE returning the
    # StreamingResponse.  Raising inside the body generator can't change the
    # HTTP status — by then 200 + headers are already on the wire — so an
    # agent 404/503 would otherwise reach the browser as a truncated 200.
    client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)
    try:
        req = client.build_request("GET", url, headers=headers)
        resp = await client.send(req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        log.warning("stream proxy: connect error to %s", url)
        raise HTTPException(503, "desktop agent unreachable")
    except httpx.HTTPError as exc:
        await client.aclose()
        log.warning("stream proxy: upstream error to %s: %s", url, exc)
        raise HTTPException(502, "desktop agent stream error")

    if resp.status_code != 200:
        code = resp.status_code
        await resp.aclose()
        await client.aclose()
        if code == 503:
            raise HTTPException(503, "feature not available on this agent")
        if code == 404:
            raise HTTPException(404, "tab or source not found")
        raise HTTPException(502, f"agent returned {code}")

    async def _iter():
        try:
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                yield chunk
        except (httpx.ReadError, httpx.RemoteProtocolError) as exc:
            # Client or agent closed the stream — normal MJPEG end.
            log.debug("stream proxy: read ended (%s)", exc)
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _iter(),
        media_type=f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}",
        headers={"Cache-Control": "no-cache, no-store"},
    )


# ---------------------------------------------------------------------------
# Camera stream
# ---------------------------------------------------------------------------

@router.get("/api/stream/camera")
async def api_stream_camera(
    fps: float = Query(default=15.0, ge=1.0, le=30.0),
    agent_id: str | None = Query(default=None),
    va_session: str | None = Cookie(default=None),
) -> StreamingResponse:
    """Relay a live MJPEG camera stream from the desktop-agent.

    ``fps`` — target frame rate (1–30, default 15).  The agent caps this
    at its own maximum (30 fps for camera).
    """
    await _require_session(va_session)
    return await _proxy_stream("/v1/stream/camera?" + urlencode({"fps": fps}), agent_id)


# ---------------------------------------------------------------------------
# Browser-tab stream
# ---------------------------------------------------------------------------

@router.get("/api/stream/tab")
async def api_stream_tab(
    tab_id: str | None = Query(default=None),
    fps: float = Query(default=5.0, ge=1.0, le=15.0),
    agent_id: str | None = Query(default=None),
    va_session: str | None = Cookie(default=None),
) -> StreamingResponse:
    """Relay a live MJPEG Chrome-tab stream from the desktop-agent.

    ``tab_id`` — CDP page id.  Omit to use the currently active tab.
    ``fps`` — target frame rate (1–15, default 5).  Chrome screenshots
    are heavier than camera frames so the ceiling is lower.
    """
    await _require_session(va_session)
    params: dict[str, object] = {"fps": fps}
    if tab_id:
        params["tab_id"] = tab_id
    return await _proxy_stream("/v1/stream/tab?" + urlencode(params), agent_id)
