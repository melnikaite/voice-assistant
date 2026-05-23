from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .. import desktop_client
from ._deps import _current_user

router = APIRouter()


@router.get("/api/agents")
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
