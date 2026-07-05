"""
stream_tab — start a live MJPEG stream of a Chrome tab.

Opens a MJPEG stream of a Chrome tab via the desktop-agent's CDP layer
and pushes the URL to the frontend.  The browser renders it inline as
``<img src="/api/stream/tab?tab_id=...">``.

Typical triggers:
  "покажи вкладку"
  "stream this tab"
  "share this tab to the TV"
  "stream the browser tab live"
  "покажи эту страницу на телевизоре"
  "share screen"

Architecture:
  tool  →  media_sink("/api/stream/tab?tab_id=...")  →  WS "stream_started"
        →  browser renders <img>
        →  orchestrator /api/stream/tab  →  desktop-agent /v1/stream/tab
        →  CDP Page.captureScreenshot loop

Risk = read.  Screen capture is privacy-relevant; DESKTOP_TOKEN and
va_session auth gates protect access.

Requirements:
  * desktop-agent running on the host.
  * Chrome running with --remote-debugging-port=9222.
"""
from __future__ import annotations

import logging
from urllib.parse import urlencode

from .. import desktop_client
from ..i18n import t
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

# Always advertise the optional agent_id param.  Resolution happens at
# call time via desktop_client.get_agent(), so agents that connect via
# reverse-WSS AFTER startup are still reachable (the old import-time
# snapshot left them invisible to the LLM until a restart).
_INCLUDE_AGENT_ID = True

# CDP tab screenshots are heavier than camera frames; 5 fps is smooth
# enough for a presentation/screen-share use case.
_DEFAULT_FPS: int = 5


def _schema() -> dict:
    props: dict = {
        "tab_id": {
            "type": "string",
            "description": (
                "CDP page id of the tab to stream.  Omit to stream the "
                "first open Chrome page tab (usually the active one)."
            ),
        },
        "fps": {
            "type": "integer",
            "description": (
                "Target frame rate (1–15).  Default 5.  Higher fps = "
                "smoother but more CPU/bandwidth."
            ),
            "default": _DEFAULT_FPS,
        },
    }
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": "Which desktop-agent to use.  Omit for the default.",
        }
    return {"type": "object", "properties": props, "required": []}


@tool(
    name="stream_tab",
    description=(
        "Start a live stream of a Chrome browser tab visible on any family "
        "device.  Use when the user says 'покажи вкладку', 'stream this tab', "
        "'share my screen', 'покажи эту страницу на телевизоре', "
        "'stream the browser', 'show this tab on the TV'.  "
        "Returns a live video feed of the current Chrome tab.  "
        "Requires Chrome running with --remote-debugging-port and the "
        "desktop-agent running on the host."
    ),
    params_schema=_schema(),
    risk="read",
    tier="device",
    device_kind="macos_agent",
)
async def stream_tab(
    *,
    tab_id: str | None = None,
    fps: int = _DEFAULT_FPS,
    ctx=None,
    agent_id: str | None = None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang

    chosen_agent: str | None = None
    if _INCLUDE_AGENT_ID or agent_id is not None:
        match = desktop_client.get_agent(agent_id)
        chosen_agent = match.agent_id if match else None

    if not desktop_client.is_reachable(chosen_agent):
        return ToolResult(
            text=t("desktop.legacy_unavailable", lang),
            data={"error": "desktop_unavailable"},
        )

    if not desktop_client.has_capability_cached("browser_cdp", agent_id=chosen_agent):
        return ToolResult(
            text=t("browser.cdp_unavailable", lang),
            data={"error": "cdp_unavailable"},
        )

    fps = max(1, min(int(fps or _DEFAULT_FPS), 15))

    # Build the orchestrator-relative stream URL.
    params: dict = {"fps": fps}
    if tab_id:
        params["tab_id"] = tab_id
    if chosen_agent:
        params["agent_id"] = chosen_agent
    stream_url = "/api/stream/tab?" + urlencode(params)

    await cx.progress("stream", None)
    await cx.media(stream_url, "tab")

    text = t("stream.tab_started", lang)
    return ToolResult(
        text=text,
        data={
            "stream_url": stream_url,
            "source": "tab",
            "fps": fps,
            **({"tab_id": tab_id} if tab_id else {}),
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
