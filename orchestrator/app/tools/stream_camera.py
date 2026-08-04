"""
stream_camera — start a live MJPEG camera feed visible on any family device.

Opens a MJPEG stream from the desktop-agent's camera and pushes the
URL to the frontend via the ``media_sink`` / ``stream_started`` WS
event.  The browser renders it inline as ``<img src="/api/stream/camera">``.

Typical triggers:
  "покажи камеру"
  "stream the camera to the TV"
  "show me the camera feed"
  "start camera stream"
  "show what the camera sees live"

Architecture:
  tool  →  media_sink("/api/stream/camera?fps=15")  →  WS "stream_started"
        →  browser renders <img src="/api/stream/camera?fps=15">
        →  orchestrator /api/stream/camera  →  desktop-agent /v1/stream/camera
        →  OpenCV MJPEG loop

The stream keeps running until the user closes it (stop button) or
navigates away.  The tool itself just fires the start event and returns
a spoken confirmation — it does NOT wait for the stream to end.

Risk = read.  Streaming from the host camera is privacy-relevant; the
DESKTOP_TOKEN and va_session auth gates protect access.

Requirements:
  * desktop-agent running on the host.
  * Camera accessible to the agent (OpenCV device 0).
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

# Default streaming fps offered by this tool.  Users rarely need more
# than 15 fps to monitor a room; lower values reduce CPU/bandwidth.
_DEFAULT_FPS: int = 15


def _schema() -> dict:
    props: dict = {
        "fps": {
            "type": "integer",
            "description": (
                "Target frame rate (1–30).  Default 15.  Lower = less "
                "bandwidth, higher = smoother motion."
            ),
            "default": _DEFAULT_FPS,
        },
    }
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": "Which desktop-agent's camera to stream.  Omit for the default.",
        }
    return {"type": "object", "properties": props, "required": []}


@tool(
    name="stream_camera",
    description=(
        "Stream what the CAMERA sees to any family device.  The subject is "
        "the camera — pick this whenever the user names the camera, no "
        "matter which device they want it shown on.  Use for 'покажи "
        "камеру', 'покажи камеру на телевизоре', 'stream the camera', "
        "'show the camera feed', 'покажи что видит камера в прямом эфире', "
        "'show me the camera live on the TV', 'start camera stream'.  "
        "For streaming a Chrome TAB rather than the camera, use "
        "`stream_tab` — mentioning a TV or another device does not make it "
        "a tab.  Returns a live video feed visible on the current device "
        "and any other device connected to the assistant.  "
        "Requires the desktop-agent running on the host with camera access."
    ),
    params_schema=_schema(),
    risk="read",
    tier="device",
    device_kind="macos_agent",
)
async def stream_camera(
    *,
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

    if not desktop_client.has_capability_cached("camera", agent_id=chosen_agent):
        return ToolResult(
            text=t("camera.unavailable", lang),
            data={"error": "camera_unavailable"},
        )

    fps = max(1, min(int(fps or _DEFAULT_FPS), 30))

    # Build the orchestrator-relative stream URL.
    params = {"fps": fps}
    if chosen_agent:
        params["agent_id"] = chosen_agent
    stream_url = "/api/stream/camera?" + urlencode(params)

    # Notify the frontend to start rendering the stream.
    await cx.progress("stream", None)
    await cx.media(stream_url, "camera")

    text = t("stream.camera_started", lang)
    return ToolResult(
        text=text,
        data={
            "stream_url": stream_url,
            "source": "camera",
            "fps": fps,
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
