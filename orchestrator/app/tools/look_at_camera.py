"""
look_at_camera — visual Q&A via the host's physical camera.

Bridges the same two services as ``look_at_screen``, but captures
from the camera device (webcam / built-in iSight) instead of the
screen:

  * ``desktop-agent`` provides ``/v1/camera`` (JPEG frame via OpenCV).
  * ``vision.py`` routes multimodal Q&A through the project's primary
    local LLM (Gemma 4 via LM Studio).  Fully offline, no API key.

Typical triggers: "посмотри камерой", "что видит камера", "look
through the camera", "what do you see on camera", "check the room".

The LLM prompt asks for a concise description that includes a brief
quality assessment (sharpness, lighting, motion) — this is the
"focus/lighting/motion diagnostics" feature requested in #47.  Simple
sentence like "Image is slightly dark but sharp; the room appears…"
is enough; no hard-coded thresholds or numeric metrics, just what the
model perceives.

Risk = read.  Camera capture is privacy-relevant but the user
explicitly invoked the tool; the shared ``DESKTOP_TOKEN`` is the
strong gate at the daemon layer.
"""
from __future__ import annotations

import logging

from .. import desktop_client, vision
from ..i18n import t
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


# Always advertise the optional agent_id param.  Resolution happens at
# call time via desktop_client.get_agent(), so agents that connect via
# reverse-WSS AFTER startup are still reachable (the old import-time
# snapshot left them invisible to the LLM until a restart).
_INCLUDE_AGENT_ID = True


def _schema() -> dict:
    props: dict = {
        "question": {
            "type": "string",
            "description": (
                "What to look for through the camera.  Be specific: "
                "'is anyone in the room' or 'what's on the table' "
                "beats a generic 'what do you see'.  Pass the user's "
                "original question when unsure."
            ),
        },
    }
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": (
                "Which desktop-agent's camera to use.  Omit for the "
                "default agent."
            ),
        }
    return {"type": "object", "properties": props, "required": ["question"]}


# Vision prompt that drives focus/lighting/motion diagnostics.
# Kept out of the decorator so it's easy to tune without touching the
# schema or the decorator block.
_VISION_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Also include a brief quality note (one clause is fine): is the image "
    "sharp or blurry, well-lit or dark, and is anything in motion or blurred?"
)


@tool(
    name="look_at_camera",
    description=(
        "Look through the device camera and answer a question about what it "
        "sees.  Use when the user says 'посмотри камерой', 'what does the "
        "camera see', 'look through the camera', 'check the room', "
        "'is anyone there', or similar phrases in any language.  "
        "Returns a concise voice-friendly answer that also notes image "
        "quality (sharpness, lighting, motion).  Requires the desktop-agent "
        "running on the host and a camera device accessible to it."
    ),
    params_schema=_schema(),
    risk="read",
    tier="device",
    device_kind="macos_agent",
)
async def look_at_camera(
    question: str,
    *,
    ctx=None,
    agent_id: str | None = None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang

    chosen_agent: str | None = None
    if _INCLUDE_AGENT_ID or agent_id is not None:
        match = desktop_client.get_agent(agent_id)
        chosen_agent = match.agent_id if match else None

    # Guard: agent must be reachable and advertise camera capability.
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

    await cx.progress("camera", None)
    try:
        jpg_bytes = await desktop_client.camera_capture(agent_id=chosen_agent)
    except desktop_client.DesktopUnavailable as exc:
        log.info("look_at_camera: desktop unavailable: %s", exc)
        return ToolResult(
            text=t("desktop.legacy_unavailable", lang),
            data={
                "error": "desktop_unavailable", "detail": str(exc),
                **({"agent_id": chosen_agent} if chosen_agent else {}),
            },
        )
    except Exception as exc:
        log.exception("look_at_camera: camera capture failed")
        return ToolResult(
            text=t("camera.failed", lang),
            data={"error": "camera_failed", "detail": str(exc)},
        )

    if not jpg_bytes:
        return ToolResult(
            text=t("camera.failed", lang),
            data={"error": "empty_frame"},
        )

    # Build a prompt that requests diagnostics alongside the answer.
    vision_prompt = _VISION_PROMPT_TEMPLATE.format(question=question)

    await cx.progress("vision", None)
    try:
        answer = await vision.analyze_image_bytes(
            jpg_bytes,
            vision_prompt,
            client_id=cx.client_id,
            tool_name="look_at_camera",
        )
    except Exception as exc:
        log.exception("look_at_camera: vision call failed")
        return ToolResult(
            text=t("vision.failed", lang),
            data={"error": "vision_failed", "detail": str(exc)},
        )

    if not answer:
        return ToolResult(
            text=t("vision.empty", lang),
            data={
                "error": "empty_answer", "bytes": len(jpg_bytes),
                **({"agent_id": chosen_agent} if chosen_agent else {}),
            },
        )
    return ToolResult(
        text=answer,
        data={
            "bytes": len(jpg_bytes), "question": question,
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
