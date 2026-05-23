"""
look_at_screen — visual Q&A on the host's current screen.

Bridges two host-side services we already have:

  * ``desktop-agent`` provides ``/v1/screenshot`` (raw PNG of the host
    display).
  * ``vision.py`` routes multimodal Q&A through the project's primary
    local LLM (Gemma 4 via LM Studio).  No cloud provider, no API key
    — vision works as long as LM Studio is running on the host.

The tool exists for utterances like "what's on my screen", "read the
error in the window", "what's playing in Apple Music".  It is
read-only — never controls the screen, never moves the mouse, never
types.  Driving the screen lives in the ``desktop`` tool, which can
optionally chain through here as a fallback when AppleScript can't
locate an element by name.

Risk = read.  Even though it physically captures the screen (which is a
privacy-relevant act), the user invoking the tool clearly meant to;
gating it behind a passphrase would force a re-auth every time, which
defeats the point of "just tell me what's on screen".  The strong
control is at the daemon layer: ``desktop-agent`` won't honour the
``/v1/screenshot`` request without the shared ``DESKTOP_TOKEN``.
"""
from __future__ import annotations

import logging

from .. import desktop_client, vision
from ..i18n import t
from .base import ToolResult, tool

log = logging.getLogger(__name__)


# Whether to advertise the ``agent_id`` field to the LLM.  Computed once
# at import; agents don't get added mid-process (operator restarts the
# orchestrator after editing DESKTOP_AGENTS).  Same pattern is used by
# tools/computer_use.py — the dynamic-schema trick avoids surfacing
# an extra param when the user only has one device anyway.
_INCLUDE_AGENT_ID = len(desktop_client.list_agents()) > 1


def _schema() -> dict:
    """Build the LLM-visible schema; hides agent_id in single-agent installs."""
    props: dict = {
        "question": {
            "type": "string",
            "description": (
                "What to look for.  Be specific: 'what error is shown' "
                "beats 'what's on screen' — the visual model answers "
                "more precisely with focused questions.  Pass the "
                "user's original question if unsure."
            ),
        },
    }
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": (
                "Which desktop-agent to screenshot.  Use when the user "
                "names a device (e.g. 'look at the Mac screen', "
                "'what's on the work-PC screen').  Omit to use the "
                "default agent."
            ),
        }
    return {"type": "object", "properties": props, "required": ["question"]}


@tool(
    name="look_at_screen",
    description=(
        "Look at the user's screen and answer a question about what is on "
        "it.  Use when the user asks 'what's on my screen', 'read this "
        "error', 'what's there right now', 'what's playing', 'what does "
        "this dialog say' in any supported language.  Returns a concise "
        "voice-friendly answer.  Captures the screen via the host-side "
        "desktop-agent; requires the daemon to be running."
    ),
    params_schema=_schema(),
    risk="read",
)
async def look_at_screen(
    question: str, *, ctx=None, agent_id: str | None = None,
) -> ToolResult:
    progress = getattr(ctx, "progress_sink", None) if ctx else None
    client_id = getattr(ctx, "client_id", None) if ctx else None
    lang = getattr(ctx, "user_lang", None) if ctx else None

    # Resolve which agent to talk to BEFORE the call so we can stamp
    # the resolved id into ToolResult.data — the WS layer surfaces
    # that as `target_agent` for the UI to highlight.
    chosen_agent: str | None = None
    if _INCLUDE_AGENT_ID or agent_id is not None:
        match = desktop_client.get_agent(agent_id)
        chosen_agent = match.agent_id if match else None

    # Vision now rides on the same local LLM as everything else
    # (Gemma 4 multimodal via LM Studio).  If LLM_URL is unreachable
    # the orchestrator wouldn't have started, so we skip an `available`
    # gate here and let the vision call surface the transport error
    # if LM Studio went down mid-session.

    if progress is not None:
        await progress("screenshot", None)
    try:
        png_bytes = await desktop_client.screenshot(agent_id=chosen_agent)
    except desktop_client.DesktopUnavailable as exc:
        log.info("look_at_screen: desktop unavailable: %s", exc)
        return ToolResult(
            text=t("desktop.legacy_unavailable", lang),
            data={
                "error": "desktop_unavailable", "detail": str(exc),
                **({"agent_id": chosen_agent} if chosen_agent else {}),
            },
        )
    except Exception as exc:
        log.exception("look_at_screen: screenshot failed")
        return ToolResult(
            text=t("vision.failed", lang),
            data={"error": "screenshot_failed", "detail": str(exc)},
        )

    if not png_bytes:
        return ToolResult(
            text=t("vision.failed", lang),
            data={"error": "empty_screenshot"},
        )

    if progress is not None:
        await progress("vision", None)
    try:
        answer = await vision.analyze_image_bytes(
            png_bytes,
            question,
            client_id=client_id,
            tool_name="look_at_screen",
        )
    except Exception as exc:
        log.exception("look_at_screen: vision call failed")
        return ToolResult(
            text=t("vision.failed", lang),
            data={"error": "vision_failed", "detail": str(exc)},
        )

    if not answer:
        return ToolResult(
            text=t("vision.empty", lang),
            data={
                "error": "empty_answer", "bytes": len(png_bytes),
                **({"agent_id": chosen_agent} if chosen_agent else {}),
            },
        )
    return ToolResult(
        text=answer,
        data={
            "bytes": len(png_bytes), "question": question,
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
