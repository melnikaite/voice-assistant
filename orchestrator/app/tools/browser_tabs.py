"""
list_browser_tabs — show what Chrome tabs are currently open.

Bridges desktop-agent's ``/v1/browser/tabs`` endpoint (Chrome
DevTools Protocol) so the user can ask "what do I have open in
Chrome" or "what tabs are open" and get a voice-friendly answer.

Typical triggers:
  "что у меня открыто в браузере"
  "what tabs do I have open"
  "list my Chrome tabs"
  "покажи открытые вкладки"

Requirements:
  * desktop-agent running on the host.
  * Chrome started with ``--remote-debugging-port=9222``
    (or whatever CHROME_DEBUG_PORT is set to).

Risk = read.  We're only listing titles and URLs — no content
is read and no action is taken.  The DESKTOP_TOKEN on the agent
side is the enforcing gate.
"""
from __future__ import annotations

import logging

from .. import desktop_client
from ..i18n import t
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

# Always advertise the optional agent_id param.  Resolution happens at
# call time via desktop_client.get_agent(), so agents that connect via
# reverse-WSS AFTER startup are still reachable (the old import-time
# snapshot left them invisible to the LLM until a restart).
_INCLUDE_AGENT_ID = True


def _schema() -> dict:
    props: dict = {}
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": (
                "Which desktop-agent to query.  Omit for the default agent."
            ),
        }
    return {"type": "object", "properties": props, "required": []}


@tool(
    name="list_browser_tabs",
    description=(
        "READ-ONLY enumeration of the Chrome tabs currently open on the "
        "user's computer — it returns a list and changes nothing.  Use when "
        "the user says 'what tabs do I have open', 'что у меня в браузере', "
        "'show me my Chrome tabs', 'list open tabs', or similar.  "
        "It CANNOT close, open, reorder, focus or otherwise act on a tab — "
        "any request that CHANGES browser or window state ('close every tab "
        "except the first', 'switch to the other window') belongs to "
        "`computer_use`, even though it mentions tabs.  Requires Chrome "
        "running with --remote-debugging-port and the desktop-agent running."
    ),
    params_schema=_schema(),
    risk="read",
    tier="device",
    device_kind="macos_agent",
)
async def list_browser_tabs(
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

    if not desktop_client.is_reachable(chosen_agent):
        return ToolResult(
            text=t("desktop.legacy_unavailable", lang),
            data={"error": "desktop_unavailable"},
        )

    await cx.progress("browser", None)
    try:
        tabs = await desktop_client.browser_list_tabs(agent_id=chosen_agent)
    except desktop_client.DesktopUnavailable as exc:
        log.info("list_browser_tabs: desktop unavailable: %s", exc)
        return ToolResult(
            text=t("browser.cdp_unavailable", lang),
            data={"error": "cdp_unavailable", "detail": str(exc)},
        )
    except Exception as exc:
        log.exception("list_browser_tabs: failed")
        return ToolResult(
            text=t("browser.cdp_unavailable", lang),
            data={"error": "browse_failed", "detail": str(exc)},
        )

    if not tabs:
        return ToolResult(
            text=t("browser.no_tabs", lang),
            data={"tabs": [], **({"agent_id": chosen_agent} if chosen_agent else {})},
        )

    # Format a voice-friendly list: first 8 tabs, titles truncated at 60 chars.
    MAX_TABS = 8
    shown = tabs[:MAX_TABS]
    lines = []
    for i, tab in enumerate(shown, 1):
        title = (tab.get("title") or tab.get("url") or "Untitled")[:60]
        lines.append(f"{i}. {title}")
    if len(tabs) > MAX_TABS:
        lines.append(f"… and {len(tabs) - MAX_TABS} more.")

    count = len(tabs)
    summary = "\n".join(lines)
    text = f"{count} tab{'s' if count != 1 else ''} open:\n{summary}"

    return ToolResult(
        text=text,
        data={
            "count": count,
            "tabs": [{"id": t["id"], "title": t.get("title"), "url": t.get("url")} for t in tabs],
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
