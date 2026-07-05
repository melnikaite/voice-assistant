"""
navigate_browser — open a URL in Chrome (or navigate an existing tab).

Uses desktop-agent's ``/v1/browser/navigate`` endpoint (Chrome
DevTools Protocol) to either open a new Chrome tab or navigate an
existing one to the requested URL.

Typical triggers:
  "открой youtube.com"
  "перейди на gmail.com"
  "go to github.com"
  "open google.com in Chrome"
  "navigate to wikipedia"

Risk = low_write.  Navigation changes the tab's URL — a side effect,
but immediately reversible by going back.  The DESKTOP_TOKEN is the
primary gate at the daemon layer.

Requirements:
  * desktop-agent running on the host.
  * Chrome started with ``--remote-debugging-port=9222``
    (or the CHROME_DEBUG_PORT configured value).
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
    props: dict = {
        "url": {
            "type": "string",
            "description": (
                "The full URL to open, including the scheme "
                "(e.g. 'https://github.com').  If the user gave a bare "
                "domain like 'youtube.com', add 'https://' before it."
            ),
        },
        "tab_id": {
            "type": "string",
            "description": (
                "Navigate this specific tab instead of opening a new one.  "
                "Omit to always open a new tab (recommended; avoids "
                "interrupting the user's current page)."
            ),
        },
    }
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": "Which desktop-agent to use.  Omit for the default.",
        }
    return {"type": "object", "properties": props, "required": ["url"]}


@tool(
    name="navigate_browser",
    description=(
        "Open a URL in Chrome, or navigate an existing Chrome tab to that URL.  "
        "Use when the user says 'открой', 'перейди на', 'go to', 'open in "
        "Chrome', 'navigate to', or gives a URL and clearly wants the browser "
        "to load it (e.g. 'open youtube.com').  "
        "Requires Chrome running with --remote-debugging-port and the "
        "desktop-agent running on the host.  "
        "Do NOT use for web searches — use web_search for that.  "
        "Do NOT fabricate URLs — only navigate to URLs the user explicitly provided."
    ),
    params_schema=_schema(),
    risk="low_write",
    tier="device",
    device_kind="macos_agent",
)
async def navigate_browser(
    url: str,
    *,
    ctx=None,
    tab_id: str | None = None,
    agent_id: str | None = None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang

    # Normalise: bare domain → https://
    url = url.strip()
    if url and not url.startswith(("http://", "https://", "ftp://")):
        url = f"https://{url}"

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
        result = await desktop_client.browser_navigate(
            url, tab_id=tab_id, agent_id=chosen_agent,
        )
    except desktop_client.DesktopUnavailable as exc:
        log.info("navigate_browser: desktop unavailable: %s", exc)
        return ToolResult(
            text=t("browser.cdp_unavailable", lang),
            data={"error": "cdp_unavailable", "detail": str(exc)},
        )
    except Exception as exc:
        log.exception("navigate_browser: failed")
        return ToolResult(
            text=t("browser.cdp_unavailable", lang),
            data={"error": "navigate_failed", "detail": str(exc)},
        )

    resolved_url = result.get("url") or url
    resolved_tab = result.get("tab_id") or ""
    text = t("browser.navigate_done", lang, url=resolved_url)
    return ToolResult(
        text=text,
        data={
            "url": resolved_url,
            "tab_id": resolved_tab,
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
