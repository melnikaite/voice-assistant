"""
read_browser_tab — read a Chrome tab's content and answer a question.

Extracts the visible text of a Chrome tab via CDP
(``document.body.innerText``) and passes it to the local LLM
(Gemma 4 via LM Studio) to answer the user's question.  Fully
offline — no cloud provider, no API key needed.

Typical triggers:
  "прочитай эту страницу"
  "что написано на этом сайте"
  "summarize the current tab"
  "read what's in this article"
  "о чём эта страница"
  "what does this page say about <topic>"

Architecture:
  desktop-agent /v1/browser/page_text → text
  llm_utils.chat(text + question) → answer (spoken)

Text is capped at PAGE_TEXT_LIMIT characters before sending to the
LLM — enough for a full article but avoids context-window overflow on
very large pages.  LLM synthesises a concise voice-friendly answer.

Risk = read.  We only read, never modify, the page.  The DESKTOP_TOKEN
is the primary gate at the agent layer.

Requirements:
  * desktop-agent running on the host.
  * Chrome started with ``--remote-debugging-port=9222``
    (or the CHROME_DEBUG_PORT configured value).
  * LLM running (LM Studio / Ollama).
"""
from __future__ import annotations

import logging

from .. import desktop_client
from ..i18n import t
from ..llm_utils import chat, extract_text
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

# Text extracted from the tab is truncated here before sending to the
# LLM.  8 000 chars ≈ 2 000 tokens — fits comfortably in a 4 k context
# window while covering most articles and doc pages in full.
PAGE_TEXT_LIMIT = 8_000

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
                "What the user wants to know about the page.  "
                "Can be 'summarise this page' for a general summary, "
                "or a specific question like 'what is the pricing?' or "
                "'who is the author?'.  Pass the user's original words."
            ),
        },
        "tab_id": {
            "type": "string",
            "description": (
                "Read this specific tab.  Omit to use the first open "
                "Chrome page tab."
            ),
        },
    }
    if _INCLUDE_AGENT_ID:
        props["agent_id"] = {
            "type": "string",
            "description": "Which desktop-agent to use.  Omit for the default.",
        }
    return {"type": "object", "properties": props, "required": ["question"]}


# System prompt for the LLM Q&A call.  Short and instruction-focused —
# the page content is passed in the user message.
_SYSTEM = (
    "You are a voice assistant helping the user understand a web page they "
    "have open in Chrome.  Answer their question based only on the page "
    "content provided.  Be concise — your answer will be read aloud.  "
    "If the page content doesn't contain enough information to answer, "
    "say so briefly."
)


@tool(
    name="read_browser_tab",
    description=(
        "Read the content of the current Chrome tab and answer a question "
        "about it.  Use when the user says 'прочитай эту страницу', 'что "
        "написано здесь', 'summarise this page', 'read this article', "
        "'о чём эта страница', 'what does this say', or asks a specific "
        "question about the current browser page.  "
        "Requires Chrome running with --remote-debugging-port and the "
        "desktop-agent running on the host."
    ),
    params_schema=_schema(),
    risk="read",
    tier="device",
    device_kind="macos_agent",
)
async def read_browser_tab(
    question: str,
    *,
    ctx=None,
    tab_id: str | None = None,
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

    # ── 1. Fetch page text from Chrome ────────────────────────────────────
    await cx.progress("browser", None)
    try:
        page_info = await desktop_client.browser_page_text(
            tab_id=tab_id, agent_id=chosen_agent,
        )
    except desktop_client.DesktopUnavailable as exc:
        log.info("read_browser_tab: desktop unavailable: %s", exc)
        return ToolResult(
            text=t("browser.cdp_unavailable", lang),
            data={"error": "cdp_unavailable", "detail": str(exc)},
        )
    except Exception as exc:
        log.exception("read_browser_tab: page_text failed")
        return ToolResult(
            text=t("browser.read_failed", lang),
            data={"error": "page_text_failed", "detail": str(exc)},
        )

    raw_text = (page_info.get("text") or "").strip()
    page_title = page_info.get("title") or ""
    page_url = page_info.get("url") or ""
    resolved_tab = page_info.get("tab_id") or tab_id or ""

    if not raw_text:
        return ToolResult(
            text=t("browser.read_failed", lang),
            data={"error": "empty_page", "url": page_url, "tab_id": resolved_tab},
        )

    # Truncate before sending to LLM.
    text = raw_text[:PAGE_TEXT_LIMIT]
    if len(raw_text) > PAGE_TEXT_LIMIT:
        text += "\n[… content truncated …]"

    # ── 2. LLM Q&A on page content ────────────────────────────────────────
    await cx.progress("reading", None)
    label = f'"{page_title}"' if page_title else page_url or "this page"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Page: {label}\nURL: {page_url}\n\n"
                f"Content:\n{text}\n\n"
                f"Question: {question}"
            ),
        },
    ]
    try:
        choice = await chat(
            messages,
            temperature=0.2,
            reasoning_effort="low",
            client_id=cx.client_id,
            tool_name="read_browser_tab",
        )
        answer = extract_text(choice["message"]).strip()
    except Exception as exc:
        log.exception("read_browser_tab: LLM call failed")
        return ToolResult(
            text=t("browser.read_failed", lang),
            data={"error": "llm_failed", "detail": str(exc)},
        )

    if not answer:
        return ToolResult(
            text=t("browser.read_failed", lang),
            data={"error": "empty_answer", "url": page_url},
        )

    return ToolResult(
        text=answer,
        data={
            "question": question,
            "url": page_url,
            "title": page_title,
            "tab_id": resolved_tab,
            "chars_read": len(raw_text),
            **({"agent_id": chosen_agent} if chosen_agent else {}),
        },
    )
