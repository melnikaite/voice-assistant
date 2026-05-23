"""
Persistent per-user memory.md — split into two LLM-visible tools.

Where the orchestrator's *semantic* memory finds relevant past
utterances on its own, these tools give the LLM explicit handles to
the speaker's free-form notes file at
``/data/users/<profile_id>/memory.md``.

Split into two tools rather than one ``memory(action=…)`` because the
``risk`` field on the decorator is per-tool, not per-action.  Splitting
lets ``read_memory`` stay tier-1 (anyone can ask about their own
diary) while ``remember`` is correctly classified as ``high_write``
and goes through passphrase gating.
"""
from __future__ import annotations

import logging

from ..i18n import t
from ..user_files import append_memory, read_memory, write_memory
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


@tool(
    name="read_memory",
    description=(
        "Read back the speaker's FREE-FORM Markdown notes from their "
        "memory.md.  Use when the user explicitly asks 'what do you "
        "remember about me', 'read my memory', 'what's in my notes'.\n"
        "This is NOT the past-conversation log — for 'what did we "
        "talk about yesterday about X' use `my_history`.  Not typed "
        "preferences either — for language/voice/style use "
        "`read_settings`."
    ),
    params_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    risk="read",
)
async def read_memory_tool(*, ctx) -> ToolResult:
    cx = unwrap_ctx(ctx)
    profile_id = cx.profile_id
    lang = cx.user_lang
    if profile_id is None:
        return ToolResult(text=t("memory.no_profile", lang), data={"error": "no_profile"})
    await cx.progress("memory_read", None)
    body = await read_memory(profile_id)
    if not body.strip():
        return ToolResult(text=t("memory.empty_storage", lang), data={"empty": True})
    return ToolResult(
        text=t("memory.read_header", lang, content=body),
        data={"content": body},
    )


@tool(
    name="remember",
    description=(
        "Persist a fact about the speaker into their memory.md.  Use "
        "when the user says 'remember that …', 'don't forget that …', "
        "'make a note that …'.  Two actions:\n"
        "  • action='append' + content=<one short sentence> — add a "
        "bulleted entry (most common case).\n"
        "  • action='replace' + content=<full markdown> — overwrite the "
        "whole memory file.  Use rarely; for 'forget everything I told "
        "you' pass an empty string."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["append", "replace"],
                "description": "Append a bullet or replace the whole file.",
            },
            "content": {
                "type": "string",
                "description": (
                    "For append: one short sentence to remember "
                    "(no need to add '- ', we'll bullet it). "
                    "For replace: the full new markdown body."
                ),
            },
        },
        "required": ["action", "content"],
    },
    risk="high_write",
)
async def remember(*, ctx, action: str, content: str) -> ToolResult:
    cx = unwrap_ctx(ctx)
    profile_id = cx.profile_id
    if profile_id is None:
        return ToolResult(text=t("memory.no_profile", cx.user_lang), data={"error": "no_profile"})

    if action == "append":
        if not content or not content.strip():
            return ToolResult(
                text=t("memory.empty_arg", cx.user_lang),
                data={"error": "empty_content"},
            )
        await cx.progress("memory_write")
        snippet = content.strip()
        new_body = await append_memory(profile_id, snippet)
        return ToolResult(
            text=t("memory.saved", cx.user_lang, snippet=snippet),
            data={"snippet": snippet, "new_body": new_body},
        )

    if action == "replace":
        await cx.progress("memory_write")
        await write_memory(profile_id, content or "")
        return ToolResult(
            text=t("memory.overwritten", cx.user_lang),
            data={"new_body": content or ""},
        )

    return ToolResult(
        text=t("memory.unknown_action", cx.user_lang),
        data={"error": f"unknown_action:{action!r}"},
    )
