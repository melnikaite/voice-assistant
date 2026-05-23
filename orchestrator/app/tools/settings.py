"""
Per-user settings.json — read and update via LLM-visible tools.

The orchestrator's ``user_files.UserSettings`` Pydantic schema defines
the typed surface (language, tts_voice, formality, style_prompt,
permissions, custom).  These two tools let the user say "keep replies
short" or "address me formally from now on" and have the assistant
persist that preference into their settings file.

Split into ``read_settings`` (tier-1) and ``update_settings``
(tier-2, ``high_write``) for the same reason as memory: per-tool risk.
"""
from __future__ import annotations

import logging

from ..i18n import t
from ..user_files import UserSettings, patch_settings, read_settings
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


def _summarise_settings(s: UserSettings, lang: str | None) -> str:
    """Spoken-friendly summary of the user's typed settings."""
    bits = [
        t("settings.field.language", lang, value=s.language),
        t("settings.field.formality", lang, value=s.formality),
    ]
    if s.tts_voice:
        bits.append(t("settings.field.voice", lang, value=s.tts_voice))
    if s.style_prompt:
        # Trim to keep TTS sane — the full prompt can be long.
        snippet = s.style_prompt[:80] + ("…" if len(s.style_prompt) > 80 else "")
        bits.append(t("settings.field.style", lang, snippet=snippet))
    bits.append(t("settings.field.permissions", lang, list=", ".join(s.permissions)))
    return "; ".join(bits)


# Whitelist of fields the LLM can write via update_settings.  We do
# NOT expose `code_word_hash` — that's the auth secret and must be set
# through the explicit UI / dedicated endpoint, not via voice tool.
_EDITABLE_FIELDS = {"language", "tts_voice", "formality", "style_prompt"}


@tool(
    name="read_settings",
    description=(
        "Read the speaker's TYPED PREFERENCES — language, formality "
        "level, TTS voice, style prompt.  Use when the user asks "
        "'what are my settings', 'how do you address me', 'what voice "
        "am I on', 'what language is set'.\n"
        "This is NOT the free-form memory file — for 'what do you "
        "remember about me' use `read_memory`.  Not the past-"
        "conversation log either — for 'what did we talk about' use "
        "`my_history`."
    ),
    params_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    risk="read",
)
async def read_settings_tool(*, ctx) -> ToolResult:
    cx = unwrap_ctx(ctx)
    profile_id = cx.profile_id
    if profile_id is None:
        return ToolResult(text=t("settings.no_profile", cx.user_lang), data={"error": "no_profile"})
    s = await read_settings(profile_id)
    summary = _summarise_settings(s, cx.user_lang)
    return ToolResult(
        text=t("settings.read", cx.user_lang, summary=summary),
        data={"settings": s.model_dump()},
    )


@tool(
    name="update_settings",
    description=(
        "Change one typed preference on the speaker's profile.\n"
        "Allowed fields:\n"
        "  • language  — 'ru' | 'de' | 'en' | 'auto'\n"
        "  • formality — 'formal' | 'casual' | 'kid'\n"
        "  • tts_voice — XTTS speaker name (e.g. 'Claribel Dervla' or "
        "'clone:3' for a custom voice)\n"
        "  • style_prompt — freeform text injected into the system "
        "prompt (e.g. 'keep replies short', 'avoid profanity', 'I "
        "like jokes in answers')\n"
        "Use when the user explicitly says 'address me formally', "
        "'reply in Russian', 'keep it short'."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "enum": sorted(_EDITABLE_FIELDS),
                "description": "Which setting to change.",
            },
            "value": {
                "type": "string",
                "description": "New value as a string (will be validated against schema).",
            },
        },
        "required": ["field", "value"],
    },
    risk="high_write",
)
async def update_settings(*, ctx, field: str, value: str) -> ToolResult:
    cx = unwrap_ctx(ctx)
    profile_id = cx.profile_id
    if profile_id is None:
        return ToolResult(text=t("settings.no_profile", cx.user_lang), data={"error": "no_profile"})
    if field not in _EDITABLE_FIELDS:
        return ToolResult(
            text=t("settings.bad_field", cx.user_lang),
            data={"error": f"unknown_field:{field!r}"},
        )
    try:
        updated = await patch_settings(profile_id, **{field: value})
    except Exception as exc:
        log.warning("update_settings: validation failed %s=%r: %s", field, value, exc)
        return ToolResult(
            text=t("settings.unknown_value", cx.user_lang, value=value, field=field),
            data={"error": f"validation:{exc.__class__.__name__}"},
        )
    return ToolResult(
        text=t("settings.updated", cx.user_lang, field=field, value=value),
        data={"field": field, "value": value, "settings": updated.model_dump()},
    )
