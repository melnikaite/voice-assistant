"""
translate — translate user-supplied text into a target language via the LLM.

Single LLM call with a deliberately stripped system prompt: the model
returns ONLY the translation, no commentary, no quotes, no source-lang
preamble. ``reasoning_effort="low"`` because translation doesn't benefit
from extended chain-of-thought, and ``temperature=0.2`` because we want
deterministic-ish output for the same input.

The target-language argument is free-form (ISO 639-1 like "en" or a
natural-language name like "German").  Pass straight to the model —
modern multilingual LLMs handle both forms.
"""
from __future__ import annotations

import logging

import httpx

from ..i18n import t
from ..llm_utils import chat, extract_text
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a translator. Translate the user's text into the requested "
    "target language. Return ONLY the translation — no explanations, no "
    "quotes, no source-language preamble, no language tags. Preserve "
    "proper nouns, numbers, dates and technical terms verbatim. If the "
    "input is already in the target language, return it unchanged."
)


@tool(
    name="translate",
    description=(
        "Translate user-supplied text into a target language. Use when the "
        "user says 'translate this to German', 'how do you say this in "
        "English', 'translate ... to French', etc.  Arguments: `text` is "
        "the source phrase, `target_lang` is the destination language "
        "(ISO 639-1 like 'en', 'de' or a natural name like 'German')."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Source text to translate.",
            },
            "target_lang": {
                "type": "string",
                "description": (
                    "Target language — ISO 639-1 code ('en', 'de', 'fr', "
                    "'ru') or a natural-language name ('English', "
                    "'German', 'French')."
                ),
            },
        },
        "required": ["text", "target_lang"],
    },
    risk="read",
)
async def translate(text: str, target_lang: str, *, ctx=None) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang

    if not text or not text.strip():
        return ToolResult(text=t("translate.no_text", lang), data={"error": "no_text"})
    if not target_lang or not target_lang.strip():
        return ToolResult(text=t("translate.no_target", lang), data={"error": "no_target"})

    await cx.progress("translate", target_lang)

    log.info("translate: %d chars → %r", len(text), target_lang)

    user_msg = f"Target language: {target_lang}\nText: {text}"
    try:
        choice = await chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            reasoning_effort="low",
            client_id=cx.client_id,
            tool_name="translate",
        )
    except httpx.TimeoutException:
        log.warning("translate: timeout")
        return ToolResult(
            text=t("translate.failed", lang),
            data={"text": text, "target_lang": target_lang, "error": "timeout"},
        )
    except httpx.HTTPError as e:
        log.warning("translate: http error: %s", e)
        return ToolResult(
            text=t("translate.failed", lang),
            data={
                "text": text,
                "target_lang": target_lang,
                "error": f"{e.__class__.__name__}",
            },
        )

    out = extract_text(choice["message"]).strip()
    # Strip surrounding quotes the model sometimes adds despite instructions.
    out = out.strip('"').strip("'").strip()
    if not out:
        log.warning("translate: model returned empty output")
        return ToolResult(
            text=t("translate.failed", lang),
            data={"text": text, "target_lang": target_lang, "error": "empty"},
        )
    return ToolResult(
        text=out,
        data={"text": text, "target_lang": target_lang, "translation": out},
    )
