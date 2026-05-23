"""
general_answer — default conversational tool.

Internally uses a forced-tool inner LLM call so the model returns a
structured pair (answer, confident). When ``confident=False`` we surface
``ToolResult.data["unknown"] = True`` and mark this tool ``terminal=False``,
which lets the agent loop in agent.py see the result, notice the signal,
and chain into ``web_search``.

We never parse natural-language "I don't know" strings — confidence is
purely structural via the inner tool call.
"""
from __future__ import annotations

import logging

import httpx

from ..i18n import t
from ..llm_utils import chat, extract_text, parse_tool_calls
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "Your reply in the user's language, 2–4 conversational sentences, "
                "no markdown / lists / formulas / quotes. Will be read aloud."
            ),
        },
        "confident": {
            "type": "boolean",
            "description": (
                "True ONLY if the answer is grounded in stable, well-known "
                "facts you actually remember. False whenever you'd be guessing, "
                "the answer requires fresh data (news, prices, weather, recent "
                "releases, current officeholders, today's events), or you're "
                "unsure about a specific person, company, version or number. "
                "Setting this to false redirects the question to web search — "
                "this is preferable to hedging with vague phrases like 'maybe', "
                "'probably', 'depends', 'try to clarify'."
            ),
        },
    },
    "required": ["answer", "confident"],
}


_SYSTEM_PROMPT = (
    "You are an erudite voice assistant. Think carefully before answering — "
    "include concrete facts, names, numbers and dates when relevant.\n"
    "Reply in the user's language; honour an explicit request to switch.\n"
    "2–4 lively sentences, conversational tone. No markdown, no bullet "
    "lists, no greetings, no 'as an AI I cannot' disclaimers. No LaTeX, "
    "no `$` or backslashes. Spell formulas in words (say 'energy equals "
    "mass times the speed of light squared', not 'E=mc²'). The text "
    "is read aloud — write for the ear.\n"
    "\n"
    "You MUST reply by calling the `respond_with_confidence` tool. Two fields:\n"
    "  • `answer`     — the reply text (rules above).\n"
    "  • `confident`  — boolean. See its schema description carefully.\n"
    "\n"
    "If you have to pad an answer with 'maybe', 'probably', 'it depends': "
    "that's the moment to set `confident=false` instead. The user prefers a "
    "fresh search result over a vague guess. When you set confident=false, "
    "leave `answer` empty (or one short sentence at most) — it will be "
    "replaced by the web-search summary anyway."
)


@tool(
    name="general_answer",
    description=(
        "Answer a general question conversationally. Use as default when no "
        "specific tool fits but the user clearly asked something. If you are "
        "not confident, this tool will signal so and the agent loop will "
        "retry with `web_search` automatically — pick this over `web_search` "
        "whenever the answer might be in stable training knowledge."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The user's question, verbatim"},
        },
        "required": ["question"],
    },
    # Terminal by default: the happy-path answer is the user's reply.
    # When this tool signals `unknown=True` in its result data, the agent
    # loop dynamically downgrades it to non-terminal so the model can
    # chain into web_search — see agent._execute_one().
    risk="read",
)
async def general_answer(question: str, *, ctx=None) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang
    await cx.progress("thinking", None)
    try:
        choice = await chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
            # Was "high" — for "what's the capital of Germany"-style timeless
            # facts the model knows immediately, deep reasoning just burns
            # tokens.  Deep / nuanced queries naturally route via web_search
            # anyway (it signals `unknown=True` and the agent chains).  Bump
            # back to "medium" if you start seeing wrong answers on common
            # knowledge that the model should just know.
            reasoning_effort="low",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "respond_with_confidence",
                        "description": (
                            "Return your answer along with an explicit "
                            "confidence flag."
                        ),
                        "parameters": _REPLY_SCHEMA,
                    },
                }
            ],
            # LM Studio rejects dict tool_choice ({"type":"function",...}) —
            # only accepts "none" | "auto" | "required". Registering exactly
            # one tool above and asking for "required" gives the same effect.
            tool_choice="required",
            client_id=cx.client_id,
            tool_name="general_answer",
        )
    except httpx.TimeoutException:
        log.warning("general_answer: timeout")
        return ToolResult(
            text=t("answer.thought_too_long", lang),
            data={"question": question, "error": "timeout"},
        )
    except httpx.HTTPError as e:
        log.warning("general_answer: http error: %s", e)
        return ToolResult(
            text=t("answer.model_unreachable", lang),
            data={"question": question, "error": f"{e.__class__.__name__}"},
        )

    msg = choice["message"]
    tool_calls = parse_tool_calls(msg)

    # Happy path: model followed `tool_choice="required"` and we have args.
    if tool_calls:
        args = tool_calls[0].args
        confident = bool(args.get("confident", True))
        answer = (args.get("answer") or "").strip()
        log.info(
            "general_answer: confident=%s, answer_chars=%d", confident, len(answer)
        )
        if not confident:
            # Agent loop sees `unknown=True`, asks the LLM to retry with
            # web_search. Our `answer` text is discarded in that case.
            return ToolResult(
                text=answer, data={"question": question, "unknown": True}
            )
        if not answer:
            return ToolResult(
                text="",
                data={"question": question, "unknown": True, "reason": "empty_answer"},
            )
        return ToolResult(text=answer, data={"question": question, "confident": True})

    # Fallback: model ignored tool_choice. Use whatever free text it emitted,
    # mark unknown if empty so the agent loop tries a different tool.
    out = extract_text(msg)
    if not out:
        if choice.get("finish_reason") == "length":
            log.warning("general_answer: hit max_tokens, no usable text")
            return ToolResult(
                text=t("answer.hit_max_tokens", lang),
                data={"question": question, "error": "length"},
            )
        log.warning("general_answer: model returned neither tool_call nor text")
        return ToolResult(
            text="",
            data={"question": question, "unknown": True, "reason": "no_output"},
        )
    log.warning(
        "general_answer: model bypassed tool_choice, used free text (%d chars)",
        len(out),
    )
    return ToolResult(text=out, data={"question": question, "confident": True})
