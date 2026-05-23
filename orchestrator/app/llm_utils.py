"""Shared LLM helpers — used by both the top-level dispatcher and individual tools."""
import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator, NamedTuple

import httpx

log = logging.getLogger(__name__)

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]

# Optional endpoint split.  Defaults preserve the single-provider setup
# (text and vision both hit LLM_URL with LLM_MODEL).  Set the pair below
# only if you want to point text and vision at DIFFERENT providers —
# e.g. Ollama for text + a separate multimodal endpoint for vision, or
# a cloud vision model alongside a local text LLM.  Same OpenAI-style
# `/v1/chat/completions` contract on both sides.
LLM_TEXT_URL = os.environ.get("LLM_TEXT_URL", LLM_URL)
LLM_TEXT_MODEL = os.environ.get("LLM_TEXT_MODEL", LLM_MODEL)
LLM_VISION_URL = os.environ.get("LLM_VISION_URL", LLM_URL)
LLM_VISION_MODEL = os.environ.get("LLM_VISION_MODEL", LLM_MODEL)

# Default = full Gemma 4 E4B context (131072 tokens). Per-deployment overrides
# go through LLM_MAX_TOKENS env var. Since input prompts are tiny, the model
# stops on EOS long before hitting this — it's just an upper bound.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "131072"))

# Per-request HTTP timeout (seconds). 90s is enough for ~2000 output tokens
# at typical M-chip speeds; if the model doesn't finish, we fall back.
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "90"))


class ParsedToolCall(NamedTuple):
    id: str
    name: str
    args: dict


def parse_tool_calls(message: dict) -> list[ParsedToolCall]:
    """
    Pull tool invocations out of an OpenAI-style assistant message.

    Each call comes with its `id` (needed for the matching `role: "tool"`
    response in multi-step agent loops), function name, and pre-parsed args.
    JSON decode errors fall back to an empty dict — the caller decides how
    to handle a malformed argument blob.
    """
    out: list[ParsedToolCall] = []
    for tc in (message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if not name:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            log.warning("parse_tool_calls: malformed args for %r: %r", name, raw_args[:200])
            args = {}
        out.append(ParsedToolCall(id=tc.get("id") or "", name=name, args=args))
    return out


def extract_text(message: dict) -> str:
    """
    Extract the final answer from an OpenAI-style assistant message.

    Some models (Gemma 3n E4B in LM Studio, DeepSeek-R1, etc.) split output:
    `reasoning_content` holds CoT, `content` holds the actual reply. If the
    model spent its budget thinking, `content` may be empty — fall back to
    the last paragraph of reasoning so the user hears *something* coherent.
    """
    content = (message.get("content") or "").strip()
    if content:
        return content
    reasoning = (message.get("reasoning_content") or "").strip()
    if not reasoning:
        return ""
    paragraphs = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1]
    lines = [ln.strip() for ln in reasoning.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def _fire_record_usage(
    *,
    usage: dict | None,
    model: str,
    client_id: str | None,
    tool_name: str | None,
    elapsed_ms: int,
) -> None:
    """Schedule an async DB insert of one LLM call's usage row.

    Wrapped in a try/except so an empty/missing usage block — or any DB
    glitch — never propagates back into the LLM call path.  Uses
    ``create_task`` so the insert (one sqlite write, ~ms) runs concurrent
    with whatever the caller does next instead of blocking the return.
    """
    if not usage:
        return
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return
    if prompt == 0 and completion == 0:
        return
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    try:
        # Late import: avoids a circular dep at module load (storage package
        # pulls in token_usage which has its own logger setup).
        from .storage import add_token_usage

        asyncio.create_task(
            add_token_usage(
                client_id=client_id,
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
                reasoning_tokens=int(reasoning) if reasoning is not None else None,
                tool_name=tool_name,
                elapsed_ms=elapsed_ms,
            )
        )
    except Exception as exc:  # pragma: no cover — DB write should never fail loud
        log.warning("token usage logging failed: %s", exc)


async def chat(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: "str | dict | None" = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_id: str | None = None,
    tool_name: str | None = None,
    endpoint_url: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Single point for OpenAI-style /v1/chat/completions calls.

    Returns the raw `choice` dict (with `message`, `finish_reason`, etc.).
    Raises httpx.* errors on failure — callers should catch and degrade gracefully.

    Logs `finish_reason` and token usage. Warns loudly if max_tokens was hit:
    that's the situation where the model "thought too long" and content is empty.

    ``client_id`` and ``tool_name`` are observability hooks: every successful
    call is logged to the ``token_usage`` table for the /api/stats dashboard.
    Both are optional for back-compat (e.g. tests, ad-hoc scripts) — when
    missing the row is still recorded, just attributed to "(anon)" / "(unknown)".

    ``endpoint_url`` and ``model`` let callers route a specific call to a
    different provider — used by :mod:`app.vision` to hit ``LLM_VISION_URL``
    when the deployment splits text and vision across endpoints.  Defaults
    to the text endpoint configured via ``LLM_TEXT_URL``/``LLM_TEXT_MODEL``.
    """
    url = endpoint_url or LLM_TEXT_URL
    mdl = model or LLM_TEXT_MODEL
    body: dict = {
        "model": mdl,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    if reasoning_effort is not None:
        # "low" | "medium" | "high" — controls how many tokens the model spends
        # in chain-of-thought before producing the final answer.
        body["reasoning_effort"] = reasoning_effort

    t_o = timeout if timeout is not None else LLM_TIMEOUT
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=t_o) as c:
        r = await c.post(f"{url}/v1/chat/completions", json=body)
    r.raise_for_status()
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data = r.json()
    choice = data["choices"][0]
    finish = choice.get("finish_reason")
    usage = data.get("usage") or {}
    completion = usage.get("completion_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if finish == "length":
        log.warning(
            "LLM hit max_tokens (%s used, reasoning=%s) — answer may be incomplete",
            completion,
            reasoning_tokens,
        )
    else:
        log.info(
            "LLM finish=%s, completion_tokens=%s (reasoning=%s)",
            finish,
            completion,
            reasoning_tokens,
        )
    _fire_record_usage(
        usage=usage,
        model=data.get("model") or mdl,
        client_id=client_id,
        tool_name=tool_name,
        elapsed_ms=elapsed_ms,
    )
    return choice


async def chat_stream(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    client_id: str | None = None,
    tool_name: str | None = None,
) -> AsyncIterator[str]:
    """Streaming version of :func:`chat` for plain-text responses.

    Yields successive text fragments as the LLM produces them.  Used by
    `web_search` to feed sentences into TTS the moment they're available
    instead of waiting for the full summary.  Time-to-first-audio drops
    from ~5 s (wait for full LLM) to ~1.5-2 s (first sentence) on a
    typical 4-sentence answer.

    Limitations vs `chat()`:
      • No tool_calls — this is for pure-text responses only.
      • No reasoning_content fallback — if the model returns all CoT and
        no content, the iterator just ends empty.  Caller should treat
        an empty stream as an error and degrade gracefully.

    Token-usage capture: ``stream_options.include_usage = true`` makes
    LM Studio emit a final SSE chunk whose ``usage`` block carries the
    same prompt/completion totals as the non-streaming endpoint, so the
    /api/stats dashboard sees streamed calls identically.

    Raises ``httpx.HTTPError`` on transport / server errors.
    """
    body: dict = {
        "model": LLM_TEXT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
        "stream": True,
        # LM Studio supports OpenAI's stream_options.include_usage flag —
        # when set, the second-to-last SSE chunk (right before [DONE])
        # carries a `usage` block at the top level with prompt/completion
        # totals.  Without this flag streamed calls give us no usage info.
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort

    t_o = timeout if timeout is not None else LLM_TIMEOUT
    t0 = time.monotonic()
    captured_usage: dict | None = None
    captured_model: str = LLM_TEXT_MODEL
    try:
        async with httpx.AsyncClient(timeout=t_o) as c:
            async with c.stream(
                "POST",
                f"{LLM_TEXT_URL}/v1/chat/completions",
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    # OpenAI SSE format: each event is `data: {...}` plus a
                    # blank line.  Final event is `data: [DONE]`.  Anything
                    # else (comments, blank lines) is ignored.
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        log.debug("chat_stream: skipping unparseable %r", payload[:120])
                        continue
                    # The final-usage chunk arrives with `choices: []` and a
                    # top-level `usage`.  Capture it regardless of where it
                    # shows up in the stream (LM Studio: just before [DONE]).
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        captured_usage = chunk_usage
                        captured_model = chunk.get("model") or captured_model
                    try:
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                    except (KeyError, IndexError, TypeError):
                        continue
                    content = delta.get("content")
                    if content:
                        yield content
    finally:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _fire_record_usage(
            usage=captured_usage,
            model=captured_model,
            client_id=client_id,
            tool_name=tool_name,
            elapsed_ms=elapsed_ms,
        )
