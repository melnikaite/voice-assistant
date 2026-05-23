"""
Multimodal vision via an OpenAI-compatible endpoint.

By default vision rides on the same endpoint as text (``LLM_URL``); for
deployments that want a separate provider for image input (e.g. local
text via Ollama + a multimodal endpoint elsewhere) the
``LLM_VISION_URL``/``LLM_VISION_MODEL`` env pair routes vision calls
independently.  See :mod:`app.llm_utils` for the env contract.

Two public entry points:

* :func:`analyze_image_bytes` — raw PNG/JPEG bytes + question → text.
  Used by the ``look_at_screen`` tool (host screenshot).

* :func:`analyze_image_b64`  — base64-string variant (with or without
  a ``data:image/...;base64,`` prefix).  Used by the pipeline's
  image-attached short-circuit when the user drops a file into the UI.

Token usage from every call is logged into ``token_usage`` via the
shared ``llm_utils.chat`` helper — so /api/stats covers vision spend
the same way it covers ordinary text turns.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

from . import llm_utils

log = logging.getLogger(__name__)


# Vision questions want a short voice-friendly answer.  Override per
# call when the question genuinely needs more (e.g. "transcribe this
# screenshot of a document" — caller passes max_tokens=4096).
_DEFAULT_MAX_TOKENS = 600


def _detect_media_type(data: bytes) -> str:
    """Sniff the image format from its magic bytes.

    OpenAI-compatible multimodal endpoints accept the media type as
    part of the data URL prefix; LM Studio is no exception.  Three
    formats cover what the frontend sends: PNG (screenshots from
    desktop-agent), JPEG (browser captures), GIF (rare).  Default
    to PNG when nothing matches.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def available() -> bool:
    """True iff vision is reachable.

    Today there's no separate liveness probe for the vision endpoint —
    it shares the LLM stack's health.  If ``LLM_VISION_URL`` is pointed
    at a different host, an in-flight call surfaces transport errors
    via the standard :func:`llm_utils.chat` path; callers degrade
    gracefully on those.
    """
    return True


async def analyze_image_bytes(
    image_data: bytes,
    question: str,
    *,
    client_id: str | None = None,
    tool_name: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Ask the local multimodal LLM a question about an image.

    Returns the model's text answer.  Raises ``httpx.HTTPError`` /
    ``httpx.TimeoutException`` on transport failure; callers wrap in
    a try/except and surface a localized message.

    Implementation: builds the OpenAI-style multimodal user message
    (a content array with one ``image_url`` and one ``text`` block)
    and routes it through the standard :func:`llm_utils.chat` helper
    so token usage is logged consistently with every other LLM call.
    """
    media_type = _detect_media_type(image_data)
    data_url = "data:" + media_type + ";base64," + base64.standard_b64encode(image_data).decode("ascii")
    sys_msg = system_prompt or (
        "You are a multilingual visual assistant.  Look at the image and "
        "answer the user's question concisely.  Reply in the same "
        "language the user used.  Keep the answer voice-friendly — 1-3 "
        "sentences, no markdown, no bullet lists, no URLs."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ],
        },
    ]
    log.info(
        "vision: media=%s bytes=%d question=%.60r",
        media_type, len(image_data), question,
    )
    choice = await llm_utils.chat(
        messages,
        temperature=0.2,
        max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
        # No tool catalog — vision turns are single-shot Q&A.
        tools=None,
        reasoning_effort="low",
        client_id=client_id,
        tool_name=tool_name or "vision",
        # Route to the vision endpoint.  Defaults equal to the text
        # endpoint (LLM_URL), so single-provider setups are unaffected.
        endpoint_url=llm_utils.LLM_VISION_URL,
        model=llm_utils.LLM_VISION_MODEL,
    )
    msg = choice.get("message") or {}
    text = llm_utils.extract_text(msg)
    return text.strip()


async def analyze_image_b64(
    image_b64: str,
    question: str,
    *,
    client_id: str | None = None,
    tool_name: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Convenience wrapper accepting base64-string input (e.g. data URLs).

    Strips a leading ``data:image/...;base64,`` prefix if present so the
    UI can pass through whatever ``FileReader.readAsDataURL`` produced.
    """
    if image_b64.startswith("data:"):
        _, _, payload = image_b64.partition(",")
        image_b64 = payload
    raw = base64.b64decode(image_b64)
    return await analyze_image_bytes(
        raw, question,
        client_id=client_id,
        tool_name=tool_name,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )
