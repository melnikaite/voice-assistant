"""
Composable vision primitives for the phase-2 vision-loop fallback.

These functions sit between the orchestrator's existing
:mod:`.vision` module (multimodal LLM via LM Studio) and
:mod:`.desktop_client` (HTTP gateway to the host agent).  They give
a higher-level driver a small, predictable vocabulary for
vision-driven UI navigation:

  • :func:`locate` — find named elements on screen.
  • :func:`click_text` — click visible text (with a destructive-text
    blacklist).
  • :func:`wait_for` — poll vision until a condition holds.
  • :func:`read_region` — OCR a screenshot region.

Wiring status
─────────────
NOT wired into any tool right now.  Wave 3's ``computer_use`` rework
went LLM-generated-AppleScript first (the model knows AppleScript
dictionaries directly).  These primitives are infrastructure for the
phase-2 vision-loop fallback that will kick in when the AppleScript
generator returns ``UNKNOWN`` — typically for goals that require
multi-step UI navigation a single script can't express.

Design choices (and the WHYs):

* The primitives DO NOT raise on transport / parse failure — they
  return ``None`` / empty list / ``False``.  The caller is always a
  higher-level intent that has its own fallback path; one failing
  vision call shouldn't take down the voice turn.

* Hard timeout = 10 s and ``max_tokens = 512`` per primitive — these
  are voice-loop helpers, not batch jobs.  A user repeating the
  question is faster than a 30 s blocking call.

* The destructive-text blacklist for :func:`click_text` covers the
  common mail / file / system destructive verbs in EN / RU / DE.
  False positives just refuse the click (the LLM can pick a
  different target); a false negative would let an LLM click
  "Delete" because the user asked something benign.

* The vision LLM is steered with a STRICT JSON system prompt that
  forbids navigation suggestions for destructive UI elements.  The
  blacklist is the second line of defence: even if a future model
  ignores the system prompt and returns ``label="Delete"``, the
  click_text check still refuses.

* All coordinates are display pixels with origin at top-left — same
  convention pyautogui uses.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from . import desktop_client, i18n, vision

log = logging.getLogger(__name__)


# Hard cap on any single vision call — these primitives are voice-loop
# helpers, not batch jobs, and a 10 s ceiling keeps an unresponsive
# vision model from stalling the user's turn.
_VISION_TIMEOUT_S = 10.0
# Vision-primitive responses are short JSON envelopes; we don't need
# a big budget.  Keeping it small also keeps inference fast.
_VISION_MAX_TOKENS = 512


# System prompt for the navigation-locator role.  EXPLICITLY forbids
# the model from helping the caller click anything that mutates state
# — even if the model's instinct would be to be helpful, the policy is
# baked into the prompt so a future model upgrade doesn't relax it.
# Click_text below ALSO enforces a regex blacklist as defence-in-depth.
_NAV_SYSTEM_PROMPT = (
    "You are providing READ-ONLY screen navigation guidance for an "
    "assistive tool. Locate UI elements precisely. NEVER suggest "
    "clicking buttons whose labels imply: delete, archive, move, "
    "forward, reply, send, mark as read, sign out, junk, spam, or any "
    "other state-changing or destructive action. If the user's goal "
    "would require such an action, return an empty array and add a "
    "note in your reasoning.\n\n"
    "Coordinates: return display coordinates (origin at top-left), in "
    "pixels.\n"
    "Format: STRICT JSON only, no prose."
)


# Destructive-text blacklist patterns live in every locale JSON under
# ``intents.destructive_text``.  When the LLM asks
# click_text(text="Delete") we refuse across ALL locales — a Russian-
# locale user can still see English destructive buttons on the screen,
# and vice versa.  Adding a new language = drop patterns into its JSON.


def is_destructive_text(text: str) -> bool:
    """Return True iff ``text`` matches the destructive-verb blacklist.

    Public so the higher-level intent handlers can pre-check labels
    coming from the LLM before they reach click_text — gives a clearer
    error message to the caller than "click_text refused" buried in
    a sub-step.  Patterns are loaded from every locale's JSON (under
    ``intents.destructive_text``) so the gate is language-agnostic.
    """
    if not text:
        return False
    for pat in i18n.patterns_for_intent("destructive_text"):
        if pat.search(text):
            return True
    return False


async def _take_screenshot(agent_id: str | None) -> bytes | None:
    """Best-effort screenshot.  Returns None on any failure."""
    try:
        return await desktop_client.screenshot(agent_id=agent_id)
    except desktop_client.DesktopUnavailable as exc:
        log.info("vision_primitives: screenshot unavailable: %s", exc)
        return None
    except Exception:
        log.exception("vision_primitives: screenshot crashed")
        return None


async def _ask_vision_json(
    png: bytes,
    question: str,
    *,
    system_prompt: str = _NAV_SYSTEM_PROMPT,
    client_id: str | None = None,
    tool_name: str = "vision_primitives",
) -> Any | None:
    """Call the multimodal LLM with JSON-only steering; return parsed JSON.

    Returns None on transport failure, parse failure, or model returning
    non-JSON prose.  The model COULD wrap its reply in a ```json fence
    even with strict instructions — we strip those before parsing.
    """
    try:
        raw = await asyncio.wait_for(
            vision.analyze_image_bytes(
                png, question,
                client_id=client_id,
                tool_name=tool_name,
                system_prompt=system_prompt,
                max_tokens=_VISION_MAX_TOKENS,
            ),
            timeout=_VISION_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, Exception):
        log.exception("vision_primitives: vision call failed")
        return None
    text = (raw or "").strip()
    if not text:
        return None
    # Strip ```json ... ``` fences if the model wrapped its reply.
    if text.startswith("```"):
        # Drop the first line (```lang) and trailing ```.
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.debug("vision_primitives: non-JSON reply: %r", text[:120])
        return None


async def locate(
    question: str,
    *,
    agent_id: str | None = None,
    client_id: str | None = None,
) -> list[dict]:
    """Find UI elements matching ``question``; return parsed bboxes.

    Each result dict: ``{label: str, bbox: [x, y, w, h], confidence: float}``.
    Empty list on any failure (no screenshot, no vision, no parse).

    Coordinate convention: display pixels, origin at top-left.
    """
    png = await _take_screenshot(agent_id)
    if png is None:
        return []
    prompt = (
        "Return a JSON array of {label, bbox: [x, y, w, h], "
        f"confidence: 0..1}} for elements matching: {question}. "
        "Use display coordinates in pixels (origin at top-left). "
        "If nothing matches, return []."
    )
    parsed = await _ask_vision_json(png, prompt, client_id=client_id)
    if not isinstance(parsed, list):
        return []
    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        except (TypeError, ValueError):
            continue
        out.append({
            "label": str(item.get("label") or ""),
            "bbox": [x, y, w, h],
            "confidence": float(item.get("confidence") or 0.0),
        })
    return out


async def click_text(
    text: str,
    *,
    agent_id: str | None = None,
    client_id: str | None = None,
) -> dict:
    """Find ``text`` on screen, click its centre — refuses destructive text.

    Returns ``{ok: bool, clicked_at?: [x, y], error?: str}``.  The
    destructive-text check is BLACKLIST-FIRST: we don't take a
    screenshot or call vision when the requested text matches a
    destructive verb — pointless to spend tokens on a request we'd
    refuse anyway.
    """
    if is_destructive_text(text):
        log.info("click_text: refused destructive text=%r", text)
        return {"ok": False, "error": "destructive_action_refused", "text": text}
    if not text:
        return {"ok": False, "error": "empty_text"}
    bboxes = await locate(
        f'visible text exactly equal to "{text}"',
        agent_id=agent_id, client_id=client_id,
    )
    if not bboxes:
        return {"ok": False, "error": "not_found", "text": text}
    first = bboxes[0]
    x, y, w, h = first["bbox"]
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    try:
        await desktop_client.run_pyautogui(
            {"action": "click", "x": cx, "y": cy},
            agent_id=agent_id,
        )
    except desktop_client.DesktopUnavailable as exc:
        return {"ok": False, "error": f"pyautogui: {exc}"}
    return {"ok": True, "clicked_at": [cx, cy], "label": first.get("label")}


async def wait_for(
    question: str,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.5,
    agent_id: str | None = None,
    client_id: str | None = None,
) -> bool:
    """Poll screenshot+vision until ``question`` is answered YES, or timeout.

    The vision prompt is steered to reply with a single token —
    ``YES`` or ``NO`` — so we don't burn tokens on prose.  Anything
    that isn't a clean YES is treated as NO (better to retry than to
    succeed on a misread).  ``poll_interval`` clamps to a minimum of
    0.25 s — a tighter loop just wastes vision calls.
    """
    poll_interval = max(0.25, float(poll_interval))
    deadline = time.monotonic() + max(0.1, float(timeout))
    yn_system = (
        "You answer one of two strings exactly: YES or NO. "
        "No punctuation, no explanation."
    )
    while time.monotonic() < deadline:
        png = await _take_screenshot(agent_id)
        if png is not None:
            try:
                raw = await asyncio.wait_for(
                    vision.analyze_image_bytes(
                        png, question,
                        client_id=client_id,
                        tool_name="vision_primitives_wait_for",
                        system_prompt=yn_system,
                        max_tokens=8,
                    ),
                    timeout=_VISION_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, Exception):
                log.debug("wait_for: vision call failed", exc_info=True)
                raw = ""
            if (raw or "").strip().upper().startswith("YES"):
                return True
        await asyncio.sleep(poll_interval)
    return False


async def read_region(
    bbox: list[int],
    *,
    agent_id: str | None = None,
    client_id: str | None = None,
) -> str:
    """Crop a screenshot to ``bbox``, OCR via vision LLM, return the text.

    ``bbox`` is ``[x, y, w, h]`` in display pixels.  Returns an empty
    string on any failure.  We crop in Pillow rather than passing the
    full screenshot + "look at this region" — the smaller image fits
    in the model's effective resolution better and the call is faster.
    """
    if not bbox or len(bbox) != 4:
        return ""
    png = await _take_screenshot(agent_id)
    if png is None:
        return ""
    try:
        from io import BytesIO
        from PIL import Image  # Pillow is already a vision.py transitive dep
        img = Image.open(BytesIO(png))
        x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        crop = img.crop((x, y, x + w, y + h))
        out = BytesIO()
        crop.save(out, format="PNG")
        cropped_png = out.getvalue()
    except Exception:
        log.exception("read_region: crop failed")
        return ""
    try:
        text = await asyncio.wait_for(
            vision.analyze_image_bytes(
                cropped_png,
                "Return only the text visible in this image.  No commentary.",
                client_id=client_id,
                tool_name="vision_primitives_read_region",
                system_prompt=(
                    "You read text from images. Return only what is "
                    "visible verbatim. No prose, no labels."
                ),
                max_tokens=_VISION_MAX_TOKENS,
            ),
            timeout=_VISION_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, Exception):
        log.debug("read_region: vision call failed", exc_info=True)
        return ""
    return (text or "").strip()
