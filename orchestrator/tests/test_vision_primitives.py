"""
vision_primitives — composable building blocks for computer_use.

Contracts we pin:

  • ``locate`` parses the multimodal LLM's STRICT-JSON reply into
    typed bbox dicts; non-list / malformed bbox entries are dropped.

  • ``click_text`` refuses destructive verbs in EN / RU / DE without
    even taking a screenshot (the regex blacklist is the cheap first
    gate; the navigation system prompt is the second).

  • ``click_text`` clicks the centre of the first returned bbox.

  • ``wait_for`` polls vision until a YES answer arrives, or returns
    False on timeout.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app import vision_primitives as vp


# ── 1. locate parses JSON ──────────────────────────────────────────────


async def test_locate_returns_parsed_bboxes():
    """JSON list of {label, bbox, confidence} → typed dicts.

    Malformed entries (missing bbox / wrong length) are silently
    dropped — callers expect a clean iterable.
    """
    canned = (
        '['
        '{"label": "Send", "bbox": [10, 20, 100, 30], "confidence": 0.95},'
        '{"label": "noBbox"},'
        '{"label": "bad", "bbox": [1, 2, 3]},'
        '{"label": "Inbox", "bbox": [200, 300, 60, 20], "confidence": 0.8}'
        ']'
    )
    with patch(
        "app.vision_primitives.desktop_client.screenshot",
        new=AsyncMock(return_value=b"\x89PNG\r\n\x1a\n"),
    ), patch(
        "app.vision_primitives.vision.analyze_image_bytes",
        new=AsyncMock(return_value=canned),
    ):
        out = await vp.locate("find buttons", agent_id=None)

    assert len(out) == 2
    assert out[0] == {
        "label": "Send",
        "bbox": [10, 20, 100, 30],
        "confidence": 0.95,
    }
    assert out[1]["label"] == "Inbox"


# ── 2. click_text destructive blacklist ────────────────────────────────


@pytest.mark.parametrize("bad_text", [
    "Delete", "delete", "Trash", "Archive",
    "Send", "Reply", "Forward",
    "Удалить", "Архивировать", "Переслать",
    "Löschen", "Antworten", "Senden",
])
async def test_click_text_refuses_destructive_keyword(bad_text):
    """Destructive verb (any of 3 languages) → refused, no vision call.

    Tests both that the regex matches and that no screenshot / vision
    call happens — the cheap first gate fires before any I/O.
    """
    with patch(
        "app.vision_primitives.desktop_client.screenshot",
        new=AsyncMock(side_effect=AssertionError("must not screenshot")),
    ), patch(
        "app.vision_primitives.desktop_client.run_pyautogui",
        new=AsyncMock(side_effect=AssertionError("must not click")),
    ):
        out = await vp.click_text(bad_text, agent_id=None)
    assert out["ok"] is False
    assert out["error"] == "destructive_action_refused"
    assert out["text"] == bad_text


# ── 3. click_text clicks centre of bbox ────────────────────────────────


async def test_click_text_clicks_center_of_bbox():
    """bbox=[x,y,w,h] → pyautogui.click(cx, cy) where (cx,cy) is centre.

    Centre for [100, 200, 50, 30] is (125, 215).
    """
    seen: list[dict] = []

    async def fake_pyautogui(payload, *, agent_id=None):
        seen.append(payload)
        return {"ok": True}

    canned = '[{"label": "OK", "bbox": [100, 200, 50, 30], "confidence": 1.0}]'
    with patch(
        "app.vision_primitives.desktop_client.screenshot",
        new=AsyncMock(return_value=b"\x89PNG\r\n\x1a\n"),
    ), patch(
        "app.vision_primitives.vision.analyze_image_bytes",
        new=AsyncMock(return_value=canned),
    ), patch(
        "app.vision_primitives.desktop_client.run_pyautogui",
        new=AsyncMock(side_effect=fake_pyautogui),
    ):
        out = await vp.click_text("OK", agent_id=None)

    assert out["ok"] is True
    assert out["clicked_at"] == [125, 215]
    assert seen == [{"action": "click", "x": 125, "y": 215}]


# ── 4. wait_for polls until match ──────────────────────────────────────


async def test_wait_for_polls_until_match_or_timeout():
    """First two polls reply NO, third replies YES → True.

    Tests the YES/NO answer path; anything that isn't a clean YES is
    treated as NO.
    """
    answers = iter(["NO", "no", "YES"])

    async def fake_vision(*args, **kwargs):
        return next(answers)

    with patch(
        "app.vision_primitives.desktop_client.screenshot",
        new=AsyncMock(return_value=b"\x89PNG\r\n\x1a\n"),
    ), patch(
        "app.vision_primitives.vision.analyze_image_bytes",
        new=AsyncMock(side_effect=fake_vision),
    ):
        result = await vp.wait_for(
            "is the inbox visible?",
            timeout=5.0,
            poll_interval=0.25,
            agent_id=None,
        )
    assert result is True


async def test_wait_for_returns_false_on_timeout():
    """Vision always says NO and timeout elapses → False.

    Use a tight timeout so the test stays fast.
    """
    with patch(
        "app.vision_primitives.desktop_client.screenshot",
        new=AsyncMock(return_value=b"\x89PNG\r\n\x1a\n"),
    ), patch(
        "app.vision_primitives.vision.analyze_image_bytes",
        new=AsyncMock(return_value="NO"),
    ):
        result = await vp.wait_for(
            "is the inbox visible?",
            timeout=0.3,
            poll_interval=0.25,
            agent_id=None,
        )
    assert result is False
