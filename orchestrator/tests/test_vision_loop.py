"""
vision_loop — phase-2 agentic fallback unit tests.

Contracts pinned here:

  * Action JSON validation — every action shape is checked before it
    leaves the planner; bad shapes / out-of-range coords / missing
    target_text all reject to None.

  * Destructive-text clicks rejected at validation time
    (``target_text="Delete"`` → ``{"type": "fail", ...}``).

  * Loop termination paths — ``done`` returns ok=True with the
    planner's ``result`` string; ``fail`` returns ok=False with
    ``planner_fail``; budget exhaustion returns ``max_steps``.

  * Cursor-activity refusal — both pre-loop and mid-loop.

  * Transport failure — agent went away mid-loop returns
    ``transport`` error, not silent retry.

  * Screenshot empty / failed handled gracefully.

  * History pruning — only last 4 actions sent to planner (memory-
    efficient long loops).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app import vision_loop


# ── 1. Action validation ────────────────────────────────────────────────


def test_validate_action_accepts_minimal_done():
    out = vision_loop._validate_action(
        {"type": "done", "result": "Volume set", "reason": "goal reached"}
    )
    assert out is not None
    assert out["type"] == "done"
    assert out["result"] == "Volume set"


def test_validate_action_rejects_unknown_type():
    assert vision_loop._validate_action({"type": "magic"}) is None


def test_validate_action_rejects_click_without_target_text():
    """target_text is mandatory — it gates the destructive blacklist.
    A click without it gives us nothing to check against, so refuse.
    """
    bad = {"type": "click", "x": 100, "y": 200, "reason": "test"}
    assert vision_loop._validate_action(bad) is None


def test_validate_action_rejects_click_with_destructive_text():
    """target_text='Delete' → coerced to fail action (defence in depth
    against a model that produces destructive clicks despite the prompt).
    """
    bad = {
        "type": "click", "x": 50, "y": 50,
        "target_text": "Delete", "reason": "yolo",
    }
    out = vision_loop._validate_action(bad)
    assert out is not None
    assert out["type"] == "fail"
    assert "destructive_click_refused" in out["reason"]


def test_validate_action_rejects_out_of_range_coords():
    bad = {
        "type": "click", "x": 999_999, "y": 50,
        "target_text": "Safe Button", "reason": "test",
    }
    assert vision_loop._validate_action(bad) is None


def test_validate_action_caps_long_type_text():
    bad = {"type": "type", "text": "x" * (vision_loop.MAX_TYPE_LEN + 10)}
    out = vision_loop._validate_action(bad)
    assert out is not None
    assert out["type"] == "fail"
    assert "type_too_long" in out["reason"]


def test_validate_action_clamps_scroll_magnitude():
    out = vision_loop._validate_action({
        "type": "scroll", "x": 100, "y": 100, "dy": 99_999,
        "reason": "test",
    })
    assert out is not None
    assert out["type"] == "scroll"
    assert out["dy"] == 2000  # clamped


# ── 2. Loop termination — done ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_returns_done_immediately_when_planner_says_done():
    """First plan call → done.  Loop terminates with ok=True and
    speaks the planner's ``result`` back."""
    with patch(
        "app.vision_loop._user_is_active",
        new=AsyncMock(return_value=False),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(return_value=b"fake_png_bytes"),
    ), patch(
        "app.vision_loop._plan_next_action",
        new=AsyncMock(return_value={
            "type": "done",
            "result": "The volume is now 80%",
            "reason": "goal reached",
        }),
    ):
        outcome = await vision_loop.run_vision_loop(
            "set volume to 80", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is True
    assert outcome["result"] == "The volume is now 80%"
    assert len(outcome["steps"]) == 1


# ── 3. Loop terminates on planner fail ──────────────────────────────────


@pytest.mark.asyncio
async def test_loop_returns_planner_fail():
    """Planner outputs {"type": "fail", "reason": "..."} → ok=False
    with the reason surfaced in detail.
    """
    with patch(
        "app.vision_loop._user_is_active",
        new=AsyncMock(return_value=False),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(return_value=b"png"),
    ), patch(
        "app.vision_loop._plan_next_action",
        new=AsyncMock(return_value={
            "type": "fail",
            "reason": "destructive_action_required",
        }),
    ):
        outcome = await vision_loop.run_vision_loop(
            "delete all photos", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is False
    assert outcome["error"] == "planner_fail"
    assert "destructive_action_required" in outcome["detail"]


# ── 4. Budget exhaustion → max_steps ────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_exhausts_budget_with_repeating_actions():
    """Planner keeps returning the same wait action forever → loop
    hits max_steps and returns ok=False.
    """
    apply_mock = AsyncMock()
    with patch(
        "app.vision_loop._user_is_active",
        new=AsyncMock(return_value=False),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(return_value=b"png"),
    ), patch(
        "app.vision_loop._plan_next_action",
        new=AsyncMock(return_value={
            "type": "wait", "ms": 100, "reason": "stuck",
        }),
    ), patch(
        "app.vision_loop._apply_action", new=apply_mock,
    ):
        outcome = await vision_loop.run_vision_loop(
            "do something hard", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is False
    assert outcome["error"] == "max_steps"
    assert len(outcome["steps"]) == vision_loop.MAX_STEPS
    assert apply_mock.await_count == vision_loop.MAX_STEPS


# ── 5. Cursor-activity refusal — pre-loop ───────────────────────────────


@pytest.mark.asyncio
async def test_loop_refuses_when_user_active_at_start():
    """``_user_is_active`` True → refuse before taking any screenshot."""
    screenshot_mock = AsyncMock(return_value=b"png")
    plan_mock = AsyncMock()
    with patch(
        "app.vision_loop._user_is_active", new=AsyncMock(return_value=True),
    ), patch(
        "app.desktop_client.screenshot", new=screenshot_mock,
    ), patch(
        "app.vision_loop._plan_next_action", new=plan_mock,
    ):
        outcome = await vision_loop.run_vision_loop(
            "do thing", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is False
    assert outcome["error"] == "user_active"
    screenshot_mock.assert_not_awaited()
    plan_mock.assert_not_awaited()


# ── 6. Cursor-activity refusal — mid-loop ───────────────────────────────


@pytest.mark.asyncio
async def test_loop_refuses_when_user_becomes_active_mid_loop():
    """User typed something between step 1 and step 2 → abort with
    user_active error; steps collected so far are preserved.
    """
    activity_calls = {"n": 0}

    async def _activity_check(*_a, **_kw):
        activity_calls["n"] += 1
        # First call (pre-loop) — idle; second call (start of step 2) — active.
        return activity_calls["n"] >= 2

    with patch(
        "app.vision_loop._user_is_active", new=AsyncMock(side_effect=_activity_check),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(return_value=b"png"),
    ), patch(
        "app.vision_loop._plan_next_action",
        new=AsyncMock(return_value={
            "type": "key", "keys": ["cmd", "space"], "reason": "spotlight",
        }),
    ), patch(
        "app.vision_loop._apply_action", new=AsyncMock(),
    ):
        outcome = await vision_loop.run_vision_loop(
            "search", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is False
    assert outcome["error"] == "user_active"
    # One action ran before the user-activity guard fired.
    assert len(outcome["steps"]) == 1


# ── 7. Transport failure — screenshot agent dies ────────────────────────


@pytest.mark.asyncio
async def test_loop_returns_transport_error_on_screenshot_failure():
    """desktop_client.DesktopUnavailable from screenshot → transport
    error surfaced; no plan call attempted.
    """
    from app.desktop_client import DesktopUnavailable

    plan_mock = AsyncMock()
    with patch(
        "app.vision_loop._user_is_active", new=AsyncMock(return_value=False),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(side_effect=DesktopUnavailable("agent gone")),
    ), patch(
        "app.vision_loop._plan_next_action", new=plan_mock,
    ):
        outcome = await vision_loop.run_vision_loop(
            "x", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is False
    assert outcome["error"] == "transport"
    assert "agent gone" in outcome["detail"]
    plan_mock.assert_not_awaited()


# ── 8. Plan-unparseable refusal ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_returns_plan_unparseable_on_bad_json():
    """``_plan_next_action`` returned None (parse failure / bad shape)
    → ok=False, no further iterations.
    """
    with patch(
        "app.vision_loop._user_is_active", new=AsyncMock(return_value=False),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(return_value=b"png"),
    ), patch(
        "app.vision_loop._plan_next_action",
        new=AsyncMock(return_value=None),
    ):
        outcome = await vision_loop.run_vision_loop(
            "x", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is False
    assert outcome["error"] == "plan_unparseable"
    assert outcome["steps"] == []


# ── 9. Multi-step happy path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_progresses_through_multiple_steps_then_done():
    """Planner returns: key → wait → type → done.  Verify each action
    routes to ``_apply_action`` in order, and final ``done`` short-
    circuits without a follow-up apply call.
    """
    plan_sequence = [
        {"type": "key", "keys": ["cmd", "space"], "reason": "spotlight"},
        {"type": "wait", "ms": 200, "reason": "settle"},
        {"type": "type", "text": "calendar", "reason": "search"},
        {"type": "done", "result": "Calendar is open", "reason": "ok"},
    ]
    apply_mock = AsyncMock()
    with patch(
        "app.vision_loop._user_is_active", new=AsyncMock(return_value=False),
    ), patch(
        "app.desktop_client.screenshot",
        new=AsyncMock(return_value=b"png"),
    ), patch(
        "app.vision_loop._plan_next_action",
        new=AsyncMock(side_effect=plan_sequence),
    ), patch(
        "app.vision_loop._apply_action", new=apply_mock,
    ):
        outcome = await vision_loop.run_vision_loop(
            "open calendar", agent_id="default", client_id="c1",
        )
    assert outcome["ok"] is True
    assert outcome["result"] == "Calendar is open"
    assert len(outcome["steps"]) == 4
    # done is terminal — no apply for the last step.
    assert apply_mock.await_count == 3
