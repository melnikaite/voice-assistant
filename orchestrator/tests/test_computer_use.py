"""
computer_use — LLM-generated AppleScript driver unit tests.

Contracts pinned here:

  * Voice ID gate (``profile_id`` required) — no identified speaker
    → ``computer_use.auth_required``, no LLM call.

  * Desktop-agent reachability — unreachable agent → ``desktop.unreachable``
    refusal before any LLM call.

  * LLM generation returns ``UNKNOWN`` → ``computer_use.cannot_express``;
    no AppleScript execution.

  * Read-only verb gate (defence in depth against a creative generator):
    ``delete``/``send``/etc. → ``computer_use.readonly_violation`` even
    if the generator emitted it.

  * Risk upgrade gate: a ``set name`` / ``new event`` style script that
    the strict gate doesn't reject but the destructive-pattern detector
    upgrades to ``high_write`` → also refused.

  * Happy path: generator emits a safe one-liner → executes via
    desktop_client.run_applescript and returns stdout (or
    ``computer_use.done`` for empty stdout).

  * AppleScript runtime error → ``computer_use.execution_failed``,
    stderr surfaced in ``ToolResult.data``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.tools import dispatch


def _seed_default_reachable():
    """Mark the registry's default agent reachable for tests.

    The tool fast-fails on ``is_reachable()`` BEFORE any LLM work;
    without this every test would short-circuit to ``unreachable``.
    """
    from app import desktop_client
    info = desktop_client.get_agent(None)
    if info is None:
        desktop_client.reload_agents_from_env()
        info = desktop_client.get_agent(None)
    assert info is not None
    info.reachable = True


def _mock_chat_response(content: str) -> dict:
    """Build a fake LM Studio /v1/chat/completions choice payload."""
    return {
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }


# ── 1. auth gate ─────────────────────────────────────────────────────────


async def test_computer_use_requires_profile(make_agent_ctx):
    """No profile_id → tool refuses BEFORE any LLM/agent call."""
    from app.i18n import t

    ctx = make_agent_ctx(profile_id=None, client_id="cli", user_lang="ru")
    # If the auth gate works, llm_utils.chat is never called.
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(side_effect=AssertionError("auth must short-circuit")),
    ):
        result = await dispatch(
            "computer_use", {"goal": "set volume to 80"}, ctx=ctx,
        )
    assert result.text == t("computer_use.auth_required", "ru")
    assert (result.data or {}).get("error") == "auth_required"


# ── 2. unreachable agent ────────────────────────────────────────────────


async def test_computer_use_refuses_when_agent_unreachable(make_agent_ctx):
    """Reachable=False → refuse with desktop.unreachable, no LLM call."""
    from app import desktop_client
    from app.i18n import t

    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    info = desktop_client.get_agent(None)
    if info is None:
        desktop_client.reload_agents_from_env()
        info = desktop_client.get_agent(None)
    info.reachable = False  # explicit: not reachable

    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(side_effect=AssertionError("must short-circuit")),
    ):
        result = await dispatch(
            "computer_use", {"goal": "set volume to 80"}, ctx=ctx,
        )
    assert result.text == t("desktop.unreachable", "en")
    assert (result.data or {}).get("error") == "desktop_unreachable"


# ── 3. generator returns UNKNOWN → falls to vision_loop ──────────────────


async def test_computer_use_falls_to_vision_loop_on_unknown(make_agent_ctx):
    """Generator UNKNOWN → vision_loop is invoked, its result speaks back.

    Verifies the phase-2 fallback wiring: ``_llm_generate_applescript``
    returns None (UNKNOWN), ``computer_use`` invokes
    :func:`vision_loop.run_vision_loop`, the loop's ``result`` becomes
    the spoken reply.
    """
    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    loop_outcome = {
        "ok": True,
        "result": "Calendar is open with today's events",
        "steps": [
            {"type": "key", "keys": ["cmd", "space"], "reason": "spotlight"},
            {"type": "type", "text": "calendar", "reason": "search"},
            {"type": "done", "result": "Calendar is open with today's events",
             "reason": "goal reached"},
        ],
    }
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response("UNKNOWN")),
    ), patch(
        "app.desktop_client.run_applescript",
        new=AsyncMock(side_effect=AssertionError("AppleScript path must not run")),
    ), patch(
        "app.vision_loop.run_vision_loop",
        new=AsyncMock(return_value=loop_outcome),
    ):
        result = await dispatch(
            "computer_use",
            {"goal": "open calendar and show today's events"},
            ctx=ctx,
        )
    assert result.text == "Calendar is open with today's events"
    data = result.data or {}
    assert data.get("path") == "vision_loop"
    assert len(data.get("steps") or []) == 3


async def test_computer_use_surfaces_vision_loop_user_active_refusal(make_agent_ctx):
    """vision_loop returns ``user_active`` → spoken refusal mapped to
    ``computer_use.user_busy``.
    """
    from app.i18n import t

    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response("UNKNOWN")),
    ), patch(
        "app.vision_loop.run_vision_loop",
        new=AsyncMock(return_value={"ok": False, "error": "user_active", "steps": []}),
    ):
        result = await dispatch(
            "computer_use", {"goal": "click something"}, ctx=ctx,
        )
    assert result.text == t("computer_use.user_busy", "en")
    assert (result.data or {}).get("error") == "user_active"


# ── 4. strict gate refuses destructive verbs ────────────────────────────


async def test_computer_use_refuses_destructive_script(make_agent_ctx):
    """Generator emitted a delete verb → refuse via the strict gate."""
    from app.i18n import t

    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    # The forbidden-substring set in desktop._FORBIDDEN_IN_READONLY
    # includes "delete" — must trip the strict reject.
    sneaky_script = 'tell application "Mail" to delete message 1 of inbox'
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response(sneaky_script)),
    ), patch(
        "app.desktop_client.run_applescript",
        new=AsyncMock(side_effect=AssertionError("must not execute")),
    ):
        result = await dispatch(
            "computer_use", {"goal": "clean my inbox"}, ctx=ctx,
        )
    assert result.text == t("computer_use.readonly_violation", "en")
    data = result.data or {}
    assert data.get("error") == "readonly_violation"
    assert data.get("goal") == "clean my inbox"
    # Excerpt preserved for debugging.
    assert "delete" in data.get("script_excerpt", "")


# ── 5. risk upgrade gate refuses 'set name' / 'new event' style ────────


async def test_computer_use_refuses_risk_upgrade(make_agent_ctx):
    """Destructive-pattern detector upgrades 'set name ...' to high_write
    even when the strict gate didn't reject.  The read-only contract
    on computer_use refuses anything above ``read``.
    """
    from app.i18n import t

    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    # "set name" matches the destructive pattern but doesn't appear in
    # the FORBIDDEN_IN_READONLY substring set.
    mutation_script = (
        'tell application "Calendar" to set name of event 1 to "Renamed"'
    )
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response(mutation_script)),
    ), patch(
        "app.desktop_client.run_applescript",
        new=AsyncMock(side_effect=AssertionError("must not execute")),
    ):
        result = await dispatch(
            "computer_use", {"goal": "rename event"}, ctx=ctx,
        )
    assert result.text == t("computer_use.readonly_violation", "en")
    data = result.data or {}
    assert data.get("error") == "risk_too_high"
    assert data.get("risk") == "high_write"


# ── 6. happy path — safe script executes ────────────────────────────────


async def test_computer_use_executes_safe_script(make_agent_ctx):
    """Read-only generator output → run_applescript called + stdout
    spoken back to the user.
    """
    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    safe_script = "set volume output volume 80"
    run_mock = AsyncMock(return_value={
        "ok": True, "stdout": "", "stderr": "",
    })
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response(safe_script)),
    ), patch(
        "app.desktop_client.run_applescript", new=run_mock,
    ):
        result = await dispatch(
            "computer_use",
            {"goal": "set volume to 80"},
            ctx=ctx,
        )
    # Empty stdout → "Done." for the English locale (CATALOG computer_use.done).
    assert result.text == "Done."
    data = result.data or {}
    assert data.get("script") == safe_script
    assert data.get("goal") == "set volume to 80"
    # run_applescript was called with the script + category.
    run_mock.assert_awaited_once()
    call = run_mock.call_args
    assert call.args[0] == safe_script
    assert call.kwargs.get("category") == "computer_use"


# ── 7. happy path — script with stdout returns the stdout ───────────────


async def test_computer_use_speaks_stdout(make_agent_ctx):
    """Read-style script (returns data) → stdout becomes the spoken reply."""
    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    script = (
        'tell application "Music" to return (name of current track) & '
        '" — " & (artist of current track)'
    )
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response(script)),
    ), patch(
        "app.desktop_client.run_applescript",
        new=AsyncMock(return_value={
            "ok": True,
            "stdout": "Bohemian Rhapsody — Queen",
            "stderr": "",
        }),
    ):
        result = await dispatch(
            "computer_use",
            {"goal": "what's playing in music right now"},
            ctx=ctx,
        )
    assert result.text == "Bohemian Rhapsody — Queen"


# ── 8. AppleScript runtime error → execution_failed ─────────────────────


async def test_computer_use_surfaces_runtime_error(make_agent_ctx):
    """osascript returned non-zero with stderr → execution_failed +
    stderr preserved in data for diagnostics.
    """
    from app.i18n import t

    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    script = 'tell application "NonExistentApp" to activate'
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response(script)),
    ), patch(
        "app.desktop_client.run_applescript",
        new=AsyncMock(return_value={
            "ok": False, "stdout": "",
            "stderr": "Application isn't running",
        }),
    ):
        result = await dispatch(
            "computer_use", {"goal": "open nonexistent app"}, ctx=ctx,
        )
    assert result.text == t("computer_use.execution_failed", "en")
    data = result.data or {}
    assert data.get("error") == "applescript_error"
    assert "Application isn't running" in data.get("stderr", "")


# ── 9. markdown-fence stripping ─────────────────────────────────────────


async def test_computer_use_strips_markdown_fences(make_agent_ctx):
    """The model wrapped the script in ```applescript ...``` despite
    the prompt — strip and execute.
    """
    _seed_default_reachable()
    ctx = make_agent_ctx(profile_id=42, client_id="cli", user_lang="en")
    fenced = "```applescript\nset volume output volume 50\n```"
    run_mock = AsyncMock(return_value={"ok": True, "stdout": "", "stderr": ""})
    with patch(
        "app.llm_utils.chat",
        new=AsyncMock(return_value=_mock_chat_response(fenced)),
    ), patch("app.desktop_client.run_applescript", new=run_mock):
        result = await dispatch(
            "computer_use", {"goal": "set volume to 50"}, ctx=ctx,
        )
    # Verify the fences were stripped before reaching run_applescript.
    assert run_mock.await_args.args[0] == "set volume output volume 50"
    assert (result.data or {}).get("script") == "set volume output volume 50"
