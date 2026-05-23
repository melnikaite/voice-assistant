"""
Smoke tests for tools that don't need external services.

These verify the tool is correctly registered (the @tool decorator ran,
the schema is in TOOL_REGISTRY, dispatch can route to it) and that the
happy path returns a sensible ToolResult.  External-network tools
(weather, news, web_search, translate) are skipped — they're better
tested via the pipeline integration tests that mock HTTP wholesale.

Each test goes through ``dispatch(name, args, ctx=...)`` so the path
exercises:
  1. TOOL_REGISTRY lookup (catches schema/decorator typos)
  2. ``wants_ctx`` detection + injection
  3. i18n key resolution (catches missing keys before they hit users)
  4. ToolResult construction
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import AgentContext
from app.tools import TOOL_REGISTRY, dispatch


def _ctx(profile_id: int | None = None, *, lang: str = "en") -> AgentContext:
    return AgentContext(
        client_id="cli-smoke",
        profile_id=profile_id,
        is_authenticated=profile_id is not None,
        user_lang=lang,
        stream_sink=None,
        progress_sink=None,
    )


# ── registration sanity ─────────────────────────────────────────────────


def test_every_registered_tool_has_a_schema():
    """Smoke: every tool in TOOL_REGISTRY exposes a valid OpenAI-style schema."""
    assert TOOL_REGISTRY, "no tools registered — import side-effects broken?"
    for name, entry in TOOL_REGISTRY.items():
        schema = entry["schema"]
        assert schema["type"] == "function", f"{name}: schema not type=function"
        fn = schema["function"]
        assert fn["name"] == name, f"{name}: schema.function.name mismatch"
        assert isinstance(fn.get("description"), str) and fn["description"], (
            f"{name}: missing description"
        )
        assert fn["parameters"]["type"] == "object", f"{name}: params not object"


def test_known_tools_are_registered():
    """Pin the set of public tool names so accidental deletions show up.

    Names mirror what the @tool decorator declares, NOT the module
    filename — they're the strings the LLM sees and chooses by.
    """
    expected = {
        "calculator", "weather", "web_search", "news_briefing", "translate",
        "general_answer", "my_history", "look_at_screen", "computer_use",
        "desktop", "remember", "read_memory", "update_settings",
        "read_settings", "reminders", "items", "categories",
        "inbox_list", "inbox_read", "inbox_reply", "inbox_summary",
        "list_pending", "approve_pending", "reject_pending",
    }
    missing = expected - set(TOOL_REGISTRY.keys())
    assert not missing, f"tools disappeared: {missing}"


# ── calculator ──────────────────────────────────────────────────────────


async def test_calculator_arith_addition():
    """2+2=4 — the hello-world smoke."""
    result = await dispatch("calculator", {"mode": "arith", "expression": "2 + 2"}, ctx=_ctx())
    assert "4" in result.text
    assert result.data is not None
    # 'value' key isn't standard; just confirm no error flag.
    assert "error" not in (result.data or {})


async def test_calculator_arith_division_by_zero():
    """Specific error path — has its own i18n key."""
    result = await dispatch("calculator", {"mode": "arith", "expression": "1 / 0"}, ctx=_ctx())
    assert result.data is not None
    assert result.data.get("error") == "div_zero"


async def test_calculator_arith_rejects_empty_expression():
    """Missing required-by-mode arg returns a structured error, not a crash."""
    result = await dispatch("calculator", {"mode": "arith", "expression": ""}, ctx=_ctx())
    assert result.data is not None
    assert result.data.get("error") == "no_expression"


# ── memory (read + write round-trip) ────────────────────────────────────


async def test_memory_round_trip(tmp_path, monkeypatch):
    """write_memory then read_memory round-trips through the user_files layer.

    Uses a temp profile id and monkey-patches DATA_DIR_CONTAINER so the
    file doesn't pollute the real /data mount.
    """
    monkeypatch.setenv("DATA_DIR_CONTAINER", str(tmp_path))
    # Re-import user_files to pick up the patched env var.
    import importlib
    from app import user_files
    importlib.reload(user_files)

    from app.storage import save_speaker_profile
    pid = await save_speaker_profile(
        client_id="cli-mem", name="Mem", embedding=b"\x00" * 4 * 256,
    )

    # write
    res_w = await dispatch(
        "remember",
        {"action": "append", "content": "loves espresso"},
        ctx=_ctx(profile_id=pid),
    )
    assert res_w.data is not None
    assert "error" not in res_w.data

    # read
    res_r = await dispatch("read_memory", {}, ctx=_ctx(profile_id=pid))
    assert res_r.data is not None
    assert "loves espresso" in (res_r.data.get("content") or "")


async def test_memory_no_profile_returns_structured_error():
    """ctx without profile_id can't read memory — error, not crash."""
    result = await dispatch("read_memory", {}, ctx=_ctx(profile_id=None))
    assert result.data is not None
    assert result.data.get("error") == "no_profile"


# ── settings ────────────────────────────────────────────────────────────


async def test_settings_read_defaults(tmp_path, monkeypatch):
    """Fresh profile has no settings file → tool returns defaults, not error."""
    monkeypatch.setenv("DATA_DIR_CONTAINER", str(tmp_path))
    import importlib
    from app import user_files
    importlib.reload(user_files)

    from app.storage import save_speaker_profile
    pid = await save_speaker_profile(
        client_id="cli-set", name="Set", embedding=b"\x00" * 4 * 256,
    )

    result = await dispatch("read_settings", {}, ctx=_ctx(profile_id=pid))
    # Defaults should not flag as an error.
    assert "error" not in (result.data or {})


# ── unknown tool dispatch ───────────────────────────────────────────────


async def test_dispatch_unknown_tool_returns_error():
    """Calling a non-existent tool name surfaces a friendly error, not a crash."""
    result = await dispatch("not_a_real_tool", {}, ctx=_ctx())
    assert result.data is not None
    assert result.data.get("error") == "unknown_tool"
