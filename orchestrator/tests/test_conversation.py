"""Conversation state: verbatim window, background summary, focus slots,
and the per-profile last-used category that seeds the "list" slot."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from app.agent import AgentResult, ToolInvocation, run_agent
from app.conversation import FOCUS_TTL_TURNS, Conversation
from app.storage.categories import (
    create_category,
    delete_category,
    get_last_category,
    set_last_category,
    share_category,
    unshare_category,
)
from app.storage.speaker_profiles import save_speaker_profile


def _inv(name, args=None, data=None):
    return ToolInvocation(name=name, args=args or {}, text="", data=data, terminal=True)


def _reply(text):
    return {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}


async def test_window_keeps_last_turns_and_folds_older_into_summary():
    chat = AsyncMock(return_value=_reply("We talked about Munich."))
    conv = Conversation(keep_turns=2)
    with patch("app.llm_utils.chat", new=chat):
        for i in range(3):
            conv.record(f"q{i}", f"a{i}")
        await conv.wait_summary()
    assert [m["content"] for m in conv.history] == ["q1", "a1", "q2", "a2"]
    assert conv.summary == "We talked about Munich."
    assert "q0" in chat.call_args.args[0][1]["content"]
    assert "Munich" in conv.messages()[0]["content"]
    assert conv.turns == 3


async def test_summary_failure_keeps_previous_summary():
    conv = Conversation(keep_turns=1, summary="old")
    with patch("app.llm_utils.chat", new=AsyncMock(side_effect=RuntimeError("down"))):
        conv.record("q0", "a0")
        conv.record("q1", "a1")
        await conv.wait_summary()
    assert conv.summary == "old"


def test_focus_from_tool_args_and_result_data():
    conv = Conversation()
    conv.record("weather in Munich?", "12 °C", [_inv("weather", {"location": "Munich"})])
    conv.record("what's on?", "Dentist at 15:00",
                [_inv("list_reminders", data={"focus": {"reminder": "dentist 15:00"}})])
    block = conv.focus_block()
    assert "location='Munich' (weather, 2 turns ago)" in block
    assert "reminder='dentist 15:00' (list_reminders, 1 turn ago)" in block
    # Most recent first.
    assert block.index("reminder=") < block.index("location=")


def test_failed_calls_and_unmapped_args_set_no_focus():
    conv = Conversation()
    conv.record("x", "y", [_inv("weather", {"location": "Nowhere"}, data={"error": "not_found"})])
    conv.record("x", "y", [_inv("calculator", {"expression": "2+2"})])
    assert conv.focus == {}
    assert conv.focus_block() == ""


def test_focus_expires_and_fallback_fills_missing_slot():
    conv = Conversation()
    conv.record("q", "a", [_inv("weather", {"location": "Munich"})])
    for _ in range(FOCUS_TTL_TURNS):
        conv.record("filler", "ok")
    assert "Munich" not in conv.focus_block()
    assert conv.focus_block({"list": "Shopping"}).endswith("list='Shopping' (last used)]")


def test_reset_clears_everything():
    conv = Conversation(summary="s")
    conv.record("q", "a", [_inv("weather", {"location": "Munich"})])
    conv.reset()
    assert (conv.history, conv.summary, conv.focus) == ([], "", {})


def test_from_rows_rebuilds_focus_from_last_tool():
    conv = Conversation.from_rows([
        ("weather in Munich?", "12 °C", "weather", json.dumps({"location": "Munich"})),
        ("thanks", "you're welcome", None, None),
    ])
    assert conv.turn == 2
    assert conv.focus["location"].value == "Munich"


async def test_run_agent_puts_focus_right_before_the_question(make_agent_ctx):
    chat = AsyncMock(return_value=_reply("ok"))
    with patch("app.agent.chat", new=chat):
        await run_agent(
            "what's the weather there?", ctx=make_agent_ctx(),
            history=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            focus_block="[Conversation focus — location='Munich']",
        )
    last = chat.call_args.args[0][-1]
    assert last == {
        "role": "user",
        "content": "[Conversation focus — location='Munich']\nwhat's the weather there?",
    }


async def test_last_category_respects_access():
    owner = await save_speaker_profile("c", "Anna", b"\0")
    other = await save_speaker_profile("c", "Ben", b"\0")
    cat = await create_category(owner_profile_id=owner, name="Shopping", kind="checklist")
    await set_last_category(owner, cat)
    assert (await get_last_category(owner))["name"] == "Shopping"

    await share_category(cat, other, permission="write")
    await set_last_category(other, cat)
    assert (await get_last_category(other))["name"] == "Shopping"
    await unshare_category(cat, other)
    assert await get_last_category(other) is None

    await delete_category(cat, owner)
    assert await get_last_category(owner) is None


async def test_categories_tool_persists_last_category():
    """End-to-end through the tools: creating a checklist persists it, and a
    fresh conversation for the same profile sees it as the list fallback."""
    from app.tools.categories import categories

    pid = await save_speaker_profile("c", "Anna", b"\0")

    class Ctx:
        client_id, profile_id, user_lang = "c", pid, "en"
        is_authenticated = True
        progress_sink = stream_sink = media_sink = None

    res = await categories("create", name="Shopping", kind="checklist", ctx=Ctx())
    assert res.data["focus"] == {"list": "Shopping"}
    assert (await get_last_category(pid))["name"] == "Shopping"
    assert "list='Shopping' (last used)" in Conversation().focus_block(
        {"list": (await get_last_category(pid))["name"]}
    )
