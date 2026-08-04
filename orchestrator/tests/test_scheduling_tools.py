"""
Scheduling tools — set_timer / set_reminder_at / list_reminders / cancel_reminder.

These four replaced a single multiplexed ``reminders`` tool.  The split is not
cosmetic: JSON-Schema ``required`` is unconditional, so one tool covering
create/list/cancel could only require ``action``, leaving the duration
optional — and a model that declines to emit optional fields then called it
with no duration at all.  ``test_required_field_contract`` is the regression
guard for exactly that, so keep it if the tools are ever reshuffled.

Everything goes through ``dispatch`` so the decorator, ctx injection and i18n
keys are exercised the same way the agent loop exercises them.
"""
from __future__ import annotations

import datetime

import pytest

from app import scheduler as sched
from app.agent import AgentContext
from app.storage import list_upcoming_reminders
from app.tools import TOOL_REGISTRY, dispatch
from app.tools.set_reminder import _parse_iso_duration


@pytest.fixture(autouse=True)
async def _scheduler():
    """Creating a reminder registers an APScheduler job, which asserts the
    scheduler is running.  Start a real one per test — it needs the event
    loop pytest-asyncio already provides, and nothing here fires."""
    sched.start()
    yield
    sched.stop()


def _ctx(*, lang: str = "en") -> AgentContext:
    return AgentContext(
        client_id="cli-sched",
        profile_id=1,
        is_authenticated=True,
        user_lang=lang,
        stream_sink=None,
        progress_sink=None,
    )


# ── ISO-8601 duration parsing ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("PT20M", 1200),
        ("PT1H30M", 5400),          # "полтора часа"
        ("PT2H30M", 9000),          # "два с половиной часа"
        ("PT90S", 90),
        ("PT45M", 2700),
        ("PT1H", 3600),
        ("P1D", 86400),
        ("P1DT2H", 93600),
        ("pt1h30m", 5400),          # models are not reliable about case
        ("  PT20M  ", 1200),        # nor about whitespace
    ],
)
def test_parse_iso_duration_accepts(text, seconds):
    assert _parse_iso_duration(text) == seconds


@pytest.mark.parametrize(
    "text",
    [
        "",
        "20m",          # Go style — deliberately NOT accepted, one format only
        "20 minutes",   # natural language stays the model's job to transcribe
        "1200",         # a bare integer is not a duration string
        "P",            # no components — must not become a 0-second timer
        "PT",
        "PT1X",
        "T1H",          # missing the leading P
        "PT1H30",       # unit-less trailing number
        None,
    ],
)
def test_parse_iso_duration_rejects(text):
    assert _parse_iso_duration(text) is None


# ── the contract the split exists to enforce ────────────────────────────


def test_required_field_contract():
    """Each scheduling tool must REQUIRE the field that carries its meaning.

    A model that only emits required fields has to produce a usable call.
    """
    expected = {
        "set_timer": "duration",
        "set_reminder_at": "fire_at",
        "cancel_reminder": "text",
    }
    for name, field in expected.items():
        params = TOOL_REGISTRY[name]["schema"]["function"]["parameters"]
        assert field in params.get("required", []), (
            f"{name}: {field!r} must be required, else a model that skips "
            f"optional fields calls {name} with nothing usable"
        )
    listing = TOOL_REGISTRY["list_reminders"]["schema"]["function"]["parameters"]
    assert not listing.get("required"), "list_reminders takes no arguments"


# ── set_timer ───────────────────────────────────────────────────────────


async def test_set_timer_creates_a_reminder():
    result = await dispatch("set_timer", {"duration": "PT20M", "text": "eggs"}, ctx=_ctx())
    assert result.data.get("reminder_id")
    assert result.data["trigger"] == "duration"
    upcoming = await list_upcoming_reminders("cli-sched")
    assert len(upcoming) == 1
    # 20 minutes out, allowing for test-run slack.
    delta = upcoming[0]["fire_at"] - datetime.datetime.now().timestamp()
    assert 1150 < delta <= 1200
    assert "eggs" in upcoming[0]["push_text"]


async def test_set_timer_without_label_still_works():
    result = await dispatch("set_timer", {"duration": "PT90S"}, ctx=_ctx())
    assert result.data.get("reminder_id")
    assert len(await list_upcoming_reminders("cli-sched")) == 1


async def test_set_timer_rejects_unparseable_duration():
    """A bad duration must fail loudly and schedule nothing."""
    result = await dispatch("set_timer", {"duration": "half an hour"}, ctx=_ctx())
    assert result.data.get("error") == "bad_iso_duration"
    assert await list_upcoming_reminders("cli-sched") == []


# ── set_reminder_at ─────────────────────────────────────────────────────


async def test_set_reminder_at_schedules_absolute():
    when = datetime.datetime.now().astimezone() + datetime.timedelta(hours=3)
    result = await dispatch(
        "set_reminder_at",
        {"fire_at": when.replace(microsecond=0).isoformat(), "text": "call mom"},
        ctx=_ctx(),
    )
    assert result.data.get("reminder_id")
    assert result.data["trigger"] == "absolute"
    assert len(await list_upcoming_reminders("cli-sched")) == 1


async def test_set_reminder_at_refuses_the_past():
    when = datetime.datetime.now().astimezone() - datetime.timedelta(hours=1)
    result = await dispatch(
        "set_reminder_at", {"fire_at": when.isoformat()}, ctx=_ctx()
    )
    assert result.data.get("error") == "past_time"
    assert await list_upcoming_reminders("cli-sched") == []


# ── list / cancel ───────────────────────────────────────────────────────


async def test_list_reminders_empty_then_populated():
    empty = await dispatch("list_reminders", {}, ctx=_ctx())
    assert empty.data["reminders"] == []
    await dispatch("set_timer", {"duration": "PT10M", "text": "eggs"}, ctx=_ctx())
    listed = await dispatch("list_reminders", {}, ctx=_ctx())
    assert len(listed.data["reminders"]) == 1


async def test_cancel_reminder_matches_fuzzily():
    await dispatch("set_timer", {"duration": "PT10M", "text": "boil the eggs"}, ctx=_ctx())
    result = await dispatch("cancel_reminder", {"text": "eggs"}, ctx=_ctx())
    assert result.data.get("reminder_id")
    assert await list_upcoming_reminders("cli-sched") == []


def test_word_overlap_ignores_punctuation():
    """Regression: the stored text is i18n-punctuated ("eggs! 10 minutes has
    passed."), so tokenising on whitespace made cancel-by-label impossible."""
    from app.tools.set_reminder import _word_overlap

    assert _word_overlap("eggs", "boil the eggs! 10 minutes has passed.") > 0
    assert _word_overlap("молоко", "молоко! Прошло 10 минут.") > 0


async def test_cancel_reminder_reports_no_match():
    await dispatch("set_timer", {"duration": "PT10M", "text": "eggs"}, ctx=_ctx())
    result = await dispatch("cancel_reminder", {"text": "completely unrelated"}, ctx=_ctx())
    assert result.data.get("error") is None  # not an error — just nothing matched
    assert "score" in result.data
    assert len(await list_upcoming_reminders("cli-sched")) == 1
