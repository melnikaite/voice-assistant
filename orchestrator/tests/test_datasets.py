"""Structured datasets: schema/validation, access, the ``datasets`` tool,
conversation hold and the focused-category context block."""
from __future__ import annotations

import datetime as dt
import time

import pytest

from app import datasets as dsl
from app.conversation import HELD_TTL_TURNS, Conversation, _Inv
from app.storage.categories import create_category, set_last_category, share_category
from app.storage.datasets import (
    add_entry,
    get_dataset,
    list_datasets,
    list_entries,
    resolve_dataset,
)
from app.tools.datasets import _datasets_risk, _summary, context_block, datasets

FIELDS = [
    {"name": "month", "type": "month"},
    {"name": "service", "type": "string"},
    {"name": "amount", "type": "number", "unit": "EUR"},
    {"name": "provider", "type": "string", "required": False},
    {"name": "meter.from", "type": "number", "required": False},
    {"name": "meter.to", "type": "number", "required": False},
]


class _Ctx:
    def __init__(self, profile_id=1, held=None, authenticated=True):
        self.profile_id = profile_id
        self.user_lang = "en"
        self.progress_sink = None
        self.stream_sink = None
        self.is_authenticated = authenticated
        self.client_id = f"c{profile_id}"
        self.held = held
        self.transcript = ""


@pytest.fixture(autouse=True)
async def _profiles():
    from app.storage.db import _conn
    for pid in (1, 2, 3):
        _conn().execute(
            "INSERT OR IGNORE INTO speaker_profiles (id, client_id, name, embedding,"
            " sample_count, created_at) VALUES (?, 'c', ?, X'00', 1, ?)",
            (pid, f"P{pid}", time.time()),
        )


async def _bills(profile_id=1, category="Utilities") -> dict:
    cid = await create_category(owner_profile_id=profile_id, name=category)
    await set_last_category(profile_id, cid)
    r = await datasets("create", ctx=_Ctx(profile_id), name="Bills", category=category,
                       description="monthly utility bills", fields=FIELDS)
    assert r.data.get("dataset_id"), r.text
    return await get_dataset(profile_id, r.data["dataset_id"])


# ── Schema + validation (pure) ────────────────────────────────────────────

def test_build_schema_nests_dotted_names_and_hints():
    s = dsl.build_schema(FIELDS)
    assert s["properties"]["meter"]["properties"]["from"] == {"type": "number"}
    assert s["properties"]["month"] == {"type": "string", "format": "month"}
    assert s["required"] == ["month", "service", "amount"]
    assert dsl.guess_hints(s) == ("amount", "month")


def test_validate_reports_missing_unknown_and_coerces():
    s = dsl.build_schema(FIELDS)
    clean, missing, unknown, invalid = dsl.validate(
        {"month": "09.2026", "amount": "84,5 EUR", "tariff": "night",
         "meter": {"from": "1203"}}, s)
    assert clean == {"month": "2026-09", "amount": 84.5, "meter": {"from": 1203}}
    assert missing == ["service"]
    assert unknown == ["tariff"]
    assert invalid == []
    _, _, _, invalid = dsl.validate({"month": "soon", "service": "x", "amount": 1}, s)
    assert invalid == ["month"]


def test_merge_schema_adds_optional_fields():
    s = dsl.merge_schema(dsl.build_schema(FIELDS), [{"name": "tariff", "type": "string"}])
    assert "tariff" in s["properties"] and "tariff" not in s["required"]


def test_unflatten_accepts_dotted_keys():
    assert dsl.unflatten({"meter.from": 1, "meter": {"to": 2}}) == {"meter": {"from": 1, "to": 2}}


def test_period_bounds():
    d = dt.date(2026, 2, 10)
    assert dsl.period_bounds("this_year", d) == ("2026", "2026")
    assert dsl.period_bounds("last_month", d) == ("2026-01", "2026-01")
    assert dsl.period_bounds("last_12_months", d) == ("2025-03", "2026-02")


def test_forecast_needs_twelve_periods():
    short = [(f"2026-{m:02d}", 80.0) for m in range(1, 8)]
    assert dsl.forecast_monthly(short) == {"enough": False, "periods": 7, "needed": 12}


def test_forecast_follows_trend_and_season():
    series = []
    for i in range(36):
        y, m = 2023 + i // 12, i % 12 + 1
        series.append((f"{y}-{m:02d}", 100 + i + (40 if m in (12, 1, 2) else 0)))
    fc = dsl.forecast_monthly(series, horizon=2)
    jan, feb = fc["points"]
    assert jan["period"] == "2026-01" and feb["period"] == "2026-02"
    # trend says ~136; winter season adds most of +40
    assert 160 < jan["value"] < 180
    assert jan["low"] < jan["value"] < jan["high"]


# ── Tool: create / add / hold / extend ────────────────────────────────────

async def test_create_and_add_complete_entry():
    ds = await _bills()
    assert ds["amount_field"] == "amount" and ds["period_field"] == "month"
    r = await datasets("add", ctx=_Ctx(), dataset="bills", entry={
        "month": "2026-09", "service": "electricity", "amount": 84, "currency": None,
        "provider": "Stadtwerke", "meter": {"from": 1203, "to": 1415}})
    assert r.data["saved"] is True, r.text
    assert r.data["hold"] is None
    rows = await list_entries(1, ds["id"])
    assert rows[0]["data"]["meter"] == {"from": 1203, "to": 1415}


async def test_missing_field_is_held_then_completed():
    ds = await _bills()
    r = await datasets("add", ctx=_Ctx(), dataset="Bills",
                       entry={"service": "water", "amount": 31})
    assert r.data["saved"] is False and r.data["missing"] == ["month"]
    assert await list_entries(1, ds["id"]) == []
    held = r.data["hold"]
    # next turn: the user only says the month
    r2 = await datasets("add", ctx=_Ctx(held=held), entry={"month": "2026-08"})
    assert r2.data["saved"] is True, r2.text
    assert r2.data["entry"] == {"month": "2026-08", "service": "water", "amount": 31}


async def test_unknown_field_queues_extension_and_saves_nothing():
    from app.storage import list_pending_actions
    ds = await _bills()
    r = await datasets("add", ctx=_Ctx(), dataset="Bills", entry={
        "month": "2026-09", "service": "gas", "amount": 40, "tariff": "night"})
    assert r.data["saved"] is False and r.data["unknown"] == ["tariff"]
    assert await list_entries(1, ds["id"]) == []
    pend = await list_pending_actions(profile_id=1, client_id="c1")
    assert len(pend) == 1 and pend[0]["tool_args"]["action"] == "extend"
    # approval replays the queued call with an authenticated context
    r2 = await datasets(ctx=_Ctx(), **pend[0]["tool_args"])
    assert r2.data["saved"] is True, r2.text
    rows = await list_entries(1, ds["id"])
    assert rows[0]["data"]["tariff"] == "night"


def test_risk_and_summary():
    assert _datasets_risk({"action": "add"}) == "low_write"
    assert _datasets_risk({"action": "stats"}) == "read"
    for a in ("create", "extend"):
        assert _datasets_risk({"action": a, "fields": [{"name": "x"}]}) == "high_write"
        assert _datasets_risk({"action": a}) == "read"  # refuses without writing
    for a in ("delete", "bogus"):
        assert _datasets_risk({"action": a}) == "high_write"
    s = _summary({"action": "create", "name": "Bills", "category": "Utilities",
                  "fields": [{"name": "amount", "unit": "EUR"}, {"name": "month"}]})
    assert s == "create dataset «Bills» in «Utilities»: amount (EUR), month"


async def test_deferred_create_uses_summary_hook(make_agent_ctx):
    from app.agent import _execute_one
    from app.llm_utils import ParsedToolCall
    from app.storage import list_pending_actions
    tc = ParsedToolCall(id="1", name="datasets", args={
        "action": "create", "name": "Bills", "fields": [{"name": "amount", "type": "number"}]})
    inv = await _execute_one(tc, make_agent_ctx(profile_id=1, is_authenticated=False))
    assert inv.data["deferred"] is True
    pend = await list_pending_actions(profile_id=1, client_id="test-client")
    assert pend[0]["summary"] == "create dataset «Bills»: amount"


# ── Stats / forecast ──────────────────────────────────────────────────────

async def _seed(ds, rows):
    for r in rows:
        assert await add_entry(1, ds["id"], r)


async def test_stats_sum_filter_period_and_grouping():
    ds = await _bills()
    y = dt.date.today().year
    await _seed(ds, [
        {"month": f"{y}-01", "service": "electricity", "amount": 80},
        {"month": f"{y}-02", "service": "Electricity", "amount": 90.5},
        {"month": f"{y}-02", "service": "water", "amount": 30},
        {"month": f"{y - 1}-12", "service": "electricity", "amount": 70},
    ])
    r = await datasets("stats", ctx=_Ctx(), dataset="Bills", op="sum",
                       filters={"service": "electricity"}, period="this_year")
    assert r.data["value"] == 170.5 and r.data["entries"] == 2
    assert "170.5 EUR" in r.text
    r = await datasets("stats", ctx=_Ctx(), dataset="Bills", op="sum", group_by="service")
    # grouping keeps raw values; filters are case-insensitive
    assert {row["group"]: row["value"] for row in r.data["rows"]} == {
        "electricity": 150, "Electricity": 90.5, "water": 30}
    r = await datasets("stats", ctx=_Ctx(), dataset="Bills", op="count", group_by="quarter",
                       period_from=f"{y}")
    assert [(row["group"], row["value"]) for row in r.data["rows"]] == [(f"{y}-Q1", 3)]
    r = await datasets("stats", ctx=_Ctx(), dataset="Bills", field="nope")
    assert r.data["error"] == "unknown_field"


async def test_stats_over_category_subtree():
    home = await create_category(owner_profile_id=1, name="Home")
    a = await create_category(owner_profile_id=1, name="Flat A", parent_id=home)
    b = await create_category(owner_profile_id=1, name="Flat B", parent_id=home)
    ids = []
    for cat in ("Home", "Flat A", "Flat B"):
        r = await datasets("create", ctx=_Ctx(), name="Bills", category=cat, fields=FIELDS)
        ids.append(r.data["dataset_id"])
    for i, ds_id in enumerate(ids):
        assert await add_entry(1, ds_id, {"month": "2026-01", "service": "gas", "amount": 10 * (i + 1)})
    r = await datasets("stats", ctx=_Ctx(), dataset="Bills", category="Home",
                       include_subtree=True)
    assert r.data["value"] == 60
    r = await datasets("stats", ctx=_Ctx(), dataset="Bills", category="Flat B")
    assert r.data["value"] == 30
    assert a and b


async def test_forecast_not_enough_data_is_honest():
    ds = await _bills()
    await _seed(ds, [{"month": f"2026-0{m}", "service": "gas", "amount": 50} for m in range(1, 5)])
    r = await datasets("forecast", ctx=_Ctx(), dataset="Bills")
    assert r.data["enough"] is False and r.data["periods"] == 4
    assert "Not enough data" in r.text


async def test_forecast_with_history():
    ds = await _bills()
    await _seed(ds, [{"month": f"{2024 + i // 12}-{i % 12 + 1:02d}", "service": "gas",
                      "amount": 50 + i} for i in range(24)])
    r = await datasets("forecast", ctx=_Ctx(), dataset="Bills")
    assert r.data["enough"] is True and r.data["points"][0]["period"] == "2026-01"


# ── Access ────────────────────────────────────────────────────────────────

async def test_other_profile_needs_a_share():
    ds = await _bills()
    assert await add_entry(1, ds["id"], {"month": "2026-01", "service": "gas", "amount": 5})
    # no share: invisible to profile 2, every path
    assert await resolve_dataset(2, "Bills") is None
    assert await list_entries(2, ds["id"]) is None
    r = await datasets("stats", ctx=_Ctx(2), dataset="Bills")
    assert r.data["error"] == "dataset_not_found"
    assert await add_entry(2, ds["id"], {"month": "2026-02", "service": "gas", "amount": 1}) is None
    # read share: stats yes, writing no
    await share_category(ds["category_id"], 2, permission="read")
    r = await datasets("stats", ctx=_Ctx(2), dataset="Bills")
    assert r.data["value"] == 5
    r = await datasets("add", ctx=_Ctx(2), dataset="Bills",
                       entry={"month": "2026-02", "service": "gas", "amount": 1})
    assert r.data["error"] == "forbidden"
    # write share: adding yes, schema changes still owner-only
    await share_category(ds["category_id"], 2, permission="write")
    r = await datasets("add", ctx=_Ctx(2), dataset="Bills",
                       entry={"month": "2026-02", "service": "gas", "amount": 1})
    assert r.data["saved"] is True
    r = await datasets("delete", ctx=_Ctx(2), dataset="Bills")
    assert r.data["error"] == "forbidden"
    # profile 3 still sees nothing
    assert await list_datasets(3) == []


async def test_share_on_parent_reaches_subtree_and_delete_hides():
    home = await create_category(owner_profile_id=1, name="Home")
    await create_category(owner_profile_id=1, name="Utilities", parent_id=home)
    await datasets("create", ctx=_Ctx(), name="Bills", category="Utilities", fields=FIELDS)
    await share_category(home, 2, permission="read")
    assert await resolve_dataset(2, "Bills") is not None
    r = await datasets("delete", ctx=_Ctx(), dataset="Bills")
    assert r.data.get("dataset_id"), r.text
    assert await resolve_dataset(1, "Bills") is None


# ── Conversation hold + context block ─────────────────────────────────────

def test_conversation_holds_until_cleared_or_expired():
    conv = Conversation()
    hold = {"dataset": "Bills", "dataset_id": 1, "entry": {}, "missing": ["month"]}
    conv.record("add water 31", "need month", [_Inv("datasets", {"action": "add"}, {"hold": hold})])
    assert conv.held == hold
    conv.record("September", "saved", [_Inv("datasets", {"action": "add"}, {"hold": None})])
    assert conv.held is None
    conv.record("again", "need month", [_Inv("datasets", {"action": "add"}, {"hold": hold})])
    for _ in range(HELD_TTL_TURNS):
        conv.record("weather?", "sunny")
    assert conv.held is None


async def test_context_block_lists_focused_category_datasets():
    await _bills()
    block = await context_block(1, "Utilities", today=dt.date(2026, 10, 18))
    assert block.startswith("[Datasets in «Utilities»")
    assert "month*:month" in block and "amount*:number EUR" in block and "meter.from:number" in block
    assert "Today is 2026-10-18" in block
    assert await context_block(2, "Utilities") == ""  # not shared
    held = {"dataset": "Bills", "missing": ["month"]}
    assert "still needs: month" in await context_block(1, None, held)


def test_month_names_and_period_cues():
    assert dsl.normalize_month("сентябрь 2026") == "2026-09"
    assert dsl.normalize_month("Sept. 2026") == "2026-09"
    assert dsl.normalize_month("2026 März") == "2026-03"
    assert dsl.normalize_month("сентябрь") is None
    assert dsl.has_period_cue("За прошлый месяц.")
    assert dsl.has_period_cue("for August")
    assert not dsl.has_period_cue("Запиши счёт за воду, 31 евро.")


async def test_made_up_month_and_provider_are_dropped():
    ds = await _bills()
    ctx = _Ctx()
    ctx.transcript = "Запиши счёт за воду, 31 евро."
    r = await datasets("add", ctx=ctx, dataset="Bills", entry={
        "month": "2026-10", "service": "вода", "amount": 31, "provider": "Водоканал"})
    assert r.data["saved"] is False and r.data["missing"] == ["month"]
    assert r.data["hold"]["entry"] == {"service": "вода", "amount": 31}
    assert await list_entries(1, ds["id"]) == []


def test_relative_period():
    d = dt.date(2026, 1, 10)
    assert dsl.relative_period("За прошлый месяц.", d) == {"month": "2025-12"}
    assert dsl.relative_period("Für letzten Monat", d) == {"month": "2025-12"}
    assert dsl.relative_period("paid yesterday", d) == {"date": "2026-01-09", "month": "2026-01"}
    assert dsl.relative_period("31 euros", d) == {}


async def test_relative_month_overrides_model_arithmetic():
    ds = await _bills()
    ctx = _Ctx()
    ctx.transcript = "За прошлый месяц."
    held = {"dataset": "Bills", "dataset_id": ds["id"], "entry": {"service": "water", "amount": 31},
            "missing": ["month"]}
    ctx.held = held
    r = await datasets("add", ctx=ctx, entry={"month": dt.date.today().strftime("%Y-%m")})
    prev = dt.date.today().replace(day=1) - dt.timedelta(days=1)
    assert r.data["entry"]["month"] == prev.strftime("%Y-%m")


async def test_routes_list_and_delete_entry():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage import create_session

    ds = await _bills()
    entry_id = await add_entry(1, ds["id"], {"month": "2026-01", "service": "gas", "amount": 5})
    with TestClient(app) as c:
        c.cookies.set("va_session", await create_session(1))
        body = c.get(f"/api/users/1/categories/{ds['category_id']}/datasets").json()
        assert body["datasets"][0]["entries"][0]["data"]["amount"] == 5
        assert [f["path"] for f in body["datasets"][0]["fields"]][:3] == ["month", "service", "amount"]
        assert c.delete(f"/api/users/1/dataset-entries/{entry_id}").status_code == 200
        assert c.delete(f"/api/users/1/dataset-entries/{entry_id}").status_code == 404
        c.cookies.set("va_session", await create_session(2))
        body = c.get(f"/api/users/2/categories/{ds['category_id']}/datasets").json()
        assert body["datasets"] == []
