"""
datasets — structured JSON datasets inside a category (7 actions).

A dataset is a named JSON Schema owned by a category ("Utilities" →
"Bills", "Meter readings"); each entry is one denormalized JSON document.
The model drafts schemas and picks parameters; validation, statistics
and forecasts are computed here, never by the model.

Actions and their risk levels:
  list      read        — datasets of a category with their fields
  create    high_write  — new dataset from a drafted field list (confirmed)
  add       low_write   — validate + save one entry; asks for missing fields
  stats     read        — sum/avg/min/max/count, grouped and filtered
  forecast  read        — trend + seasonality, honest below 12 periods
  extend    high_write  — add fields to the schema (confirmed), then
                          optionally save the entry that needed them
  delete    high_write  — soft-delete a dataset

Missing required fields: ``add`` saves nothing and returns
``data["hold"]``; the Conversation keeps that partial entry and hands it
back through ``ctx.held`` on the next turn, where ``add`` merges the
user's answer into it.  Unknown fields: ``add`` queues an ``extend``
call in ``pending_actions`` instead of saving — the user approves it
like any other deferred action.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any

from .. import datasets as dsl
from ..i18n import t
from ..storage import enqueue_action
from ..storage.categories import (
    create_category,
    get_last_category,
    resolve_category_by_name,
    set_last_category,
)
from ..storage.datasets import (
    add_entry,
    aggregate,
    create_dataset,
    delete_dataset,
    distinct_values,
    list_datasets,
    resolve_dataset,
    subtree_datasets,
    update_dataset_schema,
)
from .base import RiskLevel, ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)

_RISK: dict[str, RiskLevel] = {
    "list": "read", "stats": "read", "forecast": "read",
    "add": "low_write",
    "create": "high_write", "extend": "high_write", "delete": "high_write",
}


def _datasets_risk(args: dict) -> RiskLevel:
    # A create/extend without fields only answers "which fields?" — let it
    # run instead of parking a call that can't succeed in the pending queue.
    if args.get("action") in ("create", "extend") and not args.get("fields"):
        return "read"
    return _RISK.get(args.get("action", ""), "high_write")


def _summary(args: dict) -> str:
    """Pending-queue readback: the drafted fields, not ``k=v`` noise."""
    action = args.get("action")
    name = args.get("dataset") or args.get("name") or "?"
    fields = ", ".join(
        f.get("name", "?") + (f" ({f['unit']})" if f.get("unit") else "")
        for f in args.get("fields") or [] if isinstance(f, dict)
    )
    if action == "create":
        cat = f" in «{args['category']}»" if args.get("category") else ""
        return f"create dataset «{name}»{cat}: {fields}"
    if action == "extend":
        return f"add fields to «{name}»: {fields}"
    if action == "delete":
        return f"delete dataset «{name}»"
    return f"datasets {action} «{name}»"


@tool(
    name="datasets",
    description=(
        "Structured records the user tracks inside a folder: bills, meter "
        "readings, expenses, workouts, weights.  Each dataset has fixed "
        "fields; the runtime validates entries and computes all numbers.\n"
        "ACTIONS:\n"
        "  create    Start tracking something new ('start tracking my utility "
        "bills: month, service, amount').  ALWAYS pass `fields`, drafted from the sentence "
        "(English snake_case names, type month for a billing month, date for "
        "a day, number for amounts with `unit`), plus `name`, `description`, "
        "`category`.  The user confirms the draft.\n"
        "  add       Record ONE fact.  `entry` holds ONLY values the user "
        "actually said, each under the closest field name (the billed thing → "
        "service) — never guess a month, date or "
        "provider the user didn't say; leave it out and the runtime asks.  A "
        "value with no matching field goes under its own new key (e.g. "
        "'tariff'), never into another field.\n"
        "  stats     Past totals/averages/min/max/count.  `filters` {field: "
        "value} for every qualifier the user names (e.g. service), `period` "
        "for 'this year'/'last month', `op`, optional `group_by` "
        "(month|quarter|year|<field>) only when asked per month etc.\n"
        "  forecast  FUTURE amounts ('will I pay', 'next month', 'expect'): "
        "`filters` as for stats, `horizon` months.\n"
        "  list      Show datasets and their fields.\n"
        "  extend    Add `fields` to an existing dataset (confirmed).\n"
        "  delete    Delete a dataset (confirmed).\n"
        "Never compute sums or forecasts yourself — call stats/forecast."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["list", "create", "add", "stats", "forecast", "extend", "delete"]},
            "dataset": {"type": "string", "description": "Dataset name, e.g. 'Bills'."},
            "category": {"type": "string",
                         "description": "Folder that owns the dataset; default: the focused folder."},
            "name": {"type": "string", "description": "create: name of the new dataset."},
            "description": {"type": "string",
                            "description": "create: what it is and when to add an entry."},
            "fields": {
                "type": "array",
                "description": "create/extend: field definitions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case; 'meter.from' nests."},
                        "type": {"type": "string",
                                 "enum": ["string", "number", "integer", "boolean", "month", "date"]},
                        "unit": {"type": "string", "description": "e.g. EUR, kWh, kg."},
                        "required": {"type": "boolean", "description": "default true"},
                    },
                    "required": ["name", "type"],
                },
            },
            "amount_field": {"type": "string", "description": "create: field holding the amount."},
            "period_field": {"type": "string", "description": "create: field holding the month/date."},
            "entry": {"type": "object", "description": "add: the fact as JSON."},
            "field": {"type": "string", "description": "stats/forecast: numeric field."},
            "op": {"type": "string", "enum": ["sum", "avg", "min", "max", "count"]},
            "group_by": {"type": "string", "description": "month | quarter | year | a field name."},
            "filters": {"type": "object", "description": "stats/forecast: {field: value} equality filters."},
            "period": {"type": "string", "enum": list(dsl.PERIOD_KEYWORDS)},
            "period_from": {"type": "string", "description": "YYYY, YYYY-MM or YYYY-MM-DD, inclusive."},
            "period_to": {"type": "string", "description": "YYYY, YYYY-MM or YYYY-MM-DD, inclusive."},
            "include_subtree": {"type": "boolean",
                                "description": "stats: also same-named datasets in sub-folders."},
            "horizon": {"type": "integer", "description": "forecast: months ahead (default 1)."},
        },
        "required": ["action"],
    },
    risk=_datasets_risk,
    focus={"category": "list", "dataset": "dataset"},
    summary=_summary,
)
async def datasets(action: str, *, ctx=None, **kw: Any) -> ToolResult:
    cx = unwrap_ctx(ctx)
    if cx.profile_id is None:
        return ToolResult(text=t("auth.profile_required", cx.user_lang), data={"error": "no_profile"})
    await cx.progress("datasets", action)
    for k in ("entry", "filters", "fields"):  # some models send JSON as a string
        if isinstance(kw.get(k), str):
            try:
                kw[k] = json.loads(kw[k])
            except ValueError:
                kw[k] = None
    handler = _HANDLERS.get(action)
    if handler is None:
        return ToolResult(text=t("items.unknown_action", cx.user_lang, action=action),
                          data={"error": "unknown_action"})
    return await handler(cx, **kw)


# ── Resolution ────────────────────────────────────────────────────────────

async def _category(cx, name: str | None) -> dict | None:
    if name:
        return await resolve_category_by_name(name, cx.profile_id)
    return await get_last_category(cx.profile_id)


async def _dataset(cx, name: str | None, category: str | None) -> tuple[dict | None, ToolResult | None]:
    lang = cx.user_lang
    held = cx.held or {}
    if not name:
        name = held.get("dataset")
    cat = await _category(cx, category)
    if not name:
        rows = await list_datasets(cx.profile_id, cat["id"]) if cat else []
        if len(rows) == 1:
            return rows[0], None
        return None, ToolResult(text=t("datasets.which", lang), data={"error": "dataset_required"})
    ds = await resolve_dataset(cx.profile_id, name, cat["id"] if cat else None)
    if ds is None:
        return None, ToolResult(text=t("datasets.not_found", lang, name=name),
                                data={"error": "dataset_not_found", "dataset": name})
    return ds, None


def _value_fields(ds: dict) -> list[str]:
    """Required string fields — their existing values let the model pick
    filters that match what is stored ('электричество', not 'electricity')."""
    return [f["path"] for f in dsl.flatten(ds["schema"])
            if f["required"] and f["type"] == "string"][:2]


def _focus(ds: dict) -> dict:
    return {"list": ds["category_name"], "dataset": ds["name"]}


def _num(v: float | None, unit: str | None = None) -> str:
    if v is None:
        return "—"
    s = f"{v:,.2f}".rstrip("0").rstrip(".").replace(",", " ")
    return f"{s} {unit}" if unit else s


def _unit(ds: dict, path: str | None) -> str | None:
    for f in dsl.flatten(ds["schema"]):
        if f["path"] == path:
            return f["unit"]
    return None


# ── Handlers ──────────────────────────────────────────────────────────────

async def _list(cx, *, category=None, **_) -> ToolResult:
    cat = await resolve_category_by_name(category, cx.profile_id) if category else None
    if category and cat is None:
        return ToolResult(text=t("items.category_not_found", cx.user_lang, name=category),
                          data={"error": "category_not_found"})
    rows = await list_datasets(cx.profile_id, cat["id"] if cat else None)
    if not rows:
        return ToolResult(text=t("datasets.none", cx.user_lang), data={"datasets": []})
    body = "; ".join(dsl.describe(d).replace("*", "") for d in rows[:5])
    return ToolResult(text=body, data={"datasets": [{"id": d["id"], "name": d["name"],
                                                     "category": d["category_name"]} for d in rows]})


async def _create(cx, *, name=None, dataset=None, category=None, description="",
                  fields=None, amount_field=None, period_field=None, **_) -> ToolResult:
    lang = cx.user_lang
    name = (name or dataset or "").strip()
    if not name or not fields:
        return ToolResult(text=t("datasets.create_args", lang), data={"error": "missing_args"})
    try:
        schema = dsl.build_schema(fields)
    except dsl.SchemaError as e:
        return ToolResult(text=t("datasets.bad_schema", lang, error=str(e)), data={"error": "bad_schema"})
    cat = await _category(cx, category)
    if cat is None and category:
        cid = await create_category(owner_profile_id=cx.profile_id, name=category)
        cat = {"id": cid, "name": category}
    if cat is None:
        return ToolResult(text=t("datasets.which_category", lang), data={"error": "category_required"})
    existing = await resolve_dataset(cx.profile_id, name, cat["id"])
    if existing is not None and existing["category_id"] == cat["id"]:
        return ToolResult(text=t("datasets.exists", lang, name=name), data={"error": "exists"})
    g_amount, g_period = dsl.guess_hints(schema)
    amount = amount_field if amount_field and dsl.known_path(schema, amount_field) else g_amount
    period = period_field if period_field and dsl.known_path(schema, period_field) else g_period
    ds_id = await create_dataset(profile_id=cx.profile_id, category_id=cat["id"], name=name,
                                 description=description or "", schema=schema,
                                 amount_field=amount, period_field=period)
    if ds_id is None:
        return ToolResult(text=t("datasets.no_write", lang), data={"error": "forbidden"})
    await set_last_category(cx.profile_id, cat["id"])
    ds = {"name": name, "description": "", "schema": schema}
    return ToolResult(
        text=t("datasets.created", lang, name=name, category=cat["name"],
               fields=dsl.describe(ds).split(" — ", 1)[1].replace("*", "")),
        data={"dataset_id": ds_id, "focus": {"list": cat["name"], "dataset": name}},
    )


async def _add(cx, *, dataset=None, category=None, entry=None, **_) -> ToolResult:
    lang = cx.user_lang
    ds, err = await _dataset(cx, dataset, category)
    if err:
        return err
    held = cx.held if (cx.held or {}).get("dataset_id") == ds["id"] else None
    entry = dict(entry or {})
    for k, v in (("category", ds["category_name"]), ("dataset", ds["name"])):
        # Models echo tool arguments into the entry; not a new field.
        if k not in dsl.flatten_names(ds["schema"]) and str(entry.get(k, "")).lower() == v.lower():
            entry.pop(k)
    new, dropped = _ground(dsl.unflatten(entry), ds["schema"], cx.transcript)
    raw = dsl.deep_merge((held or {}).get("entry") or {}, new)
    clean, missing, unknown, invalid = dsl.validate(raw, ds["schema"])
    missing = missing + [p for p in invalid if p not in missing]
    if missing:
        return ToolResult(
            text=t("datasets.missing", lang, dataset=ds["name"], fields=", ".join(missing)),
            data={"saved": False, "missing": missing, "focus": _focus(ds),
                  "hold": {"dataset": ds["name"], "dataset_id": ds["id"],
                           "entry": raw, "missing": missing}},
        )
    if unknown:
        fields = [{"name": p, "type": _infer_type(dsl.get_path(raw, p))} for p in unknown]
        args = {"action": "extend", "dataset": ds["name"], "category": ds["category_name"],
                "fields": fields, "entry": raw}
        try:
            action_id = await enqueue_action(profile_id=cx.profile_id, client_id=cx.client_id,
                                             tool_name="datasets", tool_args=args,
                                             summary=_summary(args))
        except Exception:
            log.exception("datasets: enqueue extend failed")
            return ToolResult(text=t("auth.queue_error", lang), data={"error": "enqueue_failed"})
        return ToolResult(
            text=t("datasets.propose_extend", lang, dataset=ds["name"], fields=", ".join(unknown)),
            data={"saved": False, "unknown": unknown, "pending_action_id": action_id,
                  "focus": _focus(ds), "hold": None},
        )
    entry_id = await add_entry(cx.profile_id, ds["id"], clean)
    if entry_id is None:
        return ToolResult(text=t("datasets.no_write", lang), data={"error": "forbidden"})
    await set_last_category(cx.profile_id, ds["category_id"])
    shown = ", ".join(f"{p}={dsl.get_path(clean, p)}" for p in dsl.leaf_paths(clean)[:6])
    return ToolResult(text=t("datasets.saved", lang, dataset=ds["name"], entry=shown),
                      data={"saved": True, "entry_id": entry_id, "entry": clean,
                            "focus": _focus(ds), "hold": None})


def _ground(entry: dict, schema: dict, transcript: str) -> tuple[dict, list[str]]:
    """Drop values the model made up — a month/date when the utterance names
    no time, an optional text value that shares no word with it — and set
    relative periods ("last month") from the utterance.  Without a
    transcript (pending replays, tests) the entry is kept as is."""
    if not transcript:
        return entry, []
    fields = {f["path"]: f for f in dsl.flatten(schema)}
    rel = dsl.relative_period(transcript)
    for f in fields.values():
        if f["type"] in rel and "." not in f["path"]:
            entry[f["path"]] = rel[f["type"]]  # code beats the model's arithmetic
    dropped = []
    for p in dsl.leaf_paths(entry):
        f = fields.get(p)
        if f is None:
            continue
        v = dsl.get_path(entry, p)
        if (f["type"] in ("month", "date") and not dsl.has_period_cue(transcript)) or (
                f["type"] == "string" and not f["required"] and not dsl.grounded(v, transcript)):
            dropped.append(p)
    for p in dropped:
        *parents, leaf = p.split(".")
        node = entry
        for q in parents:
            node = node.get(q, {})
        node.pop(leaf, None)
    if dropped:
        log.info("datasets: dropped ungrounded %s", dropped)
    return entry, dropped


def _infer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if dsl.normalize_month(v) and len(str(v)) <= 7:
        return "month"
    return "string"


def _filters(ds: dict, filters: dict | None) -> tuple[list[tuple[str, Any]], list[str]]:
    flat = dsl.unflatten(filters or {})
    pairs = [(p, dsl.get_path(flat, p)) for p in dsl.leaf_paths(flat)]
    bad = [p for p, _ in pairs if not dsl.known_path(ds["schema"], p)]
    return [(p, v) for p, v in pairs if p not in bad], bad


async def _scope(cx, ds: dict, include_subtree: bool) -> list[int]:
    if not include_subtree:
        return [ds["id"]]
    return [d["id"] for d in await subtree_datasets(cx.profile_id, ds)]


async def _stats(cx, *, dataset=None, category=None, field=None, op="sum", group_by=None,
                 filters=None, period=None, period_from=None, period_to=None,
                 include_subtree=False, **_) -> ToolResult:
    lang = cx.user_lang
    ds, err = await _dataset(cx, dataset, category)
    if err:
        return err
    op = op if op in ("sum", "avg", "min", "max", "count") else "sum"
    field = field or ds["amount_field"]
    if op != "count" and (not field or not dsl.known_path(ds["schema"], field)):
        return ToolResult(text=t("datasets.unknown_field", lang, field=field or "?", dataset=ds["name"]),
                          data={"error": "unknown_field"})
    flt, bad = _filters(ds, filters)
    if bad:
        return ToolResult(text=t("datasets.unknown_field", lang, field=", ".join(bad), dataset=ds["name"]),
                          data={"error": "unknown_field"})
    p_from, p_to = dsl.period_bounds(period)
    p_from, p_to = period_from or p_from, period_to or p_to
    group_path = None
    if group_by and group_by not in ("month", "quarter", "year"):
        if not dsl.known_path(ds["schema"], group_by):
            return ToolResult(text=t("datasets.unknown_field", lang, field=group_by, dataset=ds["name"]),
                              data={"error": "unknown_field"})
        group_path = group_by
    if (p_from or p_to or group_by in ("month", "quarter", "year")) and not ds["period_field"]:
        return ToolResult(text=t("datasets.no_period", lang, dataset=ds["name"]), data={"error": "no_period"})
    rows = await aggregate(
        await _scope(cx, ds, include_subtree), op=op, value_path=field,
        group=group_by, group_path=group_path, period_path=ds["period_field"],
        filters=flt, period_from=p_from, period_to=p_to,
    )
    unit = None if op == "count" else _unit(ds, field)
    n = sum(r[2] for r in rows)
    data = {"op": op, "field": field, "group_by": group_by, "period_from": p_from,
            "period_to": p_to, "filters": dict(flt), "entries": n,
            "rows": [{"group": r[0], "value": r[1], "n": r[2]} for r in rows],
            "focus": _focus(ds)}
    if n == 0:
        return ToolResult(text=t("datasets.stats_empty", lang, dataset=ds["name"]), data=data)
    scope = _scope_text(flt, p_from, p_to)
    if group_by:
        body = "; ".join(f"{r[0]}: {_num(r[1], unit)}" for r in rows[:12])
        text = t("datasets.stats_grouped", lang, op=t(f"datasets.op.{op}", lang), field=field,
                 scope=scope, body=body)
    else:
        data["value"] = rows[0][1]
        text = t("datasets.stats_total", lang, op=t(f"datasets.op.{op}", lang), field=field,
                 scope=scope, value=_num(rows[0][1], unit), n=n)
    return ToolResult(text=text, data=data)


def _scope_text(flt: list[tuple[str, Any]], p_from: str | None, p_to: str | None) -> str:
    bits = [f"{v}" for _, v in flt]
    if p_from or p_to:
        bits.append(p_from if p_from == p_to else f"{p_from or '…'}–{p_to or '…'}")
    return f" ({', '.join(bits)})" if bits else ""


async def _forecast(cx, *, dataset=None, category=None, field=None, filters=None,
                    horizon=1, include_subtree=False, **_) -> ToolResult:
    lang = cx.user_lang
    ds, err = await _dataset(cx, dataset, category)
    if err:
        return err
    field = field or ds["amount_field"]
    if not field or not dsl.known_path(ds["schema"], field):
        return ToolResult(text=t("datasets.unknown_field", lang, field=field or "?", dataset=ds["name"]),
                          data={"error": "unknown_field"})
    if not ds["period_field"]:
        return ToolResult(text=t("datasets.no_period", lang, dataset=ds["name"]), data={"error": "no_period"})
    flt, bad = _filters(ds, filters)
    if bad:
        return ToolResult(text=t("datasets.unknown_field", lang, field=", ".join(bad), dataset=ds["name"]),
                          data={"error": "unknown_field"})
    rows = await aggregate(
        await _scope(cx, ds, include_subtree), op="sum", value_path=field, group="month",
        group_path=None, period_path=ds["period_field"], filters=flt,
        period_from=None, period_to=None,
    )
    fc = dsl.forecast_monthly([(r[0], r[1]) for r in rows if r[0]], horizon=min(max(int(horizon or 1), 1), 12))
    fc["focus"] = _focus(ds)
    if not fc["enough"]:
        return ToolResult(text=t("datasets.forecast_not_enough", lang, dataset=ds["name"],
                                 have=fc["periods"], need=fc["needed"]), data=fc)
    unit = _unit(ds, field)
    body = "; ".join(f"{p['period']}: ~{_num(p['value'], unit)} ({_num(p['low'])}–{_num(p['high'])})"
                     for p in fc["points"])
    return ToolResult(text=t("datasets.forecast", lang, field=field, scope=_scope_text(flt, None, None),
                             body=body, periods=fc["periods"]), data=fc)


async def _extend(cx, *, dataset=None, category=None, fields=None, entry=None, **_) -> ToolResult:
    lang = cx.user_lang
    ds, err = await _dataset(cx, dataset, category)
    if err:
        return err
    if not fields:
        return ToolResult(text=t("datasets.create_args", lang), data={"error": "missing_args"})
    try:
        schema = dsl.merge_schema(ds["schema"], fields)
    except dsl.SchemaError as e:
        return ToolResult(text=t("datasets.bad_schema", lang, error=str(e)), data={"error": "bad_schema"})
    if not await update_dataset_schema(cx.profile_id, ds["id"], schema):
        return ToolResult(text=t("datasets.owner_only", lang), data={"error": "forbidden"})
    names = ", ".join(f.get("name", "?") for f in fields)
    text = t("datasets.extended", lang, dataset=ds["name"], fields=names)
    if not entry:
        return ToolResult(text=text, data={"dataset_id": ds["id"], "focus": _focus(ds)})
    clean, missing, unknown, invalid = dsl.validate(entry, schema)
    if missing or unknown or invalid:
        return ToolResult(text=text + " " + t("datasets.missing", lang, dataset=ds["name"],
                                              fields=", ".join(missing + unknown + invalid)),
                          data={"dataset_id": ds["id"], "saved": False, "focus": _focus(ds)})
    entry_id = await add_entry(cx.profile_id, ds["id"], clean)
    return ToolResult(text=text + " " + t("datasets.saved_short", lang),
                      data={"dataset_id": ds["id"], "saved": entry_id is not None,
                            "entry_id": entry_id, "focus": _focus(ds)})


async def _delete(cx, *, dataset=None, category=None, **_) -> ToolResult:
    lang = cx.user_lang
    ds, err = await _dataset(cx, dataset, category)
    if err:
        return err
    if not await delete_dataset(cx.profile_id, ds["id"]):
        return ToolResult(text=t("datasets.owner_only", lang), data={"error": "forbidden"})
    return ToolResult(text=t("datasets.deleted", lang, dataset=ds["name"]),
                      data={"dataset_id": ds["id"], "hold": None})


_HANDLERS = {
    "list": _list, "create": _create, "add": _add, "stats": _stats,
    "forecast": _forecast, "extend": _extend, "delete": _delete,
}


# ── Context for the focused category ──────────────────────────────────────

async def context_block(profile_id: int, category_name: str | None,
                        held: dict | None = None, today: _dt.date | None = None) -> str:
    """One bracketed line for the agent: the focused category's datasets
    and, if any, the entry still waiting for fields.  Empty when the
    category has no datasets and nothing is held."""
    parts = []
    if category_name:
        cat = await resolve_category_by_name(category_name, profile_id)
        rows = await list_datasets(profile_id, cat["id"]) if cat else []
        if rows:
            descs = []
            for d in rows[:4]:
                vals = await distinct_values(d["id"], _value_fields(d))
                descs.append(dsl.describe(d, values=vals))
            parts.append(f"Datasets in «{cat['name']}» (* required): " + "; ".join(descs)
                         + ". For datasets add, copy only values the user said — never "
                         "fill a missing field yourself, the runtime asks; a value that "
                         "fits no field gets its own new key")
    if held:
        parts.append(
            f"Unsaved entry for «{held['dataset']}» still needs: {', '.join(held['missing'])}. "
            "If the message supplies them, call datasets add with only the new values"
        )
    if not parts:
        return ""
    d = today or _dt.date.today()
    return "[" + ". ".join(parts) + f". Today is {d.isoformat()}.]"
