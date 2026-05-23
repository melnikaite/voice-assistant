"""LLM token usage persistence — insert + daily/per-tool/per-user aggregates.

Each row is a single ``/v1/chat/completions`` call: prompt_tokens,
completion_tokens, optional reasoning_tokens, plus context for billing
attribution (client_id, model, tool_name) and observability (elapsed_ms).

Aggregations are intentionally plain SQL — the dataset is small (one row
per LLM call, retained forever) and queries run on demand from the
``/api/stats`` endpoint.  No materialised views, no caching: SQLite
handles tens of thousands of rows instantly.
"""
from __future__ import annotations

import asyncio
import time

from .db import _conn, _lock


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _add_token_usage_sync(
    ts: float,
    client_id: str | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int | None,
    tool_name: str | None,
    elapsed_ms: int | None,
) -> None:
    with _lock:
        c = _conn()
        try:
            c.execute(
                "INSERT INTO token_usage("
                "ts, client_id, model, prompt_tokens, completion_tokens, "
                "reasoning_tokens, tool_name, elapsed_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    client_id,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    tool_name,
                    elapsed_ms,
                ),
            )
        finally:
            c.close()


async def add_token_usage(
    *,
    client_id: str | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int | None = None,
    tool_name: str | None = None,
    elapsed_ms: int | None = None,
) -> None:
    """Record one LLM call's token usage.  Fire-and-forget at call sites."""
    await asyncio.to_thread(
        _add_token_usage_sync,
        time.time(),
        client_id,
        model,
        int(prompt_tokens or 0),
        int(completion_tokens or 0),
        int(reasoning_tokens) if reasoning_tokens is not None else None,
        tool_name,
        int(elapsed_ms) if elapsed_ms is not None else None,
    )


# ---------------------------------------------------------------------------
# Read — aggregations
# ---------------------------------------------------------------------------

def _get_daily_usage_sync(days: int) -> list[dict]:
    """Daily prompt/completion totals over the last ``days`` days, oldest→newest.

    Date bucketing uses SQLite ``date(ts, 'unixepoch', 'localtime')`` so the
    rows group by local-day boundaries (the orchestrator's TZ is set to
    Europe/Berlin via the container env).  This matches what a human looking
    at "yesterday's usage" expects.
    """
    since = time.time() - days * 86400
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                """
                SELECT date(ts, 'unixepoch', 'localtime') AS day,
                       SUM(prompt_tokens)     AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(COALESCE(reasoning_tokens, 0)) AS reasoning_tokens,
                       COUNT(*)               AS calls
                FROM   token_usage
                WHERE  ts > ?
                GROUP  BY day
                ORDER  BY day ASC
                """,
                (since,),
            ).fetchall()
            return [
                {
                    "day": row[0],
                    "prompt_tokens": int(row[1] or 0),
                    "completion_tokens": int(row[2] or 0),
                    "reasoning_tokens": int(row[3] or 0),
                    "calls": int(row[4] or 0),
                }
                for row in rows
            ]
        finally:
            c.close()


async def get_daily_usage(days: int = 30) -> list[dict]:
    return await asyncio.to_thread(_get_daily_usage_sync, days)


# Cap the per-tool / per-user dashboards.  Without a cap, every smoke-
# test client_id from CI lifetime ends up in a horizontal bar chart, and
# the chart canvas grows to ~40 px per row — easily a dozen full screens
# tall.  Top-N keeps the UI scannable and rolls the long tail into one
# explicit "etc (N)" bucket so the displayed totals still add up to the
# truth.
_DASHBOARD_TOP_N = 8


def _collapse_long_tail(rows: list[dict], label_key: str, etc_label: str) -> list[dict]:
    """Keep the top _DASHBOARD_TOP_N entries; sum the rest into one bucket.

    Rows are assumed sorted by total tokens DESC by the SQL query.
    Returns a NEW list — never mutates the input.
    """
    if len(rows) <= _DASHBOARD_TOP_N:
        return rows
    head = rows[:_DASHBOARD_TOP_N]
    tail = rows[_DASHBOARD_TOP_N:]
    summed: dict = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 0,
    }
    for r in tail:
        summed["prompt_tokens"]     += r.get("prompt_tokens", 0)
        summed["completion_tokens"] += r.get("completion_tokens", 0)
        summed["reasoning_tokens"]  += r.get("reasoning_tokens", 0)
        summed["calls"]             += r.get("calls", 0)
    summed[label_key] = f"{etc_label} ({len(tail)})"
    return head + [summed]


def _get_per_tool_usage_sync(days: int) -> list[dict]:
    since = time.time() - days * 86400
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                """
                SELECT COALESCE(tool_name, '(unknown)') AS tool,
                       SUM(prompt_tokens)     AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(COALESCE(reasoning_tokens, 0)) AS reasoning_tokens,
                       COUNT(*)               AS calls
                FROM   token_usage
                WHERE  ts > ?
                GROUP  BY tool
                ORDER  BY (SUM(prompt_tokens) + SUM(completion_tokens)) DESC
                """,
                (since,),
            ).fetchall()
            return [
                {
                    "tool_name": row[0],
                    "prompt_tokens": int(row[1] or 0),
                    "completion_tokens": int(row[2] or 0),
                    "reasoning_tokens": int(row[3] or 0),
                    "calls": int(row[4] or 0),
                }
                for row in rows
            ]
        finally:
            c.close()


async def get_per_tool_usage(days: int = 7) -> list[dict]:
    rows = await asyncio.to_thread(_get_per_tool_usage_sync, days)
    return _collapse_long_tail(rows, "tool_name", "other tools")


def _get_per_user_usage_sync(days: int) -> list[dict]:
    since = time.time() - days * 86400
    with _lock:
        c = _conn()
        try:
            # COALESCE preserves NULL client_ids as a labelled "(anon)" bucket
            # rather than dropping them — keeps the totals honest when a
            # caller (e.g. /dev/respond without explicit client_id) didn't
            # plumb one through.
            rows = c.execute(
                """
                SELECT COALESCE(client_id, '(anon)') AS who,
                       SUM(prompt_tokens)     AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(COALESCE(reasoning_tokens, 0)) AS reasoning_tokens,
                       COUNT(*)               AS calls
                FROM   token_usage
                WHERE  ts > ?
                GROUP  BY who
                ORDER  BY (SUM(prompt_tokens) + SUM(completion_tokens)) DESC
                """,
                (since,),
            ).fetchall()
            return [
                {
                    "client_id": row[0],
                    "prompt_tokens": int(row[1] or 0),
                    "completion_tokens": int(row[2] or 0),
                    "reasoning_tokens": int(row[3] or 0),
                    "calls": int(row[4] or 0),
                }
                for row in rows
            ]
        finally:
            c.close()


async def get_per_user_usage(days: int = 30) -> list[dict]:
    rows = await asyncio.to_thread(_get_per_user_usage_sync, days)
    return _collapse_long_tail(rows, "client_id", "other clients")


# ---------------------------------------------------------------------------
# Rolling-window GC — called from the periodic scheduler tick in gc.py
# ---------------------------------------------------------------------------


def _purge_old_sync(cutoff_ts: float) -> int:
    """Delete rows older than ``cutoff_ts`` (unix seconds).  Returns rowcount.

    ``token_usage`` retains forever by default; this helper is the
    only thing that prunes it.  Called from ``gc.run_all`` on a
    timer.  All stats endpoints query within the retention window so
    pruned rows are invisible to callers.
    """
    with _lock:
        c = _conn()
        try:
            cur = c.execute(
                "DELETE FROM token_usage WHERE ts < ?", (cutoff_ts,)
            )
            return cur.rowcount
        finally:
            c.close()


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------

# Per-million-token rates ($/MTok).  Local Gemma runs free; the two cloud
# entries are "what would this have cost" projections.  Sonnet/Mini pair
# spans the price spectrum (50× input ratio) so the UI can show both an
# upper and a lower bound.
#
# Loaded from ``PRICING_PATH`` (default ``/data/pricing.json``) when the
# module imports — so an operator can update rates without rebuilding
# the image, and the defaults below act as the fallback when the file
# is missing or malformed.  The dict shape:
#
#     { "<model-id>": { "prompt": <$/MTok>, "completion": <$/MTok> }, ... }
#
# Add or override entries as cloud pricing changes.  This dict drives
# the projected-cost figures in /api/stats and the legend in the UI;
# nothing here affects routing or which model actually serves a call.

_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4.5": {"prompt": 3.0,  "completion": 15.0},
    "gpt-4o-mini":       {"prompt": 0.15, "completion": 0.60},
    "gemma-local":       {"prompt": 0.0,  "completion": 0.0},
}

import json as _json
import os as _os
_PRICING_PATH = _os.environ.get("PRICING_PATH", "/data/pricing.json")


def _load_pricing() -> dict[str, dict[str, float]]:
    """Read ``PRICING_PATH`` if present; fall back to defaults on any error.

    Validates only the shape — each entry must be ``{prompt: float,
    completion: float}``.  Unknown extra keys are ignored.  A missing
    or malformed file logs a warning once and uses defaults.
    """
    try:
        with open(_PRICING_PATH, encoding="utf-8") as f:
            raw = _json.load(f)
    except FileNotFoundError:
        return dict(_DEFAULT_PRICING)
    except (OSError, _json.JSONDecodeError) as exc:
        import logging
        logging.getLogger(__name__).warning(
            "pricing: failed to load %s (%s) — using defaults",
            _PRICING_PATH, exc,
        )
        return dict(_DEFAULT_PRICING)
    out: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        return dict(_DEFAULT_PRICING)
    for model, rates in raw.items():
        if not isinstance(rates, dict):
            continue
        try:
            out[str(model)] = {
                "prompt": float(rates["prompt"]),
                "completion": float(rates["completion"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out or dict(_DEFAULT_PRICING)


PRICING: dict[str, dict[str, float]] = _load_pricing()


def compute_projected_cost(daily: list[dict]) -> dict[str, dict]:
    """Sum tokens across the daily buckets, project costs for each pricing tier.

    Returns ``{model: {prompt_tokens, completion_tokens, cost_usd}}``.  The
    frontend formats this — we keep raw floats here so the cost figure is
    reproducible client-side if needed.
    """
    total_prompt = sum(d["prompt_tokens"] for d in daily)
    total_completion = sum(d["completion_tokens"] for d in daily)
    out: dict[str, dict] = {}
    for model, rates in PRICING.items():
        cost = (
            total_prompt * rates["prompt"] / 1_000_000.0
            + total_completion * rates["completion"] / 1_000_000.0
        )
        out[model] = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "prompt_rate_per_mtok": rates["prompt"],
            "completion_rate_per_mtok": rates["completion"],
            "cost_usd": round(cost, 6),
        }
    return out
