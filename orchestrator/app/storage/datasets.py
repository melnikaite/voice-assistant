"""
Dataset storage — JSON datasets owned by a category, one row per entry.

Access follows the owning category and is re-checked on every call:
the category owner has full rights, a ``category_shares`` grant on the
category or any ancestor gives ``read`` or ``write`` (write = add and
delete entries).  Schema changes and dataset deletion are owner-only.
Deleted categories hide their datasets.

Stats run in SQLite over ``json_extract``; paths are bound parameters
built from fields the schema knows, never from free text.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .categories import _slugify
from .db import _conn

_RANK = {"read": 1, "write": 2, "owner": 3}

_COLS = (
    "d.id, d.category_id, d.owner_profile_id, d.name, d.slug, d.description, "
    "d.schema_json, d.amount_field, d.period_field, d.created_at, d.updated_at, "
    "c.name"
)


def _row(r: tuple, perm: str) -> dict:
    return {
        "id": r[0], "category_id": r[1], "owner_profile_id": r[2], "name": r[3],
        "slug": r[4], "description": r[5], "schema": json.loads(r[6]),
        "amount_field": r[7], "period_field": r[8], "created_at": r[9],
        "updated_at": r[10], "category_name": r[11], "permission": perm,
    }


def json_path(path: str) -> str:
    return "$" + "".join('."' + p.replace('"', "") + '"' for p in path.split("."))


# ── Access ────────────────────────────────────────────────────────────────

def _access_sync(profile_id: int) -> dict[int, str]:
    """category_id → best permission for ``profile_id`` (owner/write/read)."""
    rows = _conn().execute(
        """
        WITH RECURSIVE acc(id, perm) AS (
            SELECT id, 'owner' FROM categories
              WHERE owner_profile_id = ? AND deleted_at IS NULL
            UNION
            SELECT s.category_id, s.permission FROM category_shares s
              JOIN categories c ON c.id = s.category_id
              WHERE s.profile_id = ? AND c.deleted_at IS NULL
            UNION
            SELECT c.id, acc.perm FROM categories c JOIN acc ON c.parent_id = acc.id
              WHERE c.deleted_at IS NULL
        )
        SELECT id, perm FROM acc
        """,
        (profile_id, profile_id),
    ).fetchall()
    out: dict[int, str] = {}
    for cid, perm in rows:
        if _RANK.get(perm, 0) > _RANK.get(out.get(cid, ""), 0):
            out[cid] = perm
    return out


def _select(profile_id: int, where: str, params: tuple) -> list[dict]:
    acc = _access_sync(profile_id)
    if not acc:
        return []
    marks = ",".join("?" * len(acc))
    rows = _conn().execute(
        f"SELECT {_COLS} FROM datasets d JOIN categories c ON c.id = d.category_id"
        f"  WHERE d.deleted_at IS NULL AND d.category_id IN ({marks}) AND {where}"
        "   ORDER BY d.name",
        (*acc.keys(), *params),
    ).fetchall()
    return [_row(r, acc[r[1]]) for r in rows]


# ── Datasets ──────────────────────────────────────────────────────────────

def _create_sync(category_id, owner_profile_id, name, description, schema,
                 amount_field, period_field) -> int:
    now = time.time()
    cur = _conn().execute(
        "INSERT INTO datasets (category_id, owner_profile_id, name, slug, description,"
        "  schema_json, amount_field, period_field, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (category_id, owner_profile_id, name, _slugify(name), description or "",
         json.dumps(schema, ensure_ascii=False), amount_field, period_field, now, now),
    )
    return cur.lastrowid


async def create_dataset(*, profile_id: int, category_id: int, name: str,
                         description: str, schema: dict,
                         amount_field: str | None, period_field: str | None) -> int | None:
    """Owner of the category only.  Returns the id, or None without access."""
    def run():
        if _access_sync(profile_id).get(category_id) != "owner":
            return None
        return _create_sync(category_id, profile_id, name, description, schema,
                            amount_field, period_field)
    return await asyncio.to_thread(run)


async def list_datasets(profile_id: int, category_id: int | None = None) -> list[dict]:
    if category_id is None:
        return await asyncio.to_thread(_select, profile_id, "1", ())
    return await asyncio.to_thread(_select, profile_id, "d.category_id = ?", (category_id,))


async def get_dataset(profile_id: int, dataset_id: int) -> dict | None:
    rows = await asyncio.to_thread(_select, profile_id, "d.id = ?", (dataset_id,))
    return rows[0] if rows else None


async def resolve_dataset(profile_id: int, name: str,
                          prefer_category_id: int | None = None) -> dict | None:
    """Name or slug match among accessible datasets; the preferred
    category wins, then own datasets."""
    rows = await asyncio.to_thread(
        _select, profile_id, "(LOWER(d.name) = LOWER(?) OR d.slug = ?)",
        (name.strip(), _slugify(name)),
    )
    if not rows:
        return None
    rows.sort(key=lambda d: (d["category_id"] != prefer_category_id,
                             d["owner_profile_id"] != profile_id))
    return rows[0]


async def update_dataset_schema(profile_id: int, dataset_id: int, schema: dict,
                                *, amount_field: str | None = None,
                                period_field: str | None = None) -> bool:
    def run():
        ds = _select(profile_id, "d.id = ?", (dataset_id,))
        if not ds or ds[0]["permission"] != "owner":
            return False
        _conn().execute(
            "UPDATE datasets SET schema_json = ?, updated_at = ?,"
            "  amount_field = COALESCE(?, amount_field),"
            "  period_field = COALESCE(?, period_field)"
            " WHERE id = ?",
            (json.dumps(schema, ensure_ascii=False), time.time(),
             amount_field, period_field, dataset_id),
        )
        return True
    return await asyncio.to_thread(run)


async def delete_dataset(profile_id: int, dataset_id: int) -> bool:
    """Soft-delete; entries stay but become unreachable."""
    def run():
        ds = _select(profile_id, "d.id = ?", (dataset_id,))
        if not ds or ds[0]["permission"] != "owner":
            return False
        _conn().execute("UPDATE datasets SET deleted_at = ? WHERE id = ?",
                        (time.time(), dataset_id))
        return True
    return await asyncio.to_thread(run)


# ── Entries ───────────────────────────────────────────────────────────────

async def add_entry(profile_id: int, dataset_id: int, data: dict) -> int | None:
    """Caller validated ``data``.  None without write access."""
    def run():
        ds = _select(profile_id, "d.id = ?", (dataset_id,))
        if not ds or _RANK[ds[0]["permission"]] < _RANK["write"]:
            return None
        cur = _conn().execute(
            "INSERT INTO dataset_entries (dataset_id, created_by_profile_id, data, created_at)"
            " VALUES (?, ?, ?, ?)",
            (dataset_id, profile_id, json.dumps(data, ensure_ascii=False), time.time()),
        )
        return cur.lastrowid
    return await asyncio.to_thread(run)


async def list_entries(profile_id: int, dataset_id: int, limit: int = 100) -> list[dict] | None:
    def run():
        ds = _select(profile_id, "d.id = ?", (dataset_id,))
        if not ds:
            return None
        # Field names come from the model: bind the path, never inline it.
        period = ds[0]["period_field"]
        rows = _conn().execute(
            "SELECT id, data, created_by_profile_id, created_at FROM dataset_entries"
            " WHERE dataset_id = ? AND deleted_at IS NULL"
            " ORDER BY json_extract(data, ?) DESC, created_at DESC LIMIT ?",
            (dataset_id, json_path(period) if period else "$.__none__", limit),
        ).fetchall()
        return [{"id": r[0], "data": json.loads(r[1]), "created_by_profile_id": r[2],
                 "created_at": r[3]} for r in rows]
    return await asyncio.to_thread(run)


async def delete_entry(profile_id: int, entry_id: int) -> bool:
    def run():
        row = _conn().execute(
            "SELECT dataset_id FROM dataset_entries WHERE id = ? AND deleted_at IS NULL",
            (entry_id,),
        ).fetchone()
        if not row:
            return False
        ds = _select(profile_id, "d.id = ?", (row[0],))
        if not ds or _RANK[ds[0]["permission"]] < _RANK["write"]:
            return False
        _conn().execute("UPDATE dataset_entries SET deleted_at = ? WHERE id = ?",
                        (time.time(), entry_id))
        return True
    return await asyncio.to_thread(run)


async def subtree_datasets(profile_id: int, ds: dict) -> list[dict]:
    """Same-named datasets in the subtree of ``ds``'s category (inclusive)."""
    def run():
        ids = {r[0] for r in _conn().execute(
            "WITH RECURSIVE sub(id) AS (SELECT ? UNION ALL"
            "  SELECT c.id FROM categories c JOIN sub ON c.parent_id = sub.id"
            "  WHERE c.deleted_at IS NULL) SELECT id FROM sub",
            (ds["category_id"],),
        ).fetchall()}
        return [d for d in _select(profile_id, "d.slug = ?", (ds["slug"],))
                if d["category_id"] in ids]
    return await asyncio.to_thread(run)


def _distinct_sync(dataset_id: int, paths: list[str], limit: int) -> dict[str, list]:
    out = {}
    for p in paths:
        rows = _conn().execute(
            "SELECT json_extract(data, ?) AS v, COUNT(*) AS n FROM dataset_entries"
            " WHERE dataset_id = ? AND deleted_at IS NULL AND v IS NOT NULL"
            " GROUP BY v ORDER BY n DESC LIMIT ?",
            (json_path(p), dataset_id, limit),
        ).fetchall()
        if rows:
            out[p] = [r[0] for r in rows]
    return out


async def distinct_values(dataset_id: int, paths: list[str], limit: int = 5) -> dict[str, list]:
    """Most frequent stored values per field.  No access check: callers
    pass ids of datasets they already resolved for the profile."""
    if not paths:
        return {}
    return await asyncio.to_thread(_distinct_sync, dataset_id, paths, limit)


# ── Aggregation ───────────────────────────────────────────────────────────

_OPS = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}


def _aggregate_sync(dataset_ids: list[int], *, op: str, value_path: str | None,
                    group: str | None, group_path: str | None, period_path: str | None,
                    filters: list[tuple[str, Any]], period_from: str | None,
                    period_to: str | None) -> list[tuple[Any, float | None, int]]:
    params: list[Any] = []
    if op == "count" or not value_path:
        agg = "COUNT(*)"
    else:
        agg = f"{_OPS[op]}(CAST(json_extract(data, ?) AS REAL))"
        params.append(json_path(value_path))
    key = "NULL"
    key_params: list[Any] = []
    if group in ("month", "quarter", "year") and period_path:
        p = "json_extract(data, ?)"
        key = {
            "month": f"substr({p}, 1, 7)",
            "year": f"substr({p}, 1, 4)",
            "quarter": f"substr({p}, 1, 4) || '-Q' || ((CAST(substr({p}, 6, 2) AS INTEGER) + 2) / 3)",
        }[group]
        key_params = [json_path(period_path)] * key.count("?")
    elif group_path:
        key = "json_extract(data, ?)"
        key_params = [json_path(group_path)]
    conds = [f"dataset_id IN ({','.join('?' * len(dataset_ids))})", "deleted_at IS NULL"]
    cparams: list[Any] = list(dataset_ids)
    for path, val in filters:
        conds.append("LOWER(CAST(json_extract(data, ?) AS TEXT)) = LOWER(?)")
        cparams += [json_path(path), str(val)]
    if period_path and period_from:
        conds.append("substr(json_extract(data, ?), 1, ?) >= ?")
        cparams += [json_path(period_path), len(period_from), period_from]
    if period_path and period_to:
        conds.append("substr(json_extract(data, ?), 1, ?) <= ?")
        cparams += [json_path(period_path), len(period_to), period_to]
    if op != "count" and value_path:
        conds.append("json_extract(data, ?) IS NOT NULL")
        cparams.append(json_path(value_path))
    sql = (f"SELECT {key} AS k, {agg}, COUNT(*) FROM dataset_entries"
           f" WHERE {' AND '.join(conds)} GROUP BY k ORDER BY k")
    return _conn().execute(sql, (*key_params, *params, *cparams)).fetchall()


async def aggregate(dataset_ids: list[int], **kw) -> list[tuple[Any, float | None, int]]:
    """``[(group_key, value, n)]``; one row with key None when ungrouped.
    Callers pass ids from ``get_dataset`` / ``resolve_dataset`` only."""
    if not dataset_ids:
        return []
    return await asyncio.to_thread(_aggregate_sync, dataset_ids, **kw)
