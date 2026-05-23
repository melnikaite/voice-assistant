"""
Personal item storage — links, text, videos, screenshots, checklist items.

Media files live under $DATA_DIR_CONTAINER/items/<id>.<ext>.  Embeddings
are stored as float32 BLOBs and populated asynchronously by the ingest
pipeline — items are visible immediately but not semantically searchable
until the embedding arrives (~1-2 s after save).

All mutating helpers are scoped by owner_profile_id so callers don't
have to re-check ownership in the tool layer.

The module is intentionally thin: it owns SQL and disk I/O only.
Higher-level logic (URL fetching, LLM summarisation, embedding) lives
in app/items/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .db import _conn

log = logging.getLogger(__name__)

_DATA_DIR = Path(os.environ.get("DATA_DIR_CONTAINER", "/data"))
ITEMS_DIR = _DATA_DIR / "items"


def _ensure_dir() -> None:
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)


# ── Row helpers ──────────────────────────────────────────────────────────

_SELECT_COLS = (
    "id, owner_profile_id, created_by_profile_id, category_id, kind, status, "
    "title, summary, url, media_path, source_meta, body, embedding, "
    "sort_order, completed_at, created_at, deleted_at"
)


def _row_to_dict(r: tuple) -> dict:
    return {
        "id": r[0],
        "owner_profile_id": r[1],
        "created_by_profile_id": r[2],
        "category_id": r[3],
        "kind": r[4],
        "status": r[5],
        "title": r[6],
        "summary": r[7],
        "url": r[8],
        "media_path": r[9],
        "source_meta": r[10],
        "body": r[11],
        "embedding": r[12],          # bytes — callers decode as needed
        "sort_order": r[13],
        "completed_at": r[14],
        "created_at": r[15],
        "deleted_at": r[16],
    }


# ── Insert ───────────────────────────────────────────────────────────────

def _create_sync(
    *,
    owner_profile_id: int,
    created_by_profile_id: int,
    category_id: int | None,
    kind: str,
    title: str | None,
    url: str | None,
    body: str | None,
    source_meta: str | None,
    status: str,
    sort_order: float,
) -> int:
    now = time.time()
    c = _conn()
    cur = c.execute(
        """
        INSERT INTO items
          (owner_profile_id, created_by_profile_id, category_id, kind,
           status, title, url, body, source_meta, sort_order, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner_profile_id, created_by_profile_id, category_id, kind,
            status, title, url, body, source_meta, sort_order, now,
        ),
    )
    if not cur.lastrowid:
        raise RuntimeError("INSERT returned no lastrowid")
    return cur.lastrowid


async def create_item(
    *,
    owner_profile_id: int,
    created_by_profile_id: int,
    category_id: int | None = None,
    kind: str,
    title: str | None = None,
    url: str | None = None,
    body: str | None = None,
    source_meta: str | None = None,
    status: str = "active",
    sort_order: float = 0.0,
) -> int:
    """Insert a new item row.  Returns the new row id.

    ``source_meta`` must be a JSON string (or None) encoding platform-
    specific metadata: ``{"domain": "...", "video_id": "...", ...}``.
    Embeddings, summaries, and media_path are filled in separately by
    the ingest pipeline via :func:`set_item_embedding` etc.
    """
    return await asyncio.to_thread(
        _create_sync,
        owner_profile_id=owner_profile_id,
        created_by_profile_id=created_by_profile_id,
        category_id=category_id,
        kind=kind,
        title=title,
        url=url,
        body=body,
        source_meta=source_meta,
        status=status,
        sort_order=sort_order,
    )


# ── Read ─────────────────────────────────────────────────────────────────

def _get_one_sync(item_id: int) -> tuple | None:
    c = _conn()
    return c.execute(
        f"SELECT {_SELECT_COLS} FROM items WHERE id=?", (item_id,)
    ).fetchone()


async def get_item(item_id: int) -> dict | None:
    """Fetch one item by id (no ownership check — caller must verify).

    Callers MUST check ``row['owner_profile_id'] == requesting_profile_id``
    before exposing the result to a user.
    """
    row = await asyncio.to_thread(_get_one_sync, item_id)
    return _row_to_dict(row) if row else None


def _batch_get_sync(item_ids: list[int]) -> list[tuple]:
    if not item_ids:
        return []
    placeholders = ",".join("?" * len(item_ids))
    c = _conn()
    return c.execute(
        f"SELECT {_SELECT_COLS} FROM items WHERE id IN ({placeholders})",
        tuple(item_ids),
    ).fetchall()


async def batch_get_items(item_ids: list[int]) -> dict[int, dict]:
    """Fetch many items by id in one query (no ownership check).

    Returns a ``{item_id: row_dict}`` map for every id that exists.  Missing
    ids are silently dropped from the result (caller can `.get(iid)` to
    detect).  No ordering guarantee — caller must re-apply the desired
    rank order.

    Used by the hybrid-search semantic leg to fetch top-N items in one
    round-trip instead of N separate ``get_item()`` calls.  Same
    ownership-check contract as :func:`get_item`.
    """
    rows = await asyncio.to_thread(_batch_get_sync, item_ids)
    return {r[0]: _row_to_dict(r) for r in rows}


def _list_sync(
    owner_profile_id: int,
    *,
    category_id: int | None,
    include_subtree: bool,
    kind: str | None,
    deleted_only: bool,
    sort: str,
    limit: int,
    offset: int,
) -> list[tuple]:
    c = _conn()

    # Build WHERE clause parts.
    conds: list[str] = ["i.owner_profile_id = ?"]
    params: list[Any] = [owner_profile_id]

    if deleted_only:
        conds.append("i.deleted_at IS NOT NULL")
    else:
        conds.append("i.deleted_at IS NULL")

    if kind:
        conds.append("i.kind = ?")
        params.append(kind)

    if category_id is not None:
        if include_subtree:
            # Recursive CTE: collect all descendant category ids.
            conds.append(
                "i.category_id IN ("
                "  WITH RECURSIVE sub(id) AS ("
                "    SELECT id FROM categories WHERE id = ?"
                "    UNION ALL"
                "    SELECT c.id FROM categories c JOIN sub ON c.parent_id = sub.id"
                "  ) SELECT id FROM sub"
                ")"
            )
            params.append(category_id)
        else:
            conds.append("i.category_id = ?")
            params.append(category_id)

    order = {
        "date_desc":  "i.created_at DESC",
        "date_asc":   "i.created_at ASC",
        "title_asc":  "COALESCE(i.title, '') ASC",
        "sort_order": "i.sort_order ASC, i.created_at DESC",
    }.get(sort, "i.created_at DESC")

    where = " AND ".join(conds)
    params.extend([limit, offset])
    return c.execute(
        f"SELECT {_SELECT_COLS} FROM items i"
        f"  WHERE {where}"
        f"  ORDER BY {order}"
        "   LIMIT ? OFFSET ?",
        params,
    ).fetchall()


async def list_items(
    owner_profile_id: int,
    *,
    category_id: int | None = None,
    include_subtree: bool = False,
    kind: str | None = None,
    deleted_only: bool = False,
    sort: str = "date_desc",
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List items visible to owner_profile_id.

    ``category_id=None`` → all items across categories.
    ``include_subtree=True`` → recurse into child categories.
    ``deleted_only=True`` → return only soft-deleted items (Trash view).
    ``sort``: ``'date_desc'`` | ``'date_asc'`` | ``'title_asc'`` | ``'sort_order'``
    """
    rows = await asyncio.to_thread(
        _list_sync,
        owner_profile_id,
        category_id=category_id,
        include_subtree=include_subtree,
        kind=kind,
        deleted_only=deleted_only,
        sort=sort,
        limit=min(limit, 200),
        offset=max(offset, 0),
    )
    return [_row_to_dict(r) for r in rows]


def _fts_search_sync(
    owner_profile_id: int,
    query: str,
    *,
    category_id: int | None,
    include_subtree: bool,
    kind: str | None,
    limit: int,
) -> list[tuple]:
    """BM25 search via FTS5.  Returns items rows sorted by relevance."""
    c = _conn()

    # Category filter: build a subquery for the in-scope category ids.
    cat_filter = ""
    cat_params: list[Any] = []
    if category_id is not None:
        if include_subtree:
            cat_filter = (
                " AND i.category_id IN ("
                "   WITH RECURSIVE sub(id) AS ("
                "     SELECT id FROM categories WHERE id = ?"
                "     UNION ALL"
                "     SELECT c.id FROM categories c JOIN sub ON c.parent_id = sub.id"
                "   ) SELECT id FROM sub"
                " )"
            )
            cat_params.append(category_id)
        else:
            cat_filter = " AND i.category_id = ?"
            cat_params.append(category_id)

    kind_filter = ""
    kind_params: list[Any] = []
    if kind:
        kind_filter = " AND i.kind = ?"
        kind_params.append(kind)

    # Qualify all column names with the 'i' alias to avoid "ambiguous column
    # name" errors — the FTS5 virtual table exposes title/summary/body with
    # the same names as the backing items table.
    qualified_cols = ", ".join(f"i.{col.strip()}" for col in _SELECT_COLS.split(","))
    params: list[Any] = [query, owner_profile_id] + cat_params + kind_params + [limit]
    return c.execute(
        f"SELECT {qualified_cols}"
        "  FROM items_fts fts"
        "  JOIN items i ON fts.rowid = i.id"
        "  WHERE items_fts MATCH ?"
        "    AND i.owner_profile_id = ?"
        "    AND i.deleted_at IS NULL"
        + cat_filter + kind_filter +
        "  ORDER BY rank"
        "  LIMIT ?",
        params,
    ).fetchall()


async def fts_search(
    owner_profile_id: int,
    query: str,
    *,
    category_id: int | None = None,
    include_subtree: bool = False,
    kind: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """BM25 full-text search via FTS5.

    Returns up to ``limit`` items sorted by FTS5 ``rank`` (BM25 score).
    Used by the search layer as one leg of the hybrid search — the
    semantic leg lives in ``app/items/search.py``.
    """
    rows = await asyncio.to_thread(
        _fts_search_sync,
        owner_profile_id,
        query,
        category_id=category_id,
        include_subtree=include_subtree,
        kind=kind,
        limit=min(limit, 100),
    )
    return [_row_to_dict(r) for r in rows]


# ── Update ───────────────────────────────────────────────────────────────

_SENTINEL = object()  # "leave this field unchanged"


def _update_sync(
    item_id: int,
    owner_profile_id: int,
    *,
    title: Any,
    body: Any,
    summary: Any,
    category_id: Any,
    status: Any,
) -> bool:
    """Partial-update: only columns whose value is not _SENTINEL are written."""
    sets: list[str] = []
    params: list[Any] = []
    for col, val in [
        ("title", title),
        ("body", body),
        ("summary", summary),
        ("category_id", category_id),
        ("status", status),
    ]:
        if val is not _SENTINEL:
            sets.append(f"{col} = ?")
            params.append(val)
    if not sets:
        return True  # no-op
    params.extend([item_id, owner_profile_id])
    c = _conn()
    cur = c.execute(
        f"UPDATE items SET {', '.join(sets)}"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        params,
    )
    return cur.rowcount > 0


async def update_item(
    item_id: int,
    owner_profile_id: int,
    *,
    title: str | None = _SENTINEL,
    body: str | None = _SENTINEL,
    summary: str | None = _SENTINEL,
    category_id: int | None = _SENTINEL,
    status: str | None = _SENTINEL,
) -> bool:
    """Partial-update an item.  Omitted kwargs leave the column unchanged.

    Returns False if no row matched (wrong id or wrong owner).
    """
    return await asyncio.to_thread(
        _update_sync, item_id, owner_profile_id,
        title=title, body=body, summary=summary,
        category_id=category_id, status=status,
    )


# ── Soft-delete / restore / purge ────────────────────────────────────────

def _delete_sync(item_id: int, owner_profile_id: int) -> bool:
    now = time.time()
    c = _conn()
    cur = c.execute(
        "UPDATE items SET deleted_at = ?"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (now, item_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def delete_item(item_id: int, owner_profile_id: int) -> bool:
    """Soft-delete: stamp deleted_at.  Row stays in DB for 7 days then purges."""
    return await asyncio.to_thread(_delete_sync, item_id, owner_profile_id)


def _restore_sync(item_id: int, owner_profile_id: int) -> bool:
    c = _conn()
    cur = c.execute(
        "UPDATE items SET deleted_at = NULL"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NOT NULL",
        (item_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def restore_item(item_id: int, owner_profile_id: int) -> bool:
    """Clear deleted_at — undo a soft-delete."""
    return await asyncio.to_thread(_restore_sync, item_id, owner_profile_id)


def _purge_sync(item_id: int, owner_profile_id: int) -> bool:
    """Hard-delete row + media file.  Irreversible."""
    c = _conn()
    row = c.execute(
        "SELECT media_path FROM items WHERE id = ? AND owner_profile_id = ?",
        (item_id, owner_profile_id),
    ).fetchone()
    if not row:
        return False
    media_path = row[0]
    cur = c.execute(
        "DELETE FROM items WHERE id = ? AND owner_profile_id = ?",
        (item_id, owner_profile_id),
    )
    if cur.rowcount > 0 and media_path:
        try:
            (ITEMS_DIR / media_path).unlink(missing_ok=True)
        except Exception:
            log.warning("items: failed to unlink media %r", media_path, exc_info=True)
    return cur.rowcount > 0


async def purge_item(item_id: int, owner_profile_id: int) -> bool:
    """Permanently delete item + any on-disk media.

    Called by the Trash GC scheduler or explicit "delete forever" UI action.
    """
    return await asyncio.to_thread(_purge_sync, item_id, owner_profile_id)


# ── Move / reorder / check ───────────────────────────────────────────────

def _move_sync(item_id: int, owner_profile_id: int, new_category_id: int | None) -> bool:
    c = _conn()
    cur = c.execute(
        "UPDATE items SET category_id = ?"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (new_category_id, item_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def move_item(item_id: int, owner_profile_id: int, new_category_id: int | None) -> bool:
    """Change an item's category.  ``new_category_id=None`` → uncategorised."""
    return await asyncio.to_thread(_move_sync, item_id, owner_profile_id, new_category_id)


def _reorder_sync(item_id: int, owner_profile_id: int, sort_order: float) -> bool:
    c = _conn()
    cur = c.execute(
        "UPDATE items SET sort_order = ?"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (sort_order, item_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def reorder_item(item_id: int, owner_profile_id: int, sort_order: float) -> bool:
    """Set sort_order for drag-and-drop positioning within a checklist or folder."""
    return await asyncio.to_thread(_reorder_sync, item_id, owner_profile_id, sort_order)


def _toggle_checked_sync(item_id: int, owner_profile_id: int) -> dict:
    now = time.time()
    c = _conn()
    row = c.execute(
        "SELECT completed_at FROM items"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (item_id, owner_profile_id),
    ).fetchone()
    if row is None:
        return {"found": False}
    was_completed = row[0] is not None
    new_val = None if was_completed else now
    c.execute(
        "UPDATE items SET completed_at = ? WHERE id = ?",
        (new_val, item_id),
    )
    return {"found": True, "completed": new_val is not None, "completed_at": new_val}


async def toggle_checked(item_id: int, owner_profile_id: int) -> dict:
    """Check/uncheck a checklist item.

    Returns ``{"found": bool, "completed": bool, "completed_at": float | None}``.
    Idempotent direction-flip: checked → unchecked, unchecked → checked.
    """
    return await asyncio.to_thread(_toggle_checked_sync, item_id, owner_profile_id)


# ── Async-filled columns (set by ingest pipeline) ────────────────────────

def _set_embedding_sync(item_id: int, embedding: bytes) -> None:
    _conn().execute("UPDATE items SET embedding = ? WHERE id = ?", (embedding, item_id))


async def set_item_embedding(item_id: int, embedding: bytes) -> None:
    """Store a fastembed float32 BLOB.  Called by the ingest pipeline after encode."""
    await asyncio.to_thread(_set_embedding_sync, item_id, embedding)


def _set_summary_sync(item_id: int, summary: str) -> None:
    _conn().execute("UPDATE items SET summary = ? WHERE id = ?", (summary, item_id))


async def set_item_summary(item_id: int, summary: str) -> None:
    """Cache the LLM/vision-generated summary.  Called async after ingest."""
    await asyncio.to_thread(_set_summary_sync, item_id, summary)


def _set_media_path_sync(item_id: int, media_path: str) -> None:
    _conn().execute("UPDATE items SET media_path = ? WHERE id = ?", (media_path, item_id))


async def set_item_media_path(item_id: int, media_path: str) -> None:
    """Update the relative media_path once the file is written to disk."""
    await asyncio.to_thread(_set_media_path_sync, item_id, media_path)


# ── Trash GC ─────────────────────────────────────────────────────────────

def _purge_expired_sync(max_age_days: int) -> int:
    cutoff = time.time() - max_age_days * 86400
    c = _conn()
    # Collect ids + media_paths first so we can unlink files.
    rows = c.execute(
        "SELECT id, owner_profile_id, media_path FROM items"
        "  WHERE deleted_at IS NOT NULL AND deleted_at < ?",
        (cutoff,),
    ).fetchall()
    if not rows:
        return 0
    ids = [r[0] for r in rows]
    for _id, _owner, media_path in rows:
        if media_path:
            try:
                (ITEMS_DIR / media_path).unlink(missing_ok=True)
            except Exception:
                log.warning("items GC: unlink %r failed", media_path, exc_info=True)
    placeholders = ",".join("?" * len(ids))
    c.execute(f"DELETE FROM items WHERE id IN ({placeholders})", ids)
    log.info("items GC: purged %d expired trash items", len(ids))
    return len(ids)


async def purge_expired_trash(max_age_days: int = 7) -> int:
    """Hard-delete items whose deleted_at is older than max_age_days.

    Intended to run periodically from the scheduler.  Returns the count
    of rows removed.
    """
    return await asyncio.to_thread(_purge_expired_sync, max_age_days)


# ── Semantic search support ───────────────────────────────────────────────

def _get_embeddings_sync(owner_profile_id: int, limit: int) -> list[tuple]:
    c = _conn()
    return c.execute(
        "SELECT id, embedding FROM items"
        "  WHERE owner_profile_id = ? AND deleted_at IS NULL"
        "    AND embedding IS NOT NULL"
        "  LIMIT ?",
        (owner_profile_id, limit),
    ).fetchall()


async def get_item_embeddings(owner_profile_id: int, limit: int = 2000) -> list[tuple]:
    """Return ``[(id, embedding_bytes), ...]`` for cosine-similarity search.

    The semantic search layer in ``app/items/search.py`` decodes the bytes
    and scores them in-process with numpy — fast enough for ≤10 k items.
    """
    return await asyncio.to_thread(_get_embeddings_sync, owner_profile_id, limit)
