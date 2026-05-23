"""
Category storage — hierarchical folders and checklists for the item store.

Categories are owned by a single profile (owner_profile_id) and can be
shared with other profiles via category_shares (see share_category /
unshare_category).  The tree is unbounded depth; the parent_id column
creates a self-referencing hierarchy.  Subtree queries use a recursive
CTE — SQLite has supported these since 3.8.3.

Soft-delete (deleted_at IS NOT NULL) hides a category from all list
views.  Items inside a deleted category keep their category_id but
become unreachable via list_items(category_id=…) because that query
joins against active categories.  Restoring a category makes the items
visible again without any extra writes.
"""
from __future__ import annotations

import asyncio
import re
import time

from .db import _conn


# ── Row helpers ──────────────────────────────────────────────────────────

_SELECT_COLS = (
    "id, owner_profile_id, parent_id, name, slug, kind, sort_order, "
    "created_at, deleted_at"
)


def _row_to_dict(r: tuple) -> dict:
    return {
        "id": r[0],
        "owner_profile_id": r[1],
        "parent_id": r[2],
        "name": r[3],
        "slug": r[4],
        "kind": r[5],
        "sort_order": r[6],
        "created_at": r[7],
        "deleted_at": r[8],
    }


def _slugify(name: str) -> str:
    """'My Recipes ✨' → 'my-recipes' (max 64 chars)."""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s[:64] or "category"


# ── Create ───────────────────────────────────────────────────────────────

def _create_sync(
    *,
    owner_profile_id: int,
    name: str,
    parent_id: int | None,
    kind: str,
    sort_order: int,
) -> int:
    now = time.time()
    slug = _slugify(name)
    c = _conn()
    cur = c.execute(
        """
        INSERT INTO categories
          (owner_profile_id, parent_id, name, slug, kind, sort_order, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (owner_profile_id, parent_id, name, slug, kind, sort_order, now),
    )
    if not cur.lastrowid:
        raise RuntimeError("INSERT returned no lastrowid")
    return cur.lastrowid


async def create_category(
    *,
    owner_profile_id: int,
    name: str,
    parent_id: int | None = None,
    kind: str = "folder",
    sort_order: int = 0,
) -> int:
    """Create a new category node.  Returns the new row id.

    ``kind``: ``'folder'`` (default) or ``'checklist'`` (items are
    sortable and have a completed_at checkbox).
    """
    return await asyncio.to_thread(
        _create_sync,
        owner_profile_id=owner_profile_id,
        name=name,
        parent_id=parent_id,
        kind=kind,
        sort_order=sort_order,
    )


# ── Read ─────────────────────────────────────────────────────────────────

def _get_sync(category_id: int) -> tuple | None:
    return _conn().execute(
        f"SELECT {_SELECT_COLS} FROM categories WHERE id = ?", (category_id,)
    ).fetchone()


async def get_category(category_id: int) -> dict | None:
    """Fetch one category by id (no ownership check — caller must verify)."""
    row = await asyncio.to_thread(_get_sync, category_id)
    return _row_to_dict(row) if row else None


_ALL_DEPTHS = object()  # sentinel for list_categories(parent_id=...)


def _list_sync(
    owner_profile_id: int,
    *,
    parent_id: object,        # _ALL_DEPTHS | None | int
    include_shared: bool,
) -> list[tuple]:
    c = _conn()
    conds = ["deleted_at IS NULL"]
    params: list = []

    if include_shared:
        # Own + shared: UNION-based approach (SQLite doesn't support
        # OR on two different join paths cleanly with one index).
        conds.append(
            "(owner_profile_id = ? OR id IN ("
            "  SELECT category_id FROM category_shares WHERE profile_id = ?"
            "))"
        )
        params.extend([owner_profile_id, owner_profile_id])
    else:
        conds.append("owner_profile_id = ?")
        params.append(owner_profile_id)

    if parent_id is not _ALL_DEPTHS:
        if parent_id is None:
            conds.append("parent_id IS NULL")
        else:
            conds.append("parent_id = ?")
            params.append(parent_id)

    where = " AND ".join(conds)
    return c.execute(
        f"SELECT {_SELECT_COLS} FROM categories"
        f"  WHERE {where}"
        "   ORDER BY sort_order ASC, name ASC",
        params,
    ).fetchall()


async def list_categories(
    owner_profile_id: int,
    *,
    parent_id: object = _ALL_DEPTHS,
    include_shared: bool = True,
) -> list[dict]:
    """List categories accessible to owner_profile_id.

    ``parent_id=_ALL_DEPTHS`` (default) → every non-deleted category.
    ``parent_id=None``         → root-level only (no parent).
    ``parent_id=N``            → children of category N.
    ``include_shared=True``    → also categories shared with this profile.

    Import ``_ALL_DEPTHS`` if you need the "all depths" sentinel; normal
    callers can omit ``parent_id`` entirely to get the full tree.
    """
    rows = await asyncio.to_thread(
        _list_sync, owner_profile_id, parent_id=parent_id, include_shared=include_shared
    )
    return [_row_to_dict(r) for r in rows]


def _subtree_sync(root_id: int, owner_profile_id: int) -> list[tuple]:
    """Recursive CTE — root + all descendants in depth-first order."""
    c = _conn()
    return c.execute(
        f"""
        WITH RECURSIVE sub(id) AS (
            SELECT id FROM categories
              WHERE id = ? AND (owner_profile_id = ? OR id IN (
                SELECT category_id FROM category_shares WHERE profile_id = ?
              ))
            UNION ALL
            SELECT c.id FROM categories c JOIN sub ON c.parent_id = sub.id
              WHERE c.deleted_at IS NULL
        )
        SELECT {_SELECT_COLS} FROM categories
          WHERE id IN (SELECT id FROM sub)
          ORDER BY sort_order ASC, name ASC
        """,
        (root_id, owner_profile_id, owner_profile_id),
    ).fetchall()


async def list_subtree(root_id: int, owner_profile_id: int) -> list[dict]:
    """Return the full subtree rooted at root_id (inclusive).

    Used by move-subtree operations to check for cycles and collect all
    affected category ids.
    """
    rows = await asyncio.to_thread(_subtree_sync, root_id, owner_profile_id)
    return [_row_to_dict(r) for r in rows]


def _resolve_by_name_sync(name: str, owner_profile_id: int) -> tuple | None:
    """Case-insensitive name or slug match among non-deleted active categories."""
    c = _conn()
    slug = _slugify(name)
    return c.execute(
        f"SELECT {_SELECT_COLS} FROM categories"
        "  WHERE deleted_at IS NULL"
        "    AND (owner_profile_id = ? OR id IN ("
        "          SELECT category_id FROM category_shares WHERE profile_id = ?)"
        "        )"
        "    AND (LOWER(name) = LOWER(?) OR slug = ?)"
        "  ORDER BY owner_profile_id = ? DESC"  # own categories first
        "  LIMIT 1",
        (owner_profile_id, owner_profile_id, name, slug, owner_profile_id),
    ).fetchone()


async def resolve_category_by_name(name: str, owner_profile_id: int) -> dict | None:
    """Find a category by name (case-insensitive) or its auto-generated slug.

    Returns the first match, preferring owned categories over shared ones.
    Returns None if no match found — callers can then prompt to create.
    """
    row = await asyncio.to_thread(_resolve_by_name_sync, name, owner_profile_id)
    return _row_to_dict(row) if row else None


# ── Update ───────────────────────────────────────────────────────────────

def _rename_sync(category_id: int, owner_profile_id: int, name: str) -> bool:
    slug = _slugify(name)
    c = _conn()
    cur = c.execute(
        "UPDATE categories SET name = ?, slug = ?"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (name, slug, category_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def rename_category(category_id: int, owner_profile_id: int, name: str) -> bool:
    """Rename a category and regenerate its slug."""
    return await asyncio.to_thread(_rename_sync, category_id, owner_profile_id, name)


def _move_sync(
    category_id: int, owner_profile_id: int, new_parent_id: int | None
) -> bool:
    """Move category to a new parent.  Cycle detection is the caller's job."""
    c = _conn()
    cur = c.execute(
        "UPDATE categories SET parent_id = ?"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (new_parent_id, category_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def move_category(
    category_id: int, owner_profile_id: int, new_parent_id: int | None
) -> bool:
    """Change the parent of a category.  ``new_parent_id=None`` → root level.

    The caller MUST verify that new_parent_id is not inside the subtree
    of category_id (would create a cycle).  Use list_subtree() for this.
    """
    return await asyncio.to_thread(_move_sync, category_id, owner_profile_id, new_parent_id)


# ── Soft-delete / restore ─────────────────────────────────────────────────

def _delete_sync(category_id: int, owner_profile_id: int) -> bool:
    now = time.time()
    c = _conn()
    cur = c.execute(
        "UPDATE categories SET deleted_at = ?"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NULL",
        (now, category_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def delete_category(category_id: int, owner_profile_id: int) -> bool:
    """Soft-delete.  Items inside retain their category_id but become unreachable."""
    return await asyncio.to_thread(_delete_sync, category_id, owner_profile_id)


def _restore_sync(category_id: int, owner_profile_id: int) -> bool:
    c = _conn()
    cur = c.execute(
        "UPDATE categories SET deleted_at = NULL"
        "  WHERE id = ? AND owner_profile_id = ? AND deleted_at IS NOT NULL",
        (category_id, owner_profile_id),
    )
    return cur.rowcount > 0


async def restore_category(category_id: int, owner_profile_id: int) -> bool:
    """Clear deleted_at — undo a soft-delete."""
    return await asyncio.to_thread(_restore_sync, category_id, owner_profile_id)


# ── Sharing ───────────────────────────────────────────────────────────────

def _share_sync(category_id: int, with_profile_id: int, permission: str) -> None:
    now = time.time()
    _conn().execute(
        """
        INSERT INTO category_shares (category_id, profile_id, permission, granted_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(category_id, profile_id) DO UPDATE
            SET permission = excluded.permission, granted_at = excluded.granted_at
        """,
        (category_id, with_profile_id, permission, now),
    )


async def share_category(
    category_id: int, with_profile_id: int, *, permission: str = "read"
) -> None:
    """Grant a profile access to a category (upsert).

    ``permission``: ``'read'`` or ``'write'``.  Subsequent calls with a
    different permission update the existing grant — no duplicate rows.
    """
    if permission not in ("read", "write"):
        raise ValueError(f"invalid permission {permission!r}")
    await asyncio.to_thread(_share_sync, category_id, with_profile_id, permission)


def _unshare_sync(category_id: int, with_profile_id: int) -> bool:
    c = _conn()
    cur = c.execute(
        "DELETE FROM category_shares WHERE category_id = ? AND profile_id = ?",
        (category_id, with_profile_id),
    )
    return cur.rowcount > 0


async def unshare_category(category_id: int, with_profile_id: int) -> bool:
    """Revoke a profile's access grant.  Returns True if a row was deleted."""
    return await asyncio.to_thread(_unshare_sync, category_id, with_profile_id)


def _list_shares_sync(category_id: int) -> list[tuple]:
    return _conn().execute(
        "SELECT profile_id, permission, granted_at"
        "  FROM category_shares WHERE category_id = ?",
        (category_id,),
    ).fetchall()


async def list_category_shares(category_id: int) -> list[dict]:
    """Return all ``{profile_id, permission, granted_at}`` for a category."""
    rows = await asyncio.to_thread(_list_shares_sync, category_id)
    return [{"profile_id": r[0], "permission": r[1], "granted_at": r[2]} for r in rows]
