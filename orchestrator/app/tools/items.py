"""
items — personal item store tool (11 actions).

One tool, one ``action`` parameter, per-action risk callable.  The LLM
picks the right action based on user intent; the tool routes internally.

Actions and their risk levels:
  save        low_write   — add a new item (text/link/video/short/screenshot)
  get         read        — fetch one item by id
  update      low_write   — edit title/body/category of an existing item
  delete      low_write   — soft-delete (move to Trash, 7-day purge)
  restore     low_write   — un-delete from Trash
  search      read        — hybrid semantic+BM25 search with filters
  list        read        — browse a category or all items
  move        low_write   — change an item's category
  reorder     low_write   — set sort_order for D&D checklist ordering
  check       low_write   — toggle a checklist item as done / not done
  auto_sort   high_write  — AI: suggest + (after approval) apply category moves

Agent B and C implement the handler bodies; this file defines the full
interface that both can code against without conflicts.
"""
from __future__ import annotations

import logging

from ..i18n import t
from ..items import auto_sort as _auto_sort
from ..items import ingest as _ingest
from ..items import search as _search
from ..storage.categories import resolve_category_by_name
from ..storage.items import (
    create_item,
    delete_item,
    get_item,
    list_items,
    move_item,
    reorder_item,
    restore_item,
    toggle_checked,
    update_item,
)
from .base import RiskLevel, ToolCtx, ToolResult, unwrap_ctx, tool

log = logging.getLogger(__name__)


# ── Per-action risk callable ──────────────────────────────────────────────

def _items_risk(args: dict) -> RiskLevel:
    """Return the appropriate risk level for the requested action."""
    return {
        "save":      "low_write",
        "get":       "read",
        "update":    "low_write",
        "delete":    "low_write",
        "restore":   "low_write",
        "search":    "read",
        "list":      "read",
        "move":      "low_write",
        "reorder":   "low_write",
        "check":     "low_write",
        "auto_sort": "high_write",
    }.get(args.get("action", ""), "low_write")


# ── Tool definition ───────────────────────────────────────────────────────

@tool(
    name="items",
    description=(
        "Manage the personal item store: save links, text notes, videos, "
        "shorts, and screenshots; search, browse, and organise them into "
        "folders or checklists.\n\n"
        "ACTIONS — pick one per call:\n"
        "  save        Add a new item.  Provide `kind` + content fields "
        "(`url` for links/videos, `body` for text, no body for screenshots "
        "— screenshots are captured separately).  Optionally pass `category` "
        "(name) to place it directly; leave blank to save uncategorised.\n"
        "  get         Fetch one item by `item_id`.\n"
        "  update      Edit `title`, `body`, or `category` of an item.\n"
        "  delete      Soft-delete (moves to Trash; auto-purged after 7 days).\n"
        "  restore     Un-delete from Trash (within 7-day window).\n"
        "  search      Hybrid full-text + semantic search.  Pass `query` "
        "and optionally `category` (name or id), `kind`, `date_from`, "
        "`date_to`, `limit` (default 10).\n"
        "  list        Browse items in `category` (or all if omitted). "
        "Supports `kind`, `sort` (date_desc|date_asc|sort_order), `limit`.\n"
        "  move        Move `item_id` to `category` (name).\n"
        "  reorder     Set `sort_order` (float) for drag-and-drop positioning.\n"
        "  check       Toggle a checklist item done/not-done by `item_id`.\n"
        "  auto_sort   AI suggests category moves for uncategorised or "
        "selected items.  Reads suggestions back aloud; user must confirm "
        "before anything is applied.\n\n"
        "KINDS: text | link | video | short | screenshot\n"
        "SORT:  date_desc (default) | date_asc | title_asc | sort_order"
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "save", "get", "update", "delete", "restore",
                    "search", "list", "move", "reorder", "check", "auto_sort",
                ],
                "description": "Which operation to perform.",
            },
            "item_id": {
                "type": "integer",
                "description": "Target item id (get/update/delete/restore/move/reorder/check).",
            },
            "kind": {
                "type": "string",
                "enum": ["text", "link", "video", "short", "screenshot"],
                "description": "Item kind (required for save; optional filter for search/list).",
            },
            "title": {
                "type": "string",
                "description": "Human-readable title (save/update).",
            },
            "url": {
                "type": "string",
                "description": "URL for link/video/short items (save).",
            },
            "body": {
                "type": "string",
                "description": "Text content for text items (save/update).",
            },
            "category": {
                "type": "string",
                "description": (
                    "Category name to place/filter/move items.  The tool "
                    "resolves the name to an id automatically.  Pass the "
                    "name the user said; do not guess ids."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query (search action).",
            },
            "sort": {
                "type": "string",
                "enum": ["date_desc", "date_asc", "title_asc", "sort_order"],
                "description": "Sort order for list/search.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for search/list (default 10, max 50).",
            },
            "date_from": {
                "type": "string",
                "description": "ISO-8601 date lower bound for search (e.g. '2025-01-01').",
            },
            "date_to": {
                "type": "string",
                "description": "ISO-8601 date upper bound for search.",
            },
            "sort_order": {
                "type": "number",
                "description": "Float sort position (reorder action).",
            },
            "include_subtree": {
                "type": "boolean",
                "description": "If true, recurse into sub-folders when filtering by category.",
            },
        },
        "required": ["action"],
    },
    risk=_items_risk,
)
async def items(
    action: str,
    *,
    item_id: int | None = None,
    kind: str | None = None,
    title: str | None = None,
    url: str | None = None,
    body: str | None = None,
    category: str | None = None,
    query: str | None = None,
    sort: str = "date_desc",
    limit: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_order: float | None = None,
    include_subtree: bool = False,
    ctx=None,
) -> ToolResult:
    cx: ToolCtx = unwrap_ctx(ctx)
    lang = cx.user_lang

    if cx.profile_id is None:
        return ToolResult(
            text=t("auth.profile_required", lang),
            data={"error": "no_profile"},
        )

    profile_id: int = cx.profile_id
    await cx.progress("items", action)

    # ── Resolve category name → id ────────────────────────────────────────
    cat_id: int | None = None
    if category:
        cat_row = await resolve_category_by_name(category, profile_id)
        if cat_row:
            cat_id = cat_row["id"]
        elif action in ("save", "move", "update"):
            # Unknown category on a write action — create it on the fly?
            # For now, tell the user and return.  Agent B can add auto-create.
            return ToolResult(
                text=t("items.category_not_found", lang, name=category),
                data={"error": "category_not_found", "category": category},
            )

    # ── Route to action handler ───────────────────────────────────────────
    if action == "save":
        return await _handle_save(
            profile_id, lang, cx,
            kind=kind, title=title, url=url, body=body, cat_id=cat_id,
        )

    if action == "get":
        return await _handle_get(profile_id, lang, item_id=item_id)

    if action == "update":
        return await _handle_update(
            profile_id, lang, item_id=item_id,
            title=title, body=body, cat_id=cat_id,
        )

    if action == "delete":
        return await _handle_delete(profile_id, lang, item_id=item_id)

    if action == "restore":
        return await _handle_restore(profile_id, lang, item_id=item_id)

    if action == "search":
        return await _handle_search(
            profile_id, lang, cx,
            query=query, cat_id=cat_id,
            include_subtree=include_subtree, kind=kind,
            date_from=date_from, date_to=date_to,
            limit=limit, sort=sort,
        )

    if action == "list":
        return await _handle_list(
            profile_id, lang, cx,
            cat_id=cat_id, include_subtree=include_subtree,
            kind=kind, sort=sort, limit=limit,
        )

    if action == "move":
        return await _handle_move(profile_id, lang, item_id=item_id, cat_id=cat_id)

    if action == "reorder":
        return await _handle_reorder(
            profile_id, lang, item_id=item_id, sort_order=sort_order
        )

    if action == "check":
        return await _handle_check(profile_id, lang, item_id=item_id)

    if action == "auto_sort":
        return await _handle_auto_sort(profile_id, lang, cx, cat_id=cat_id)

    return ToolResult(
        text=t("items.unknown_action", lang, action=action),
        data={"error": "unknown_action"},
    )


# ── Action handlers (stubs — implemented by Agent B & C) ─────────────────


async def _handle_save(
    profile_id: int,
    lang: str,
    cx: ToolCtx,
    *,
    kind: str | None,
    title: str | None,
    url: str | None,
    body: str | None,
    cat_id: int | None,
) -> ToolResult:
    """Save a new item.  Delegates to the appropriate ingest function."""
    if not kind:
        return ToolResult(
            text=t("items.kind_required", lang),
            data={"error": "kind_required"},
        )

    try:
        if kind == "text":
            if not body:
                return ToolResult(
                    text=t("items.body_required", lang),
                    data={"error": "body_required"},
                )
            item_id = await _ingest.ingest_text(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=cat_id,
                body=body,
                title=title,
            )

        elif kind in ("link", "video", "short"):
            if not url:
                return ToolResult(
                    text=t("items.url_required", lang),
                    data={"error": "url_required"},
                )
            if kind == "link":
                item_id = await _ingest.ingest_link(
                    owner_profile_id=profile_id,
                    created_by_profile_id=profile_id,
                    category_id=cat_id,
                    url=url,
                    title=title,
                )
            else:
                item_id = await _ingest.ingest_video(
                    owner_profile_id=profile_id,
                    created_by_profile_id=profile_id,
                    category_id=cat_id,
                    url=url,
                    kind=kind,
                    title=title,
                )

        elif kind == "screenshot":
            # Screenshots arrive via the REST upload endpoint, not by voice.
            return ToolResult(
                text=t("items.screenshot_via_ui", lang),
                data={"hint": "use_upload_endpoint"},
            )

        else:
            return ToolResult(
                text=t("items.unknown_kind", lang, kind=kind),
                data={"error": "unknown_kind"},
            )

    except NotImplementedError:
        return ToolResult(
            text=t("items.not_implemented", lang),
            data={"stub": True, "action": "save"},
        )

    return ToolResult(
        text=t("items.saved", lang, id=item_id),
        data={"item_id": item_id, "kind": kind},
    )


async def _handle_get(
    profile_id: int, lang: str, *, item_id: int | None
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    row = await get_item(item_id)
    if row is None or row["owner_profile_id"] != profile_id:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    title = row.get("title") or row.get("url") or f"item {item_id}"
    summary = row.get("summary") or row.get("body") or ""
    blurb = f'"{title}"' + (f": {summary[:120]}" if summary else "")
    return ToolResult(text=blurb, data={"item": _sanitise(row)})


async def _handle_update(
    profile_id: int,
    lang: str,
    *,
    item_id: int | None,
    title: str | None,
    body: str | None,
    cat_id: int | None,
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    from ..storage.items import _SENTINEL
    ok = await update_item(
        item_id, profile_id,
        title=title if title is not None else _SENTINEL,
        body=body if body is not None else _SENTINEL,
        category_id=cat_id if cat_id is not None else _SENTINEL,
    )
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(text=t("items.updated", lang), data={"item_id": item_id})


async def _handle_delete(
    profile_id: int, lang: str, *, item_id: int | None
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    ok = await delete_item(item_id, profile_id)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(text=t("items.deleted", lang), data={"item_id": item_id})


async def _handle_restore(
    profile_id: int, lang: str, *, item_id: int | None
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    ok = await restore_item(item_id, profile_id)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(text=t("items.restored", lang), data={"item_id": item_id})


async def _handle_search(
    profile_id: int,
    lang: str,
    cx: ToolCtx,
    *,
    query: str | None,
    cat_id: int | None,
    include_subtree: bool,
    kind: str | None,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    sort: str,
) -> ToolResult:
    if not query:
        return ToolResult(text=t("items.query_required", lang), data={"error": "query_required"})

    import datetime as _dt

    def _parse_ts(s: str | None) -> float | None:
        if not s:
            return None
        try:
            return _dt.datetime.fromisoformat(s).timestamp()
        except ValueError:
            return None

    try:
        await cx.progress("search", query)
        results = await _search.hybrid_search(
            profile_id, query,
            category_id=cat_id,
            include_subtree=include_subtree,
            kind=kind,
            date_from=_parse_ts(date_from),
            date_to=_parse_ts(date_to),
            limit=min(limit, 50),
            sort=sort,
        )
    except NotImplementedError:
        return ToolResult(
            text=t("items.not_implemented", lang),
            data={"stub": True, "action": "search"},
        )

    if not results:
        return ToolResult(
            text=t("items.search_empty", lang, query=query),
            data={"count": 0, "query": query},
        )

    count = len(results)
    titles = [r.get("title") or r.get("url") or f"item {r['id']}" for r in results[:3]]
    summary = t("items.search_found", lang, count=count, first=titles[0])
    return ToolResult(
        text=summary,
        data={"count": count, "results": [_sanitise(r) for r in results]},
    )


async def _handle_list(
    profile_id: int,
    lang: str,
    cx: ToolCtx,
    *,
    cat_id: int | None,
    include_subtree: bool,
    kind: str | None,
    sort: str,
    limit: int,
) -> ToolResult:
    try:
        rows = await list_items(
            profile_id,
            category_id=cat_id,
            include_subtree=include_subtree,
            kind=kind,
            sort=sort,
            limit=min(limit, 50),
        )
    except Exception as e:
        log.exception("items.list failed")
        return ToolResult(text=t("tools.crashed", lang), data={"error": str(e)})

    if not rows:
        return ToolResult(text=t("items.list_empty", lang), data={"count": 0})

    count = len(rows)
    titles = [r.get("title") or r.get("url") or f"item {r['id']}" for r in rows[:3]]
    summary = t("items.list_found", lang, count=count, first=titles[0])
    return ToolResult(
        text=summary,
        data={"count": count, "items": [_sanitise(r) for r in rows]},
    )


async def _handle_move(
    profile_id: int, lang: str, *, item_id: int | None, cat_id: int | None
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    ok = await move_item(item_id, profile_id, cat_id)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(text=t("items.moved", lang), data={"item_id": item_id, "category_id": cat_id})


async def _handle_reorder(
    profile_id: int, lang: str, *, item_id: int | None, sort_order: float | None
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    if sort_order is None:
        return ToolResult(text=t("items.sort_order_required", lang), data={"error": "sort_order_required"})
    ok = await reorder_item(item_id, profile_id, sort_order)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(text=t("items.reordered", lang), data={"item_id": item_id, "sort_order": sort_order})


async def _handle_check(
    profile_id: int, lang: str, *, item_id: int | None
) -> ToolResult:
    if not item_id:
        return ToolResult(text=t("items.id_required", lang), data={"error": "id_required"})
    result = await toggle_checked(item_id, profile_id)
    if not result.get("found"):
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    if result["completed"]:
        return ToolResult(text=t("items.checked", lang), data=result)
    return ToolResult(text=t("items.unchecked", lang), data=result)


async def _handle_auto_sort(
    profile_id: int, lang: str, cx: ToolCtx, *, cat_id: int | None
) -> ToolResult:
    """Suggest AI category moves — preview only, no changes applied yet."""
    # Fetch uncategorised items (or items in the specified category).
    rows = await list_items(
        profile_id,
        category_id=cat_id,
        sort="date_desc",
        limit=50,
    )
    if not rows:
        return ToolResult(text=t("items.auto_sort_nothing", lang), data={"count": 0})

    from ..storage.categories import list_categories as _list_cats
    cats = await _list_cats(profile_id)
    if not cats:
        return ToolResult(
            text=t("items.auto_sort_no_categories", lang),
            data={"error": "no_categories"},
        )

    await cx.progress("auto_sort", None)
    try:
        suggestions = await _auto_sort.suggest_auto_sort(rows, cats, ctx=cx)
    except NotImplementedError:
        return ToolResult(
            text=t("items.not_implemented", lang),
            data={"stub": True, "action": "auto_sort"},
        )

    if not suggestions:
        return ToolResult(
            text=t("items.auto_sort_no_suggestions", lang),
            data={"count": 0},
        )

    # Summarise for voice readback — say at most 3 suggestions aloud.
    previews = [
        f"'{s.get('category_name', '?')}' for {s.get('reason', '...')}"
        for s in suggestions[:3]
    ]
    voice_summary = t(
        "items.auto_sort_preview", lang,
        count=len(suggestions),
        previews=", ".join(previews),
    )
    return ToolResult(
        text=voice_summary,
        data={"suggestions": suggestions, "pending_apply": True},
    )


# ── Serialisation helper ──────────────────────────────────────────────────

def _sanitise(row: dict) -> dict:
    """Drop the embedding blob before serialising to JSON."""
    out = dict(row)
    out.pop("embedding", None)
    return out
