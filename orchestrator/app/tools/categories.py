"""
categories — manage item-store folders and checklists (5 actions).

Companion to the ``items`` tool.  Kept separate because category
management is less frequent than item management and the schema is
different (folders vs items).

Actions and their risk levels:
  list     read        — list folders/checklists accessible to the speaker
  create   low_write   — create a new folder or checklist
  rename   low_write   — rename a category
  move     high_write  — move a subtree to a new parent (cycle-safe)
  delete   high_write  — soft-delete a category (items kept, unreachable)

Agent C implements the handler bodies.
"""
from __future__ import annotations

import logging

from ..i18n import t
from ..storage.categories import (
    _ALL_DEPTHS,
    create_category,
    delete_category,
    list_categories,
    list_subtree,
    move_category,
    rename_category,
    resolve_category_by_name,
)
from .base import RiskLevel, ToolCtx, ToolResult, unwrap_ctx, tool

log = logging.getLogger(__name__)


def _categories_risk(args: dict) -> RiskLevel:
    return {
        "list":   "read",
        "create": "low_write",
        "rename": "low_write",
        "move":   "high_write",
        "delete": "high_write",
    }.get(args.get("action", ""), "low_write")


@tool(
    name="categories",
    description=(
        "Manage item-store folders and checklists.\n\n"
        "ACTIONS:\n"
        "  list     Show folders/checklists the speaker can access "
        "(own + shared).  Pass `parent` to list children of a specific folder.\n"
        "  create   Create a new folder or checklist.  Provide `name` and "
        "optionally `parent` (parent folder name) and `kind` (folder|checklist).\n"
        "  rename   Rename a category.  Provide `name` (current) and `new_name`.\n"
        "  move     Move a folder and its entire subtree to a new parent. "
        "Requires passphrase (high_write).  Provide `name` and `parent` "
        "(new parent name; omit to move to root).\n"
        "  delete   Soft-delete a category.  Items inside are preserved but "
        "hidden until the category is restored.  Requires passphrase."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "rename", "move", "delete"],
            },
            "name": {
                "type": "string",
                "description": "Category name (current name for rename/move/delete).",
            },
            "new_name": {
                "type": "string",
                "description": "New name (rename action).",
            },
            "parent": {
                "type": "string",
                "description": "Parent folder name (create: where to place it; move: new parent).",
            },
            "kind": {
                "type": "string",
                "enum": ["folder", "checklist"],
                "description": "Node kind for create (default: folder).",
            },
        },
        "required": ["action"],
    },
    risk=_categories_risk,
)
async def categories(
    action: str,
    *,
    name: str | None = None,
    new_name: str | None = None,
    parent: str | None = None,
    kind: str = "folder",
    ctx=None,
) -> ToolResult:
    cx: ToolCtx = unwrap_ctx(ctx)
    lang = cx.user_lang

    if cx.profile_id is None:
        return ToolResult(text=t("auth.profile_required", lang), data={"error": "no_profile"})

    profile_id: int = cx.profile_id
    await cx.progress("categories", action)

    if action == "list":
        return await _handle_list(profile_id, lang, parent=parent)

    if action == "create":
        return await _handle_create(profile_id, lang, name=name, parent=parent, kind=kind)

    if action == "rename":
        return await _handle_rename(profile_id, lang, name=name, new_name=new_name)

    if action == "move":
        return await _handle_move(profile_id, lang, name=name, parent=parent)

    if action == "delete":
        return await _handle_delete(profile_id, lang, name=name)

    return ToolResult(
        text=t("items.unknown_action", lang, action=action),
        data={"error": "unknown_action"},
    )


# ── Action handlers ───────────────────────────────────────────────────────

async def _handle_list(profile_id: int, lang: str, *, parent: str | None) -> ToolResult:
    # Default: list all depths.  Overridden to a specific id when
    # filtering by parent name.  _ALL_DEPTHS is the sentinel the storage
    # layer uses to omit the parent_id filter entirely.
    parent_sentinel: object = _ALL_DEPTHS

    if parent:
        cat = await resolve_category_by_name(parent, profile_id)
        if cat is None:
            return ToolResult(
                text=t("items.category_not_found", lang, name=parent),
                data={"error": "category_not_found"},
            )
        parent_sentinel = cat["id"]

    cats = await list_categories(profile_id, parent_id=parent_sentinel)
    if not cats:
        return ToolResult(text=t("categories.list_empty", lang), data={"categories": []})

    names = [c["name"] for c in cats]
    summary = t("categories.list_found", lang, count=len(cats), names=", ".join(names[:5]))
    return ToolResult(text=summary, data={"categories": cats})


async def _handle_create(
    profile_id: int, lang: str, *, name: str | None, parent: str | None, kind: str
) -> ToolResult:
    if not name:
        return ToolResult(text=t("categories.name_required", lang), data={"error": "name_required"})

    parent_id = None
    if parent:
        cat = await resolve_category_by_name(parent, profile_id)
        if cat is None:
            return ToolResult(
                text=t("items.category_not_found", lang, name=parent),
                data={"error": "parent_not_found"},
            )
        parent_id = cat["id"]

    cat_id = await create_category(
        owner_profile_id=profile_id,
        name=name,
        parent_id=parent_id,
        kind=kind,
    )
    return ToolResult(
        text=t("categories.created", lang, name=name, kind=kind),
        data={"category_id": cat_id, "name": name, "kind": kind},
    )


async def _handle_rename(
    profile_id: int, lang: str, *, name: str | None, new_name: str | None
) -> ToolResult:
    if not name or not new_name:
        return ToolResult(
            text=t("categories.rename_args_required", lang),
            data={"error": "missing_args"},
        )
    cat = await resolve_category_by_name(name, profile_id)
    if cat is None:
        return ToolResult(
            text=t("items.category_not_found", lang, name=name),
            data={"error": "not_found"},
        )
    ok = await rename_category(cat["id"], profile_id, new_name)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(
        text=t("categories.renamed", lang, old=name, new=new_name),
        data={"category_id": cat["id"], "new_name": new_name},
    )


async def _handle_move(
    profile_id: int, lang: str, *, name: str | None, parent: str | None
) -> ToolResult:
    if not name:
        return ToolResult(text=t("categories.name_required", lang), data={"error": "name_required"})

    cat = await resolve_category_by_name(name, profile_id)
    if cat is None:
        return ToolResult(
            text=t("items.category_not_found", lang, name=name),
            data={"error": "not_found"},
        )

    new_parent_id: int | None = None
    if parent:
        parent_cat = await resolve_category_by_name(parent, profile_id)
        if parent_cat is None:
            return ToolResult(
                text=t("items.category_not_found", lang, name=parent),
                data={"error": "parent_not_found"},
            )
        new_parent_id = parent_cat["id"]

        # Cycle detection: make sure new_parent_id is not in the subtree
        # of the category being moved.
        subtree = await list_subtree(cat["id"], profile_id)
        subtree_ids = {c["id"] for c in subtree}
        if new_parent_id in subtree_ids:
            return ToolResult(
                text=t("categories.move_cycle", lang),
                data={"error": "cycle_detected"},
            )

    ok = await move_category(cat["id"], profile_id, new_parent_id)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    dest = parent or t("categories.root", lang)
    return ToolResult(
        text=t("categories.moved", lang, name=name, dest=dest),
        data={"category_id": cat["id"], "new_parent_id": new_parent_id},
    )


async def _handle_delete(profile_id: int, lang: str, *, name: str | None) -> ToolResult:
    if not name:
        return ToolResult(text=t("categories.name_required", lang), data={"error": "name_required"})

    cat = await resolve_category_by_name(name, profile_id)
    if cat is None:
        return ToolResult(
            text=t("items.category_not_found", lang, name=name),
            data={"error": "not_found"},
        )

    ok = await delete_category(cat["id"], profile_id)
    if not ok:
        return ToolResult(text=t("items.not_found", lang), data={"error": "not_found"})
    return ToolResult(
        text=t("categories.deleted", lang, name=name),
        data={"category_id": cat["id"]},
    )
