import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ._deps import _current_user

log = logging.getLogger(__name__)

router = APIRouter()


# ─── Personal item store — Categories ────────────────────────────────────


class CategoryCreateRequest(BaseModel):
    name: str
    parent_id: int | None = None
    kind: str = "folder"


class CategoryPatchRequest(BaseModel):
    name: str | None = None
    parent_id: int | None = None  # use -1 as sentinel to move to root


class CategoryShareRequest(BaseModel):
    with_profile_id: int
    permission: str = "read"


class ItemCreateRequest(BaseModel):
    kind: str
    title: str | None = None
    url: str | None = None
    body: str | None = None
    category_id: int | None = None
    source_meta: dict | None = None  # stored as JSON string


class ItemPatchRequest(BaseModel):
    title: str | None = None
    body: str | None = None
    category_id: int | None = None
    summary: str | None = None


class ItemMoveRequest(BaseModel):
    category_id: int | None = None


class ItemReorderRequest(BaseModel):
    sort_order: float


class AutoSortApplyRequest(BaseModel):
    suggestions: list[dict]  # [{item_id: int, category_id: int}, ...]


def _strip_embedding(item: dict) -> dict:
    """Drop the embedding blob before serialising to JSON."""
    out = dict(item)
    out.pop("embedding", None)
    return out


@router.get("/api/users/{profile_id}/categories")
async def api_list_categories(
    profile_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.categories import list_categories
    cats = await list_categories(profile_id, include_shared=True)
    return JSONResponse({"categories": cats})


@router.post("/api/users/{profile_id}/categories")
async def api_create_category(
    profile_id: int,
    body: CategoryCreateRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.categories import create_category
    cat_id = await create_category(
        owner_profile_id=profile_id,
        name=body.name,
        parent_id=body.parent_id,
        kind=body.kind,
    )
    return JSONResponse({"ok": True, "category_id": cat_id})


@router.patch("/api/users/{profile_id}/categories/{category_id}")
async def api_patch_category(
    profile_id: int,
    category_id: int,
    body: CategoryPatchRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.categories import rename_category, move_category, list_subtree, get_category
    cat = await get_category(category_id)
    if cat is None or cat["owner_profile_id"] != profile_id:
        raise HTTPException(404, "category not found")
    if body.name is not None:
        ok = await rename_category(category_id, profile_id, body.name)
        if not ok:
            raise HTTPException(404, "category not found")
    if body.parent_id is not None:
        # -1 is the sentinel for "move to root"
        new_parent: int | None = None if body.parent_id == -1 else body.parent_id
        if new_parent is not None:
            subtree = await list_subtree(category_id, profile_id)
            if any(c["id"] == new_parent for c in subtree):
                raise HTTPException(400, "cycle detected: new parent is a descendant")
        ok = await move_category(category_id, profile_id, new_parent)
        if not ok:
            raise HTTPException(404, "category not found")
    return JSONResponse({"ok": True, "category_id": category_id})


@router.delete("/api/users/{profile_id}/categories/{category_id}")
async def api_delete_category(
    profile_id: int,
    category_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.categories import delete_category
    ok = await delete_category(category_id, profile_id)
    if not ok:
        raise HTTPException(404, "category not found")
    return JSONResponse({"ok": True})


@router.post("/api/users/{profile_id}/categories/{category_id}/share")
async def api_share_category(
    profile_id: int,
    category_id: int,
    body: CategoryShareRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.categories import share_category, get_category
    cat = await get_category(category_id)
    if cat is None or cat["owner_profile_id"] != profile_id:
        raise HTTPException(404, "category not found")
    try:
        await share_category(category_id, body.with_profile_id, permission=body.permission)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"ok": True})


@router.delete("/api/users/{profile_id}/categories/{category_id}/share/{with_profile_id}")
async def api_unshare_category(
    profile_id: int,
    category_id: int,
    with_profile_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.categories import unshare_category, get_category
    cat = await get_category(category_id)
    if cat is None or cat["owner_profile_id"] != profile_id:
        raise HTTPException(404, "category not found")
    await unshare_category(category_id, with_profile_id)
    return JSONResponse({"ok": True})


# ─── Personal item store — Items ─────────────────────────────────────────


@router.get("/api/users/{profile_id}/items")
async def api_list_items(
    profile_id: int,
    category_id: int | None = None,
    kind: str | None = None,
    sort: str = "date_desc",
    limit: int = 50,
    offset: int = 0,
    deleted_only: bool = False,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import list_items
    rows = await list_items(
        profile_id,
        category_id=category_id,
        kind=kind,
        sort=sort,
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
        deleted_only=deleted_only,
    )
    return JSONResponse({"items": [_strip_embedding(r) for r in rows]})


@router.post("/api/users/{profile_id}/items")
async def api_create_item(
    profile_id: int,
    body: ItemCreateRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    import json as _json
    from ..items import ingest as _ingest
    source_meta_str = _json.dumps(body.source_meta) if body.source_meta else None
    try:
        if body.kind == "text":
            if not body.body:
                raise HTTPException(400, "body required for text items")
            item_id = await _ingest.ingest_text(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=body.category_id,
                body=body.body,
                title=body.title,
            )
        elif body.kind == "link":
            if not body.url:
                raise HTTPException(400, "url required for link items")
            item_id = await _ingest.ingest_link(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=body.category_id,
                url=body.url,
                title=body.title,
            )
        elif body.kind in ("video", "short"):
            if not body.url:
                raise HTTPException(400, "url required for video/short items")
            item_id = await _ingest.ingest_video(
                owner_profile_id=profile_id,
                created_by_profile_id=profile_id,
                category_id=body.category_id,
                url=body.url,
                kind=body.kind,
                title=body.title,
            )
        else:
            raise HTTPException(400, f"unsupported kind: {body.kind!r} — use text/link/video/short; screenshots go to /items/screenshot")
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api_create_item failed")
        raise HTTPException(500, str(exc))
    return JSONResponse({"ok": True, "item_id": item_id}, status_code=201)


@router.post("/api/users/{profile_id}/items/screenshot")
async def api_upload_screenshot(
    profile_id: int,
    file: UploadFile,
    category_id: int | None = None,
    title: str | None = None,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..items import ingest as _ingest
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        item_id = await _ingest.ingest_screenshot(
            owner_profile_id=profile_id,
            created_by_profile_id=profile_id,
            category_id=category_id,
            image_bytes=data,
            title=title,
        )
    except Exception as exc:
        log.exception("api_upload_screenshot failed")
        raise HTTPException(500, str(exc))
    return JSONResponse({"ok": True, "item_id": item_id}, status_code=201)


@router.get("/api/users/{profile_id}/items/search")
async def api_search_items(
    profile_id: int,
    q: str,
    category_id: int | None = None,
    kind: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
    sort: str = "relevance",
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    import datetime as _dt
    from ..items import search as _search

    def _parse_ts(s: str | None) -> float | None:
        if not s:
            return None
        try:
            return _dt.datetime.fromisoformat(s).timestamp()
        except ValueError:
            return None

    results = await _search.hybrid_search(
        profile_id, q,
        category_id=category_id,
        kind=kind,
        date_from=_parse_ts(date_from),
        date_to=_parse_ts(date_to),
        limit=max(1, min(limit, 50)),
        sort=sort,
    )
    return JSONResponse({"results": [_strip_embedding(r) for r in results]})


@router.get("/api/users/{profile_id}/items/{item_id}")
async def api_get_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import get_item
    row = await get_item(item_id)
    if row is None or row["owner_profile_id"] != profile_id:
        raise HTTPException(404, "item not found")
    return JSONResponse({"item": _strip_embedding(row)})


@router.patch("/api/users/{profile_id}/items/{item_id}")
async def api_patch_item(
    profile_id: int,
    item_id: int,
    body: ItemPatchRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import update_item, _SENTINEL
    ok = await update_item(
        item_id, profile_id,
        title=body.title if body.title is not None else _SENTINEL,
        body=body.body if body.body is not None else _SENTINEL,
        category_id=body.category_id if body.category_id is not None else _SENTINEL,
        summary=body.summary if body.summary is not None else _SENTINEL,
    )
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@router.delete("/api/users/{profile_id}/items/{item_id}")
async def api_delete_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import delete_item
    ok = await delete_item(item_id, profile_id)
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@router.post("/api/users/{profile_id}/items/{item_id}/restore")
async def api_restore_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import restore_item
    ok = await restore_item(item_id, profile_id)
    if not ok:
        raise HTTPException(404, "item not found or not in trash")
    return JSONResponse({"ok": True})


@router.post("/api/users/{profile_id}/items/{item_id}/move")
async def api_move_item(
    profile_id: int,
    item_id: int,
    body: ItemMoveRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import move_item
    ok = await move_item(item_id, profile_id, body.category_id)
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@router.post("/api/users/{profile_id}/items/{item_id}/reorder")
async def api_reorder_item(
    profile_id: int,
    item_id: int,
    body: ItemReorderRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import reorder_item
    ok = await reorder_item(item_id, profile_id, body.sort_order)
    if not ok:
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True})


@router.post("/api/users/{profile_id}/items/{item_id}/check")
async def api_check_item(
    profile_id: int,
    item_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import toggle_checked
    result = await toggle_checked(item_id, profile_id)
    if not result.get("found"):
        raise HTTPException(404, "item not found")
    return JSONResponse({"ok": True, "completed": result["completed"], "completed_at": result["completed_at"]})


@router.post("/api/users/{profile_id}/items/auto_sort/suggest")
async def api_auto_sort_suggest(
    profile_id: int,
    category_id: int | None = None,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Ask the LLM to suggest category reassignments for items.

    Runs ``suggest_auto_sort`` from the auto_sort module and returns a list
    of ``{item_id, category_id, category_name, reason}`` suggestions.
    The caller must present these to the user before calling the apply
    endpoint — nothing is moved here.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..storage.items import list_items
    from ..storage.categories import list_categories, _ALL_DEPTHS
    from ..items.auto_sort import suggest_auto_sort

    items = await list_items(
        profile_id,
        category_id=category_id,
        limit=max(1, min(limit, 50)),
    )
    cats = await list_categories(profile_id)
    suggestions = await suggest_auto_sort(items, cats)
    return JSONResponse({"suggestions": suggestions})


@router.post("/api/users/{profile_id}/items/auto_sort/apply")
async def api_auto_sort_apply(
    profile_id: int,
    body: AutoSortApplyRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Apply an approved list of auto-sort suggestions.

    Calls ``apply_auto_sort`` which invokes ``move_item`` for each entry.
    Returns the count of successfully moved items.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    from ..items.auto_sort import apply_auto_sort

    count = await apply_auto_sort(body.suggestions, profile_id)
    return JSONResponse({"ok": True, "moved": count})
