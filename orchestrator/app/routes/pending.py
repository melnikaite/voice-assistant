from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..storage import (
    get_pending_action,
    list_pending_actions,
    list_recent_actions,
    mark_approved,
    mark_rejected,
)
from ._deps import _current_user

router = APIRouter()


@router.get("/api/users/{profile_id}/pending")
async def api_list_pending(
    profile_id: int,
    include_recent: bool = False,
    recent_limit: int = 20,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """List the speaker's pending action queue.

    With ``include_recent=true`` the response also carries the most
    recently finalised actions (executed / failed / rejected / expired)
    so the UI can show a small "Recent" panel next to the live queue.
    Capped by ``recent_limit`` (default 20).
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    items = await list_pending_actions(profile_id=profile_id)
    payload: dict = {"actions": items}
    if include_recent:
        payload["recent"] = await list_recent_actions(
            profile_id=profile_id, limit=max(1, min(int(recent_limit), 100))
        )
    return JSONResponse(payload)


@router.post("/api/pending/{action_id}/approve")
async def api_approve_pending(
    action_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    row = await get_pending_action(action_id)
    if not row:
        raise HTTPException(404, "no such action")
    # An action queued under a different profile shouldn't be approvable
    # by an unrelated user.  Allow if profile matches OR if the action
    # was queued without any profile (e.g. /dev/respond test traffic).
    if row["profile_id"] is not None and row["profile_id"] != user["profile_id"]:
        raise HTTPException(403, "cross-profile approval not allowed")
    ok = await mark_approved(action_id, via="ui")
    if not ok:
        raise HTTPException(409, "action no longer pending")
    return JSONResponse({"ok": True, "action_id": action_id, "summary": row["summary"]})


@router.post("/api/pending/{action_id}/reject")
async def api_reject_pending(
    action_id: int, user: dict = Depends(_current_user)
) -> JSONResponse:
    row = await get_pending_action(action_id)
    if not row:
        raise HTTPException(404, "no such action")
    if row["profile_id"] is not None and row["profile_id"] != user["profile_id"]:
        raise HTTPException(403, "cross-profile rejection not allowed")
    ok = await mark_rejected(action_id, via="ui")
    if not ok:
        raise HTTPException(409, "action no longer pending")
    return JSONResponse({"ok": True, "action_id": action_id, "summary": row["summary"]})
