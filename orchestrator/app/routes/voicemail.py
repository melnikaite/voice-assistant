import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..storage import (
    count_unread_voicemail,
    delete_voice_message,
    get_voice_message,
    list_outgoing_voicemail,
    list_voicemail,
    mark_voicemail_listened,
    save_voicemail_reply,
    voicemail_audio_path,
)
from ._deps import _current_user

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/users/{profile_id}/voicemail")
async def api_list_voicemail(
    profile_id: int,
    unread_only: bool = False,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """List voicemail addressed to a profile.

    Cookie-auth required and the requesting profile must match the
    inbox owner — like /memory and /settings, no cross-profile peeks.
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    items = await list_voicemail(
        profile_id,
        unread_only=bool(unread_only),
        limit=max(1, min(int(limit), 200)),
    )
    unread = await count_unread_voicemail(profile_id)
    return JSONResponse({
        "profile_id": profile_id,
        "unread_count": unread,
        "messages": items,
    })


@router.get("/api/users/{profile_id}/outgoing_voicemail")
async def api_list_outgoing_voicemail(
    profile_id: int,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """List voicemail rows the given profile *sent*.

    Same ownership shape as the inbox endpoint: cookie-auth required,
    requesting profile must equal ``profile_id``.  The Sent panel in
    the UI uses this to show the user their own outgoing messages and
    any replies that have come back — closing the delivery loop the
    sender wouldn't otherwise see (the recipient's reply is on the
    voicemail row, not pushed back via TTS today).
    """
    if user["profile_id"] != profile_id:
        raise HTTPException(403, "cross-profile access not allowed")
    items = await list_outgoing_voicemail(
        profile_id,
        limit=max(1, min(int(limit), 200)),
    )
    return JSONResponse({
        "profile_id": profile_id,
        "messages": items,
    })


@router.get("/api/voicemail/{message_id}/audio")
async def api_voicemail_audio(
    message_id: int,
    user: dict = Depends(_current_user),
) -> FileResponse:
    """Serve the raw WAV bytes of a voicemail.

    Cookie-auth required.  The row's ``to_profile_id`` must match the
    logged-in profile — anything else gets a 404 (we don't 403 since
    that would leak existence of someone else's mail).
    """
    row = await get_voice_message(message_id)
    if row is None or row["to_profile_id"] != user["profile_id"]:
        raise HTTPException(404, "not found")
    path = voicemail_audio_path(row["audio_path"] or message_id)
    if not path.exists():
        log.warning("voicemail: wav missing for id=%d (%s)", message_id, path)
        raise HTTPException(404, "audio missing")
    return FileResponse(
        str(path),
        media_type="audio/wav",
        # `inline` lets the <audio> element play it; ``filename=`` makes
        # a manual download save sensibly.
        headers={"Content-Disposition": f'inline; filename="voicemail-{message_id}.wav"'},
    )


@router.post("/api/voicemail/{message_id}/listened")
async def api_voicemail_listened(
    message_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Mark a voicemail as listened (the recipient heard it).

    Idempotent — repeated calls after the first one are no-ops.  Used
    by the frontend Inbox panel when audio playback finishes.
    """
    ok = await mark_voicemail_listened(message_id, user["profile_id"])
    return JSONResponse({"ok": True, "first_time": bool(ok)})


class VoicemailReplyRequest(BaseModel):
    reply: str


@router.post("/api/voicemail/{message_id}/reply")
async def api_voicemail_reply(
    message_id: int,
    body: VoicemailReplyRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Save the recipient's textual reply.

    Delivery to the sender is out-of-scope today — the reply lives on
    the voicemail row so the sender (if they're the host of this
    install too) can see it; cross-device push is a follow-up.
    """
    reply = (body.reply or "").strip()
    if not reply:
        raise HTTPException(400, "empty reply")
    ok = await save_voicemail_reply(message_id, user["profile_id"], reply)
    if not ok:
        raise HTTPException(404, "not found")
    return JSONResponse({"ok": True, "message_id": message_id})


@router.delete("/api/voicemail/{message_id}")
async def api_voicemail_delete(
    message_id: int,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    ok = await delete_voice_message(message_id, user["profile_id"])
    if not ok:
        raise HTTPException(404, "not found")
    return JSONResponse({"ok": True})
