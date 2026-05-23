import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import push
from ._deps import _current_user

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/push/vapid_public_key")
async def api_push_vapid_public_key() -> JSONResponse:
    """Hand the frontend our VAPID public key.

    No auth: it's a public key by design — the browser passes it as
    ``applicationServerKey`` to ``pushManager.subscribe``.  Knowing the
    key gives nobody any privileges; the matching private key lives
    exclusively in ``$DATA_DIR_CONTAINER/vapid_private.pem``.
    """
    try:
        key = push.get_public_key()
    except RuntimeError as exc:
        # init_vapid failed at startup — surface 503 so the frontend
        # knows push isn't available (rather than caching a fake key).
        raise HTTPException(503, str(exc))
    return JSONResponse({"public_key": key})


class PushSubscribeRequest(BaseModel):
    """Browser-side ``PushSubscription#toJSON()`` payload.

    We keep the field names verbatim so the frontend can do
    ``fetch('/api/push/subscribe', {body: JSON.stringify(sub)})`` without
    repacking — the lookup keys (``endpoint``, ``keys.p256dh``,
    ``keys.auth``) match the W3C Push API exactly.
    """
    endpoint: str
    keys: dict[str, str]


@router.post("/api/push/subscribe")
async def api_push_subscribe(
    body: PushSubscribeRequest,
    request: Request,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Register a PushSubscription for the logged-in profile.

    Idempotent — repeated calls with the same ``endpoint`` refresh the
    keys (browsers may rotate them) and bump ``created_at`` without
    creating duplicate rows.  The store keys subscription rows on the
    push-service endpoint, so a second profile subscribing from the
    same device transfers ownership rather than collecting both — see
    ``push_subscriptions._upsert_sync`` for the contract.
    """
    try:
        row_id = await push.register_subscription(
            user["profile_id"],
            {"endpoint": body.endpoint, "keys": body.keys},
            user_agent=request.headers.get("user-agent"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info(
        "push: subscribed profile=%d endpoint=%s",
        user["profile_id"], body.endpoint[:48] + "…",
    )
    return JSONResponse({"ok": True, "id": row_id})


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


@router.delete("/api/push/subscribe")
async def api_push_unsubscribe(
    body: PushUnsubscribeRequest,
    user: dict = Depends(_current_user),
) -> JSONResponse:
    """Delete a PushSubscription by endpoint.

    Auth-required because endpoints are sensitive — the push-service
    URL is effectively a capability token for sending to that
    subscription.  We don't cross-check ownership against the cookie:
    a user unsubscribing from someone else's browser is a no-op for
    privacy (their session can't read it anyway) and a positive for
    GC (one fewer dead row).
    """
    deleted = await push.unregister_subscription(body.endpoint)
    log.info(
        "push: unsubscribe profile=%d endpoint=%s deleted=%s",
        user["profile_id"], body.endpoint[:48] + "…", deleted,
    )
    return JSONResponse({"ok": True, "deleted": deleted})
