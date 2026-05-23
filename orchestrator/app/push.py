"""
Web Push (VAPID) — closed-browser delivery of voicemail notifications.

Today the WS path already delivers a ``voicemail_arrived`` event to every
open tab — but that only works while the tab is open.  Web Push closes
the gap: when the user is logged in on this device the orchestrator can
wake the Service Worker (``frontend/sw.js``) on the browser side and
surface a system notification even if every tab is closed.

Crypto plumbing in one paragraph:

  • VAPID identifies *us* to the push service (Mozilla / Google / Apple).
    We generate a single EC P-256 keypair on first startup and persist
    the private key under ``$DATA_DIR_CONTAINER/vapid_private.pem`` so
    it survives restarts.  The public key (base64url-encoded
    uncompressed point) is what the frontend hands to
    ``pushManager.subscribe({applicationServerKey})``.

  • Each subscription brings its own per-recipient ``p256dh`` public
    key + ``auth`` secret.  ``pywebpush`` uses those plus our VAPID key
    to derive the aes128gcm content key, encrypt the payload, and POST
    it to the push service.

We expose a small async helper :func:`send_to_profile` so the WS-side
voicemail hook can call it without learning anything about VAPID.  We
intentionally do NOT pre-import any pywebpush code at module-load time —
keeps the orchestrator startable in test environments where ``pywebpush``
isn't installed (e.g. ``pip install -e ".[test]"`` skips it).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from .storage import (
    delete_push_subscription,
    list_push_subscriptions,
    touch_push_subscription,
    upsert_push_subscription,
)

log = logging.getLogger(__name__)


# ── Paths + constants ──────────────────────────────────────────────────


# Same data-mount as voice_messages / custom_voices — single volume in
# docker-compose covers all user-data.  PRIVATE key only; the public key
# is derived from it on every call (cheap, no I/O needed).
_DATA_DIR = Path(os.environ.get("DATA_DIR_CONTAINER", "/data"))
VAPID_PRIVATE_PEM_PATH = _DATA_DIR / "vapid_private.pem"

# Subject claim for the VAPID JWT.  Per spec it must be a contact URL —
# either ``mailto:`` or ``https://``.  No real mail is delivered to it;
# the push services use it to contact us if something goes wrong with
# our subscription pattern.
VAPID_SUBJECT = os.environ.get("VAPID_SUBJECT", "mailto:noreply@voiceassistant.local")

# Default TTL we hand to the push service — keeps an undelivered message
# alive on their side for this many seconds.  5 minutes is plenty for a
# voicemail ping (the goal is "user picks up the phone right after it
# arrives", not "queue for tomorrow's commute").
DEFAULT_TTL_S = 300


# Cached private-key object so we don't re-parse the PEM on every send.
# Set lazily in :func:`init_vapid` (and re-set if the file is rotated
# while the process is alive — not a code path we hit today, but cheap).
_PRIVATE_KEY: ec.EllipticCurvePrivateKey | None = None
_PUBLIC_KEY_B64: str | None = None


# ── Key management ─────────────────────────────────────────────────────


def _b64url_no_pad(data: bytes) -> str:
    """URL-safe base64 without padding — VAPID/JOSE convention."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _derive_public_b64(priv: ec.EllipticCurvePrivateKey) -> str:
    """Return the uncompressed-point public key, base64url-encoded.

    Web Push wants the raw 65-byte X9.62 uncompressed point (one 0x04
    leading byte + 32-byte X + 32-byte Y).  ``cryptography`` exposes that
    via ``public_bytes(UncompressedPoint)`` — no manual SEC1 wrangling.
    """
    pub = priv.public_key()
    raw = pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return _b64url_no_pad(raw)


def init_vapid() -> None:
    """Load the VAPID private key, generating a fresh one if missing.

    Called once from FastAPI's lifespan startup.  Cheap on subsequent
    calls — bails out if the cached key is already populated.
    """
    global _PRIVATE_KEY, _PUBLIC_KEY_B64
    if _PRIVATE_KEY is not None:
        return
    VAPID_PRIVATE_PEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if VAPID_PRIVATE_PEM_PATH.exists():
        pem_bytes = VAPID_PRIVATE_PEM_PATH.read_bytes()
        priv = serialization.load_pem_private_key(pem_bytes, password=None)
        if not isinstance(priv, ec.EllipticCurvePrivateKey):
            raise RuntimeError(
                f"VAPID key at {VAPID_PRIVATE_PEM_PATH} is not an EC private key"
            )
        log.info("push: loaded VAPID key from %s", VAPID_PRIVATE_PEM_PATH)
    else:
        priv = ec.generate_private_key(ec.SECP256R1())
        pem_bytes = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        VAPID_PRIVATE_PEM_PATH.write_bytes(pem_bytes)
        # Lock down — the key is the orchestrator's identity to every
        # push service it talks to.  0600 on the host's mount.
        try:
            os.chmod(VAPID_PRIVATE_PEM_PATH, 0o600)
        except OSError:
            log.debug("push: chmod 600 on VAPID key failed (non-fatal)")
        log.info("push: generated new VAPID key at %s", VAPID_PRIVATE_PEM_PATH)
    _PRIVATE_KEY = priv  # type: ignore[assignment]
    _PUBLIC_KEY_B64 = _derive_public_b64(_PRIVATE_KEY)
    log.info("push: VAPID public key %s", _PUBLIC_KEY_B64[:16] + "…")


def get_public_key() -> str:
    """Return the VAPID public key (base64url, no padding) for the frontend.

    Raises ``RuntimeError`` if :func:`init_vapid` has not run yet — the
    HTTP endpoint catches that and surfaces a 503 instead of leaking the
    bare exception.
    """
    if _PUBLIC_KEY_B64 is None:
        raise RuntimeError("VAPID not initialised — call init_vapid() first")
    return _PUBLIC_KEY_B64


def _get_private_pem() -> str:
    """Return the cached private key in PEM form — what pywebpush wants.

    pywebpush re-parses the PEM internally per call; passing a PEM string
    is the path of least friction and avoids re-using the same EC object
    across threads (cryptography's EC objects are immutable, but the
    library's internal state isn't documented as thread-safe).
    """
    if _PRIVATE_KEY is None:
        raise RuntimeError("VAPID not initialised — call init_vapid() first")
    return _PRIVATE_KEY.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


# ── Subscription helpers (thin wrappers over storage) ──────────────────


async def register_subscription(
    profile_id: int,
    subscription: dict[str, Any],
    user_agent: str | None,
) -> int:
    """Persist a PushSubscription JSON dict from the browser.

    Expected shape (mirrors ``PushSubscription#toJSON()`` on the browser):
        {
            "endpoint": "https://fcm.googleapis.com/fcm/send/...",
            "keys": {"p256dh": "...", "auth": "..."}
        }
    """
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys") or {}
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")
    if not (endpoint and p256dh and auth):
        raise ValueError("subscription missing endpoint or keys.p256dh/auth")
    return await upsert_push_subscription(
        profile_id=profile_id,
        endpoint=endpoint,
        p256dh_key=p256dh,
        auth_key=auth,
        user_agent=user_agent,
    )


async def unregister_subscription(endpoint: str) -> bool:
    """Drop one subscription by endpoint.  Thin wrapper for symmetry."""
    return await delete_push_subscription(endpoint)


# ── Send ───────────────────────────────────────────────────────────────


def _send_one_sync(
    *,
    subscription_info: dict[str, Any],
    payload_bytes: bytes,
    ttl_s: int,
    vapid_private_pem: str,
    vapid_claims: dict[str, Any],
) -> int:
    """Synchronous send helper — runs in a thread via :func:`send_to_profile`.

    Returns the HTTP status code from the push service.  Re-raises
    ``WebPushException`` for non-2xx so the caller can inspect the
    ``response.status_code`` and decide whether to GC the subscription
    (410 Gone / 404 → delete).

    Imported lazily so the module loads cleanly without pywebpush in
    test environments — tests mock :func:`send_to_profile` directly.
    """
    from pywebpush import webpush  # type: ignore[import-untyped]

    resp = webpush(
        subscription_info=subscription_info,
        data=payload_bytes,
        vapid_private_key=vapid_private_pem,
        vapid_claims=dict(vapid_claims),  # copy: pywebpush mutates this
        ttl=ttl_s,
        content_encoding="aes128gcm",
    )
    return resp.status_code


def _vapid_claims_for(endpoint: str) -> dict[str, Any]:
    """Build the JWT claims pywebpush will sign.

    ``aud`` MUST match the push service's origin (scheme + host) for
    that specific endpoint — pywebpush derives it from the endpoint
    URL when ``aud`` is absent, but we set ``sub`` explicitly so it
    appears in the JWT.  ``exp`` is bounded at 24h by spec; we use the
    pywebpush default by NOT setting it (the library fills in
    ``now + 12h`` if absent).
    """
    return {"sub": VAPID_SUBJECT}


async def send_to_profile(
    profile_id: int,
    payload: dict[str, Any],
    *,
    ttl_s: int = DEFAULT_TTL_S,
) -> int:
    """Push ``payload`` to every subscription registered for ``profile_id``.

    Returns the number of successful sends.  Auto-deletes subscriptions
    that come back with 404 / 410 — the browser has revoked them on its
    side and they'll never deliver again.  Other errors are logged but
    don't bubble out: a single dead subscription mustn't break delivery
    to the others.
    """
    subs = await list_push_subscriptions(profile_id)
    if not subs:
        return 0
    if _PRIVATE_KEY is None:
        log.warning(
            "push: send_to_profile(%d) called before init_vapid()", profile_id
        )
        return 0
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    private_pem = _get_private_pem()
    sent = 0
    for sub in subs:
        endpoint = sub["endpoint"]
        sub_info = {
            "endpoint": endpoint,
            "keys": {"p256dh": sub["p256dh_key"], "auth": sub["auth_key"]},
        }
        claims = _vapid_claims_for(endpoint)
        try:
            status = await asyncio.to_thread(
                _send_one_sync,
                subscription_info=sub_info,
                payload_bytes=payload_bytes,
                ttl_s=ttl_s,
                vapid_private_pem=private_pem,
                vapid_claims=claims,
            )
            if 200 <= status < 300:
                sent += 1
                await touch_push_subscription(endpoint)
                log.debug(
                    "push: sent to profile=%d endpoint=%s status=%d",
                    profile_id, endpoint[:48] + "…", status,
                )
            else:
                log.info(
                    "push: non-2xx from %s for profile=%d (status=%d)",
                    endpoint[:48] + "…", profile_id, status,
                )
        except Exception as exc:
            # pywebpush raises WebPushException for non-2xx; its .response
            # has the status code.  Anything else (network blip, DNS) we
            # treat as transient and don't GC.
            status_code = _extract_status(exc)
            if status_code in (404, 410):
                # Subscription is dead — the browser has revoked it.
                # Drop it so we don't keep trying on every voicemail.
                await delete_push_subscription(endpoint)
                log.info(
                    "push: GC'd dead subscription (profile=%d, status=%d)",
                    profile_id, status_code,
                )
            else:
                log.warning(
                    "push: send failed for profile=%d endpoint=%s: %s",
                    profile_id, endpoint[:48] + "…", exc,
                )
    return sent


def _extract_status(exc: Exception) -> int | None:
    """Pull the HTTP status code out of a pywebpush.WebPushException, if any.

    We avoid importing ``WebPushException`` at module-load time (keeps
    test envs without pywebpush importable); duck-typing on the
    ``response`` attribute is sufficient and resilient to future class-
    name changes.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    return None


# ── Public-key probe (for tests / dev) ─────────────────────────────────


def _reset_for_tests() -> None:
    """Drop the cached keypair — test-teardown helper.

    Some tests want a fresh keypair per test (or no keypair at all).
    Removing the file is the caller's job; this just clears the in-
    process cache so the next :func:`init_vapid` call re-runs the load /
    generate dance.
    """
    global _PRIVATE_KEY, _PUBLIC_KEY_B64
    _PRIVATE_KEY = None
    _PUBLIC_KEY_B64 = None
