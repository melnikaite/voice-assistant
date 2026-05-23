"""Per-process session registry + voicemail fan-out.

Pipeline.py reaches into this module to push ``voicemail_arrived``
events to every active WebSocket session of the recipient profile,
plus a closed-browser ping via Web Push.  Kept separate from
``ws.py`` so the registry is import-cheap (no aiortc / VAD pulls in)
for callers that only need to notify.

The Session type is referenced as a forward string so we don't create
a circular import — ws.py already imports this module.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .ws import Session


# profile_id → list of active Session objects.  Sessions register
# themselves whenever a profile becomes known (cookie auth on the WS
# handshake, voice-side passphrase success, or anywhere else we land
# on a profile_id).  Pipeline.py reaches into this to push
# voicemail-arrived events live to the recipient's open tab.
# Module-level — there's exactly one Session table per process, and
# voicemail-save is process-local too.  A list (not set) because a
# user could have two tabs open; each gets its own ping.
_SESSIONS_BY_PROFILE: dict[int, list["Session"]] = {}


def register_session_profile(profile_id: int, session: "Session") -> None:
    """Add ``session`` to the registry under ``profile_id``.

    Idempotent — a session that calls this twice (e.g. cookie + voice
    both resolve the same profile) only ends up in the list once.
    """
    bucket = _SESSIONS_BY_PROFILE.setdefault(profile_id, [])
    if session not in bucket:
        bucket.append(session)
        log.info(
            "ws registry: session %d registered under profile=%d (n=%d)",
            session.session_id, profile_id, len(bucket),
        )


def unregister_session_profile(profile_id: int, session: "Session") -> None:
    """Drop ``session`` from the registry.  Safe to call multiple times."""
    bucket = _SESSIONS_BY_PROFILE.get(profile_id)
    if not bucket:
        return
    try:
        bucket.remove(session)
    except ValueError:
        return
    if not bucket:
        _SESSIONS_BY_PROFILE.pop(profile_id, None)
    log.info(
        "ws registry: session %d unregistered from profile=%d (n=%d)",
        session.session_id, profile_id, len(bucket),
    )


async def notify_voicemail_arrived(
    *,
    to_profile_id: int,
    message_id: int,
    from_name: str | None,
    duration_ms: int,
) -> int:
    """Push a ``voicemail_arrived`` event to every active session for this profile.

    Returns the number of sessions notified.  Best-effort — a failed
    ``send_json`` (closed socket etc.) is swallowed because the WS
    cleanup will deregister the stale session shortly.

    Additive Web Push step at the end: fan out a closed-browser ping to
    every PushSubscription the recipient has registered.  Push failures
    are swallowed — they shouldn't impact the live WS notification or
    the rest of the pipeline turn.
    """
    sessions = list(_SESSIONS_BY_PROFILE.get(to_profile_id, ()))
    payload = {
        "type": "voicemail_arrived",
        "message_id": message_id,
        "from_name": from_name,
        "duration_ms": duration_ms,
    }
    n = 0
    for s in sessions:
        try:
            await s._send(payload)
            n += 1
        except Exception:
            log.debug(
                "voicemail_arrived: send to session %d failed (likely closed)",
                s.session_id,
                exc_info=True,
            )
    # Closed-browser delivery via Web Push.  We resolve the recipient's
    # language for the localised body — read_settings is a tiny per-user
    # JSON file, cheap to read on demand.  ``from_name`` falls back to
    # the localised ``inbox.guest_sender`` when the voice didn't match.
    try:
        from . import push  # local import to keep this module import-cheap
        from .i18n import t
        from .user_files import read_settings

        settings = await read_settings(to_profile_id)
        # ``settings.language`` may be "auto" — pick_lang would resolve
        # from Whisper's detected lang, but we don't have that here.
        # Default to "en" (catalog fallback) when auto, or honour an
        # explicitly set lang.
        lang = settings.language if settings.language in ("en", "ru", "de") else "en"
        display_name = from_name or t("inbox.guest_sender", lang)
        push_payload = {
            "title": t("push.voicemail_title", lang),
            "body": t("push.voicemail_body", lang, from_name=display_name),
            "voicemail_id": message_id,
            "tag": f"voicemail-{message_id}",
        }
        sent = await push.send_to_profile(to_profile_id, push_payload)
        if sent:
            log.info(
                "voicemail_arrived: push delivered to %d subscription(s) for profile=%d",
                sent, to_profile_id,
            )
    except Exception:
        log.warning(
            "voicemail_arrived: push fan-out failed for profile=%d",
            to_profile_id, exc_info=True,
        )
    return n
