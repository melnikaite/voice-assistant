"""
Active WebSocket session registry.

Keyed by client_id (the persistent browser UUID).  Lets the scheduler push
TTS messages to whichever tab is currently open, without knowing anything
about the WebSocket layer.

Thread-safety note: all mutations happen on the asyncio event loop thread,
so a plain dict is sufficient — no locking needed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ws import Session

log = logging.getLogger(__name__)

# client_id → active Session object
_sessions: dict[str, "Session"] = {}


def register(client_id: str, session: "Session") -> None:
    _sessions[client_id] = session
    log.debug("registry: registered %.8s… (%d total)", client_id, len(_sessions))


def unregister(client_id: str) -> None:
    _sessions.pop(client_id, None)
    log.debug("registry: unregistered %.8s… (%d total)", client_id, len(_sessions))


async def push(client_id: str, text: str, *, reason: str = "reminder") -> bool:
    """
    Deliver a server-initiated TTS message to the active session for
    ``client_id``.  Returns True if the session was found and audio was
    enqueued, False if no session is currently connected.

    Routing has changed since the WebRTC migration: instead of asking the
    browser to render the text via Web Speech API (``push_tts`` JSON
    message), we synthesise it server-side via Piper and push the PCM
    straight into the session's outbound RTC track.  The browser hears
    the assistant speak naturally, with full AEC coverage if it bargers
    in.  Also surface a JSON ``push_tts`` event so the UI can flash a
    notification banner alongside the audio.
    """
    session = _sessions.get(client_id)
    if session is None:
        log.warning("push: no active session for %.8s…, message not delivered", client_id)
        return False
    try:
        await session._send({"type": "push_tts", "text": text, "reason": reason})
        await session.play_notification(text)
        log.info("push → %.8s…: %r", client_id, text[:80])
        return True
    except Exception as exc:
        log.warning("push to %.8s… failed: %s", client_id, exc)
        return False


async def broadcast_ptt_trigger(hold_ms: int = 10_000) -> int:
    """Send a ``ptt_trigger`` event to every currently-connected session.

    Called by the ``/api/hotkey/ptt`` endpoint when the desktop-agent's
    global hotkey fires.  ``hold_ms`` is the auto-release timeout the
    browser uses as a safety ceiling (in case the user doesn't press the
    hotkey a second time to stop).

    Returns the number of sessions that received the message.
    Best-effort — failed sends are swallowed (closed socket etc.).
    """
    payload = {"type": "ptt_trigger", "hold_ms": hold_ms}
    n = 0
    for session in list(_sessions.values()):
        try:
            await session._send(payload)
            n += 1
        except Exception:
            log.debug(
                "ptt_trigger: send to %.8s… failed (likely closed)",
                getattr(session, "client_id", "?"),
                exc_info=True,
            )
    log.info("ptt_trigger: sent to %d session(s)", n)
    return n


async def broadcast_step_up_granted(
    client_id: str,
    profile_id: int,
    window_s: int = 300,
) -> int:
    """Elevate the session that requested a step-up grant.

    Called from ``/api/step-up/approve`` after the push notification is
    tapped.  ``client_id`` and ``profile_id`` come from the consumed grant
    row — NOT from the request body — so a leaked token can't elevate an
    attacker-chosen session.  We additionally verify the live session's
    ``auth_profile_id`` matches the grant's ``profile_id`` (defence in
    depth: the client_id slot must still belong to the same profile that
    requested the grant).  Sets ``_step_up_auth_until`` so the next voice
    turn's AgentContext has ``step_up_auth=True``.

    Returns 1 if the matching session was elevated, 0 otherwise.
    """
    session = _sessions.get(client_id)
    if session is None:
        log.info("step_up_granted: no active session for %.8s…", client_id)
        return 0
    # Defence in depth: the session occupying this client_id slot must be
    # the same profile the grant was minted for.  Guards against a
    # client_id being recycled by a different profile between grant
    # creation and approval.
    session_pid = getattr(session, "auth_profile_id", None)
    if session_pid is not None and session_pid != profile_id:
        log.warning(
            "step_up_granted: profile mismatch for %.8s… (grant=%s, session=%s) — refused",
            client_id, profile_id, session_pid,
        )
        return 0
    try:
        await session.on_step_up_granted(window_s=window_s)
        log.info(
            "step_up_granted → %.8s… (profile=%s, window=%ds)",
            client_id, profile_id, window_s,
        )
        return 1
    except Exception:
        log.warning("step_up_granted: send to %.8s… failed", client_id, exc_info=True)
        return 0
