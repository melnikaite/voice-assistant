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
