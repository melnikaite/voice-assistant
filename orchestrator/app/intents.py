"""
Local intent matching — handles common commands without an LLM round-trip.

Patterns are checked against the ASR transcript before the agent loop runs.
A match short-circuits both the LLM call and any new TTS synthesis.

Two flavours of intent:
  * "replay_*" — purely TTS-layer commands. The WS layer sends the client a
    `{"type": "replay", ...}` message; the browser replays its cached
    `tts.lastText` from start or from `tts.resumeFromCharIdx`. No new
    Whisper round-trip is wasted on regenerating identical audio.
  * "new_topic" — backend-state command. Clears conversation history.

The regex patterns themselves live in ``locales/<lang>.json`` under the
``intents`` section.  ``i18n.intent_patterns`` compiles them once per
locale and caches the result, so adding a new language is a JSON-only
change.
"""
import logging

from .i18n import intent_patterns

log = logging.getLogger(__name__)


def match_intent(transcript: str, lang: str | None = None) -> str | None:
    """
    Return an intent name if the transcript matches a local command pattern,
    otherwise return None (→ proceed to LLM pipeline).

    Patterns come from the locale JSON for ``lang`` (falls back to English
    when ``lang`` is None or the locale is missing).  First match wins —
    JSON declaration order is preserved.
    """
    text = transcript.strip()
    for pattern, name in intent_patterns(lang):
        if pattern.search(text):
            log.info("local intent (%s): %r → %s", lang or "en", text[:80], name)
            return name
    return None
