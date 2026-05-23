import io
import logging
import os
import time
import wave

import httpx

log = logging.getLogger(__name__)

WHISPER_URL = os.environ["WHISPER_URL"]
WHISPER_MODEL = os.environ["WHISPER_MODEL"]

# `initial_prompt` biases Whisper toward expected vocabulary. Critical for
# foreign proper nouns / loanwords across the user's spoken languages —
# without this, "Schmetterling" can come back mangled when the user is
# speaking another language and Whisper guesses a wrong transcription
# for a borrowed word.
#
# Per-locale ``asr.whisper_hint`` strings in every ``locales/*.json``
# are concatenated at startup.  Adding a new language = drop a JSON
# with its own hint section; this code doesn't change.  Override the
# whole thing via ``WHISPER_INITIAL_PROMPT`` env if you need to.
from . import i18n


def _build_initial_prompt() -> str:
    parts: list[str] = []
    for code in i18n.SUPPORTED_LANGS:
        hint = i18n.t("asr.whisper_hint", code)
        if hint and hint != "asr.whisper_hint":  # key-missing sentinel
            parts.append(hint.strip())
    return " ".join(parts)


DEFAULT_INITIAL_PROMPT = os.environ.get(
    "WHISPER_INITIAL_PROMPT",
    _build_initial_prompt(),
)
DEFAULT_TEMPERATURE = float(os.environ.get("WHISPER_TEMPERATURE", "0.0"))


def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw Int16 mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def transcribe(
    pcm: bytes,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> tuple[str, int]:
    """
    Send PCM bytes to the whisper server, return ``(text, elapsed_ms)``.

    Note on language detection: the server we run (vLLM Whisper) only
    accepts ``response_format`` ∈ {``json``, ``text``} — ``verbose_json``
    returns 422 (vLLM issue #14818, marked stale).  So no per-utterance
    detected-language code comes back from Whisper.  Downstream callers
    that need the language do their own (e.g. ``web_search`` runs a tiny
    Gemma classification on the query before deciding whether to
    translate it).
    """
    wav = _pcm_to_wav(pcm)
    files = {"file": ("audio.wav", wav, "audio/wav")}
    data: dict[str, str] = {
        "model": WHISPER_MODEL,
        "temperature": str(DEFAULT_TEMPERATURE),
    }
    if language:
        data["language"] = language
    prompt = initial_prompt if initial_prompt is not None else DEFAULT_INITIAL_PROMPT
    if prompt:
        # OpenAI's audio API parameter is "prompt".
        data["prompt"] = prompt
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{WHISPER_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
        )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if r.status_code != 200:
        log.error("whisper error %s: %s", r.status_code, r.text[:500])
        r.raise_for_status()
    text = (r.json().get("text") or "").strip()
    log.info("asr: %dms, %d chars: %r", elapsed_ms, len(text), text[:120])
    return text, elapsed_ms
