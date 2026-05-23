"""
Locale runtime — single point of access for every user-facing string,
currency name, weather phrase, intent regex, number-to-words, and
duration phrasing.

Adding a new language is **one file**: drop ``locales/<code>.json``
and the rest of the codebase picks it up.  No Python edits.

Source of truth is ``orchestrator/app/locales/<code>.json``.  Each file
has the same shape:

    {
      "_meta":       {"code": "...", "babel_locale": "...", "num2words_lang": "..."},
      "messages":    { "<key.path>": "<template with {placeholders}>" },
      "currencies":  { "names": { "<ISO>": "<noun>" },
                       "aliases": { "<spoken phrase>": "<ISO>" } },
      "weather_codes": { "<WMO int>": "<phrase>" },
      "intents":     { "<intent_name>": [ "<regex>", ... ] }
    }

Number-to-words and duration phrasing don't live in the JSON: the
``num2words`` and ``Babel`` libraries already know how to do them
correctly for 40+ languages including English/Russian/German.  Adding
a language just means ``num2words`` + ``Babel`` know about it — they
do.

Public API
──────────
* ``t(key, lang, **fmt)``                — render a message template.
* ``num_to_words(n, lang)``               — integer → spoken words (num2words).
* ``currency_name(code, lang)``           — ISO code → spoken plural noun.
* ``currency_alias(phrase, lang)``        — spoken phrase → ISO code.
* ``weather_phrase(wmo_code, lang)``      — WMO code → phrase.
* ``intent_patterns(lang)``               — locale-bound compiled regexes.
* ``patterns_for_intent(intent_name)``    — cross-locale compiled regexes.
* ``format_duration_seconds(secs, lang)`` — duration phrase (Babel).
* ``format_when(timestamp_s, now, lang)`` — absolute/relative time (Babel).
* ``pick_lang(settings_lang, detected_lang)`` — resolve "auto" / fall back.
* ``SUPPORTED_LANGS``                     — tuple of available locale codes.

Falls back to English for any key/code/intent missing in the requested
locale; falls back to ``str(value)`` for number/duration formatting if
``num2words`` / ``Babel`` doesn't know the language.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_LANG = "en"

# ──────────────────────────────────────────────────────────────────────
# JSON load
# ──────────────────────────────────────────────────────────────────────

_LOCALES_DIR = Path(__file__).parent / "locales"
_LOCALES: dict[str, dict[str, Any]] = {}
_INTENT_CACHE: dict[str, list[tuple[re.Pattern, str]]] = {}


def _load() -> None:
    """Load every ``locales/*.json`` into the in-memory registry.

    Called at import time and re-callable for tests.  English MUST
    exist as the fallback; missing it is a hard error because every
    other locale's missing keys resolve to it.
    """
    _LOCALES.clear()
    _INTENT_CACHE.clear()
    for path in sorted(_LOCALES_DIR.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            log.exception("locale: failed to load %s — skipping", path.name)
            continue
        code = data.get("_meta", {}).get("code") or path.stem
        _LOCALES[code] = data
        log.info("locale: loaded %s (%d messages)", code, len(data.get("messages", {})))
    if DEFAULT_LANG not in _LOCALES:
        raise RuntimeError(
            f"locale: missing required {DEFAULT_LANG!r} locale at "
            f"{_LOCALES_DIR / (DEFAULT_LANG + '.json')}"
        )


_load()

# Tuple, in declaration order: en first (the fallback), then others sorted.
SUPPORTED_LANGS: tuple[str, ...] = (DEFAULT_LANG,) + tuple(
    sorted(c for c in _LOCALES if c != DEFAULT_LANG)
)


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _locale(lang: str | None) -> dict[str, Any]:
    """Return the locale dict for ``lang``, falling back to English."""
    if lang and lang in _LOCALES:
        return _LOCALES[lang]
    return _LOCALES[DEFAULT_LANG]


def _get(lang: str | None, section: str, key: str) -> Any | None:
    """Look up ``section[key]`` in ``lang``; fall back to English; else None."""
    lc = _locale(lang)
    val = (lc.get(section) or {}).get(key)
    if val is not None:
        return val
    if lang != DEFAULT_LANG:
        return (_LOCALES[DEFAULT_LANG].get(section) or {}).get(key)
    return None


# ──────────────────────────────────────────────────────────────────────
# User-facing message templates
# ──────────────────────────────────────────────────────────────────────


def t(key: str, lang: str | None = None, **fmt: Any) -> str:
    """Render a user-facing string from the locale JSON.

    ``key`` is dot-separated (e.g. ``"reminders.timer_set"``), looked
    up under the locale's ``messages`` section.  ``lang`` falls back
    to ``DEFAULT_LANG`` if unset or unsupported.  Missing keys log a
    warning and return the key string itself — typos are visible
    instead of silently returning ``""``.

    Format placeholders use Python ``str.format`` — keep them named
    (``{summary}``, not ``{0}``) so translators can reorder.
    """
    template = _get(lang, "messages", key)
    if template is None:
        log.warning("i18n: missing key %r for lang=%r", key, lang)
        return key
    if not fmt:
        return template
    try:
        return template.format(**fmt)
    except KeyError as exc:
        log.warning("i18n: missing placeholder %s in %r → %r", exc, key, template)
        return template


def pick_lang(*, settings_lang: str | None, detected_lang: str | None) -> str:
    """Resolve the language for the current turn.

    Precedence:
      1. ``settings_lang`` if set to a supported code.
      2. ``detected_lang`` (from Whisper) if supported.
      3. ``DEFAULT_LANG``.

    ``"auto"`` in settings means "use whatever Whisper detected".
    """
    if settings_lang and settings_lang in _LOCALES:
        return settings_lang
    if detected_lang and detected_lang in _LOCALES:
        return detected_lang
    return DEFAULT_LANG


# ──────────────────────────────────────────────────────────────────────
# Currencies
# ──────────────────────────────────────────────────────────────────────


def currency_name(code: str, lang: str | None = None) -> str:
    """Spoken noun form for ISO currency ``code`` (e.g. EUR → 'euros').

    Falls back to English, then to the ISO code uppercased so the TTS
    has SOMETHING to read (better than crashing on a missing locale).
    """
    code = code.upper()
    name = _get(lang, "currencies", "names")
    if name and code in name:
        return name[code]
    return code


def currency_alias(phrase: str, lang: str | None = None) -> str | None:
    """Resolve a spoken phrase to an ISO currency code, or None.

    Searches the locale's aliases first, then English (handles
    cross-language vocabulary in mixed-locale settings).  Lowercase
    + stripped before lookup.
    """
    s = phrase.strip().lower()
    aliases = _get(lang, "currencies", "aliases") or {}
    if s in aliases:
        return aliases[s]
    if lang != DEFAULT_LANG:
        en_aliases = (_LOCALES[DEFAULT_LANG].get("currencies") or {}).get("aliases") or {}
        if s in en_aliases:
            return en_aliases[s]
    return None


# ──────────────────────────────────────────────────────────────────────
# Weather
# ──────────────────────────────────────────────────────────────────────


def weather_phrase(wmo_code: int, lang: str | None = None) -> str:
    """Human-readable phrase for an Open-Meteo WMO weather code.

    Falls back to a localised "weather unclear" if the code is
    unknown in the locale (e.g. a new WMO code added upstream that
    the locale JSON hasn't been updated for).
    """
    table = _get(lang, "weather_codes", str(wmo_code))
    if table is not None:
        return table
    return t("weather.fallback_phrase", lang)


# ──────────────────────────────────────────────────────────────────────
# Intents (local zero-LLM phrase matching)
# ──────────────────────────────────────────────────────────────────────


def intent_patterns(lang: str | None = None) -> list[tuple[re.Pattern, str]]:
    """List of (compiled regex, intent_name) for the given locale.

    Order is preserved from the JSON — callers do first-match-wins.
    Compiled patterns are cached per locale.
    """
    code = lang if lang and lang in _LOCALES else DEFAULT_LANG
    if code in _INTENT_CACHE:
        return _INTENT_CACHE[code]
    raw = (_LOCALES[code].get("intents") or {})
    compiled: list[tuple[re.Pattern, str]] = []
    for name, patterns in raw.items():
        for pat in patterns:
            try:
                compiled.append((re.compile(pat, re.IGNORECASE), name))
            except re.error as exc:
                log.warning("i18n: bad intent regex %r for %s/%s: %s", pat, code, name, exc)
    _INTENT_CACHE[code] = compiled
    return compiled


_CROSS_LOCALE_INTENT_CACHE: dict[str, list[re.Pattern]] = {}


def patterns_for_intent(intent_name: str) -> list[re.Pattern]:
    """Compiled patterns for ``intent_name`` across ALL locales.

    Use when an intent should fire regardless of the user's current
    language — e.g. voicemail (you might say "tell Anna…" in English
    even with Russian as the session language).  Locale-bound
    intents (replay, new_topic) should use ``intent_patterns(lang)``
    instead so a Russian replay command doesn't match a German user.
    """
    if intent_name in _CROSS_LOCALE_INTENT_CACHE:
        return _CROSS_LOCALE_INTENT_CACHE[intent_name]
    out: list[re.Pattern] = []
    for code in _LOCALES:
        for pat in (_LOCALES[code].get("intents") or {}).get(intent_name, []):
            try:
                out.append(re.compile(pat, re.IGNORECASE))
            except re.error as exc:
                log.warning("i18n: bad cross-locale intent %r/%s: %s", intent_name, code, exc)
    _CROSS_LOCALE_INTENT_CACHE[intent_name] = out
    return out


# ──────────────────────────────────────────────────────────────────────
# Numbers and durations — delegated to num2words / Babel
# ──────────────────────────────────────────────────────────────────────


def _num2words_lang(lang: str | None) -> str:
    """Map our locale code to what ``num2words`` expects."""
    return (_locale(lang).get("_meta") or {}).get("num2words_lang", DEFAULT_LANG)


def _babel_locale(lang: str | None) -> str:
    """Map our locale code to a Babel locale identifier."""
    return (_locale(lang).get("_meta") or {}).get("babel_locale", "en_US")


def num_to_words(n: int, lang: str | None = None) -> str:
    """Render an integer as words in the requested locale.

    Uses ``num2words`` (40+ languages, handles grammar correctly).
    Returns ``str(n)`` if num2words doesn't know the language — the
    caller is then free to splice digits into the spoken reply
    instead of failing.
    """
    try:
        from num2words import num2words  # local import keeps startup cheap
    except ImportError:
        log.warning("i18n: num2words missing; falling back to digits")
        return str(n)
    try:
        return num2words(n, lang=_num2words_lang(lang))
    except NotImplementedError:
        return str(n)
    except Exception:
        log.exception("i18n: num2words failed for %r/%r", n, lang)
        return str(n)


def format_duration_seconds(seconds: int, lang: str | None = None) -> str:
    """Spoken duration phrase via Babel's ``format_timedelta``.

    Babel knows CLDR pluralisation rules for every locale it supports.
    Returns ``"{n}s"`` style
    fallback if Babel isn't available.
    """
    try:
        from babel.dates import format_timedelta
    except ImportError:
        log.warning("i18n: Babel missing; falling back to seconds")
        return f"{seconds}s"
    td = timedelta(seconds=int(seconds))
    try:
        return format_timedelta(td, locale=_babel_locale(lang), granularity="second")
    except Exception:
        log.exception("i18n: format_timedelta failed for %r/%r", seconds, lang)
        return f"{seconds}s"


def format_when(fire_ts: float, now: float, lang: str | None = None) -> str:
    """Render an absolute timestamp as a spoken phrase relative to ``now``.

    < 60 minutes out — "in 10 minutes" via Babel relative-time.
    Same day / next day — "today at 15:30" / "tomorrow at 09:00",
    locale-formatted via Babel.
    Further — locale date + time.
    """
    delta = fire_ts - now
    if delta < 60:
        return t("pending.age.expiring_soon", lang)
    # Within 60 minutes: relative phrasing.
    try:
        from babel.dates import format_datetime, format_timedelta
    except ImportError:
        return f"in {int(delta)}s"
    try:
        bcl = _babel_locale(lang)
        if delta < 3600:
            return format_timedelta(
                timedelta(seconds=int(delta)),
                add_direction=True, locale=bcl, granularity="minute",
            )
        fire_dt = datetime.fromtimestamp(fire_ts)
        now_dt = datetime.fromtimestamp(now)
        same_day = fire_dt.date() == now_dt.date()
        tomorrow = (fire_dt.date() - now_dt.date()).days == 1
        if same_day:
            return format_datetime(fire_dt, format="'today at' HH:mm", locale=bcl)
        if tomorrow:
            return format_datetime(fire_dt, format="'tomorrow at' HH:mm", locale=bcl)
        return format_datetime(fire_dt, format="EEEE, d MMM, HH:mm", locale=bcl)
    except Exception:
        log.exception("i18n: format_when failed for %r/%r", fire_ts, lang)
        return f"in {int(delta)}s"


# ──────────────────────────────────────────────────────────────────────
# Test / debug helpers
# ──────────────────────────────────────────────────────────────────────


def all_keys(section: str = "messages") -> set[str]:
    """All keys defined in the English locale's section.

    Tests use this to assert that translations stay in sync — every
    EN key should resolve in every other locale (with EN fallback).
    """
    return set((_LOCALES[DEFAULT_LANG].get(section) or {}).keys())
