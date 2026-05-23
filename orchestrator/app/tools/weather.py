"""
weather — current conditions + short-term forecast via Open-Meteo.

Open-Meteo is a free, no-API-key weather service (https://open-meteo.com).
We do a two-step call:

  1. Geocode the location name (or pass through "lat,lon" directly).
  2. Fetch current weather for those coordinates.

The geocode result is cached in-memory for 1 hour — most users ask about
the same one or two cities repeatedly. The forecast itself is always
fetched fresh; it's the cheap call (no DB lookup on Open-Meteo's side).

Weather codes are mapped via the WMO 4677 table:
  https://open-meteo.com/en/docs#weathervariables
WMO → spoken phrase comes from ``i18n.weather_phrase`` per locale.
"""
from __future__ import annotations

import logging
import re
import time

import httpx

from ..i18n import t, weather_phrase
from ..net import has_internet
from .base import ToolResult, tool

log = logging.getLogger(__name__)

# ── In-memory geocode cache ─────────────────────────────────────────────
# Keyed by the lowercased lookup string.  Values: (lat, lon, display_name,
# expires_ts).  1-hour TTL is plenty: a city doesn't move, but the cache
# also doubles as a TTL-bounded record of recently-asked locations so
# the dict doesn't grow unboundedly across a long-running orchestrator.
_GEOCODE_CACHE: dict[str, tuple[float, float, str, float]] = {}
_GEOCODE_TTL_S = 3600


_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


async def _geocode(client: httpx.AsyncClient, location: str) -> tuple[float, float, str] | None:
    """Resolve a location name to (lat, lon, display_name).

    Accepts "lat,lon" pairs as a fast-path; otherwise hits Open-Meteo's
    geocoding API with ``language=ru`` so display names come back
    Russianised when possible — matching the Russian-locale spoken
    reply.  Pin this to the user's locale when adding more locales.
    """
    # Fast path: explicit lat/lon — skip the API call entirely.
    m = _LATLON_RE.match(location)
    if m:
        try:
            lat = float(m.group(1))
            lon = float(m.group(2))
        except ValueError:
            return None
        return lat, lon, f"{lat:.4f}, {lon:.4f}"

    key = location.strip().lower()
    now = time.time()
    hit = _GEOCODE_CACHE.get(key)
    if hit and hit[3] > now:
        return hit[0], hit[1], hit[2]

    try:
        r = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "ru", "format": "json"},
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("weather geocode: http error: %s", e)
        return None

    data = r.json() or {}
    results = data.get("results") or []
    if not results:
        return None
    res = results[0]
    try:
        lat = float(res["latitude"])
        lon = float(res["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    name = res.get("name") or location
    country = res.get("country") or ""
    display = f"{name}, {country}".rstrip(", ")
    _GEOCODE_CACHE[key] = (lat, lon, display, now + _GEOCODE_TTL_S)
    return lat, lon, display


def _format_reply(
    display_name: str, temp_c: float, code: int, wind_kmh: float,
    lang: str | None,
) -> str:
    """One spoken sentence summarising the current weather, locale-aware."""
    phrase = weather_phrase(code, lang)
    # Prefix " +" for non-negative so TTS reads "plus fourteen" instead
    # of just "fourteen".  Temperature is rounded to whole degrees —
    # fractional precision is noise over speech.
    t_round = int(round(temp_c))
    sign = "+" if t_round >= 0 else ""
    w_round = int(round(wind_kmh))
    return t(
        "weather.reply", lang,
        place=display_name, sign=sign, temp=t_round,
        phrase=phrase, wind=w_round,
    )


@tool(
    name="weather",
    description=(
        "Get current weather conditions for any city or coordinates. "
        "Use for questions like 'what's the weather in Berlin', 'weather in "
        "Moscow', 'how many degrees in London right now' in any supported "
        "language. Argument `location` accepts a city name in any language "
        "or a 'lat,lon' pair."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City name (in any language) or 'lat,lon' coordinates. "
                    "Examples: 'Berlin', 'New York', '52.52,13.41'."
                ),
            },
        },
        "required": ["location"],
    },
    risk="read",
)
async def weather(location: str, *, ctx=None) -> ToolResult:
    progress = getattr(ctx, "progress_sink", None) if ctx else None
    lang = getattr(ctx, "user_lang", None) if ctx else None

    async def _progress(step: str, detail: str | None = None) -> None:
        if progress is not None:
            await progress(step, detail)

    if not location or not location.strip():
        return ToolResult(
            text=t("weather.no_location", lang), data={"error": "no_location"}
        )

    # Open-Meteo is a third-party endpoint — short-circuit if we know
    # the host is offline.  The voice loop and local tools keep working;
    # only the weather query degrades.
    if not await has_internet():
        return ToolResult(
            text=t("offline.for_tool", lang, what=t("tool.weather", lang)),
            data={"error": "offline", "location": location},
        )

    log.info("weather: %r", location)

    async with httpx.AsyncClient(timeout=8.0) as client:
        await _progress("geocode", location)
        geo = await _geocode(client, location)
        if geo is None:
            return ToolResult(
                text=t("weather.city_not_found", lang),
                data={"location": location, "error": "geocode_not_found"},
            )
        lat, lon, display = geo

        await _progress("forecast", display)
        try:
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "forecast_days": 1,
                    "timezone": "auto",
                    "wind_speed_unit": "kmh",
                },
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("weather forecast: http error: %s", e)
            return ToolResult(
                text=t("weather.fetch_failed", lang),
                data={"location": display, "error": f"{e.__class__.__name__}"},
            )

        data = r.json() or {}
        current = data.get("current") or {}
        try:
            temp_c = float(current["temperature_2m"])
            code = int(current["weather_code"])
            wind_kmh = float(current.get("wind_speed_10m") or 0.0)
        except (KeyError, TypeError, ValueError) as e:
            log.warning("weather: malformed response: %s (%r)", e, current)
            return ToolResult(
                text=t("weather.fetch_failed", lang),
                data={"location": display, "error": "bad_response"},
            )

    reply_text = _format_reply(display, temp_c, code, wind_kmh, lang)
    return ToolResult(
        text=reply_text,
        data={
            "location": display,
            "lat": lat,
            "lon": lon,
            "temperature_c": temp_c,
            "weather_code": code,
            "wind_kmh": wind_kmh,
        },
    )
