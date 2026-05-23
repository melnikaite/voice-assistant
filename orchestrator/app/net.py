"""
Shared network-availability probe for tools that depend on the internet.

Several tools (web_search, news_briefing, calculator/currency, weather)
hit third-party endpoints — they should fail FAST with a clean,
localized message when the host has no internet, instead of taking the
full HTTP timeout and surfacing a cryptic transport error.

This module owns one tiny utility: :func:`has_internet`, an async
probe with a short positive cache so repeated tool calls in the same
turn don't repeat the network round-trip.  The negative result is
cached for an even shorter window so the moment connectivity returns
the next call discovers it.

Probe target: DuckDuckGo's lite homepage.  Tiny payload (~3 KB), no
auth, very high uptime, and it's the same domain we already hit for
web_search — so a host that fails this probe genuinely cannot reach
the open internet.  No DNS lookup races because we let httpx resolve.

The probe deliberately does NOT verify the orchestrator's own
backends (LM Studio, whisper, xtts, desktop-agent) — those run on
the host and are reachable even when the household's internet is
down.  That distinction matters: we want offline-mode to disable
"web" tools but keep the voice loop usable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

# Probe URL — small, public, no auth.  Override via env if your
# environment blocks duckduckgo (some corp networks do).
_PROBE_URL = os.environ.get("OFFLINE_PROBE_URL", "https://duckduckgo.com/lite/")
# Tight timeout — we'd rather call ourselves "offline" pessimistically
# than make a tool wait 8s before refusing.
_PROBE_TIMEOUT_S = float(os.environ.get("OFFLINE_PROBE_TIMEOUT_S", "1.5"))
# How long a POSITIVE result stays in cache (seconds).  Long enough
# to cover the typical turn, short enough that we re-check on the
# scale of minutes if the network drops.
_POS_CACHE_S = 60.0
# Negative result cache (seconds).  Shorter so recovery is fast.
_NEG_CACHE_S = 10.0

_cached_result: bool | None = None
_cached_until: float = 0.0
_lock = asyncio.Lock()


async def has_internet() -> bool:
    """Return True iff the host can reach the open internet right now.

    Cached for a few seconds either way.  Never raises — a probe
    failure is treated as "offline".  Safe to call from any tool.
    """
    global _cached_result, _cached_until
    now = time.time()
    if _cached_result is not None and _cached_until > now:
        return _cached_result
    async with _lock:
        # Double-check after acquiring the lock — another coroutine may
        # have just refreshed the cache while we were waiting.
        now = time.time()
        if _cached_result is not None and _cached_until > now:
            return _cached_result
        result = await _probe()
        ttl = _POS_CACHE_S if result else _NEG_CACHE_S
        _cached_result = result
        _cached_until = now + ttl
        log.info("net: probe → %s (cached %.0fs)", "online" if result else "offline", ttl)
        return result


async def _probe() -> bool:
    """Single probe attempt; tight timeout; never raises."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as c:
            r = await c.get(_PROBE_URL)
            return r.status_code < 500
    except Exception as exc:
        log.debug("net: probe failed: %s", exc)
        return False


def invalidate_cache() -> None:
    """Force the next :func:`has_internet` call to re-probe.

    Call this from a tool that just observed a connectivity error so
    subsequent tools in the same turn correctly read offline.
    """
    global _cached_result, _cached_until
    _cached_result = None
    _cached_until = 0.0
