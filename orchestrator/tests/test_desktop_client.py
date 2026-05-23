"""
desktop-client — capability caching, health-poll, reachability and the
multi-agent registry (Wave 2 Phases 3a + 4).

We pin these contracts:

  • A failed capabilities() fetch keeps the stale cache + flips the
    agent's ``reachable`` flag to False (and marks the returned payload
    ``unreachable=True``) — so tools can fast-fail rather than block on
    a 5 s connect-timeout.
  • A subsequent successful poll restores ``reachable=True``.
  • ``DESKTOP_AGENTS`` env builds a multi-entry registry; ``get_agent``
    routes by id, defaults to ``default=True`` row, else first.
  • Back-compat: only ``DESKTOP_URL``/``DESKTOP_TOKEN`` set → one
    synthetic entry with id ``"default"``.

We don't try to stand up a real WSS server — the reverse-mode plumbing
gets its own focused test in test_agent_proxy.py.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest


@pytest.fixture
def reset_agents():
    """Wipe + restore the module-level agent registry around each test.

    desktop_client snapshots env at import time; mutating env mid-test
    requires an explicit reload.  We also clear the per-agent
    capability caches so a prior test's state doesn't leak.
    """
    from app import desktop_client
    saved_env = {
        "DESKTOP_AGENTS": os.environ.get("DESKTOP_AGENTS"),
        "DESKTOP_URL":    os.environ.get("DESKTOP_URL"),
        "DESKTOP_TOKEN":  os.environ.get("DESKTOP_TOKEN"),
    }
    yield desktop_client
    # Restore env exactly as it was; reload registry.
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    desktop_client.reload_agents_from_env()


# ── 1. capability caching on transport error ───────────────────────────


async def test_capability_caching_uses_stale_on_error(reset_agents):
    """A failing capabilities() probe keeps the prior cache + flips the
    agent's reachable flag to False; the returned payload carries an
    ``unreachable=True`` marker so callers can render the right message.
    """
    dc = reset_agents
    # Single-agent mode (default).  Seed the cache with a "good" prior
    # snapshot, then patch httpx to raise on the next call.
    info = dc.get_agent(None)
    assert info is not None
    info.capabilities_cache = {
        "agent_id": "default", "platform": "macos",
        "capabilities": {"applescript": True}, "version": "1.1.0",
    }
    info.capabilities_cached_at = 0.0  # force expiry → re-probe
    info.reachable = True

    class _FailingClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **kw):
            raise httpx.ConnectError("connection refused")

    with patch("app.desktop_client.httpx.AsyncClient", _FailingClient):
        caps = await dc.capabilities(force_refresh=True)

    # Cached data still surfaces — degrade gracefully — but flagged.
    assert caps.get("capabilities", {}).get("applescript") is True
    assert caps.get("unreachable") is True
    assert dc.is_reachable() is False


# ── 2. health-poll recovers a flaky agent ──────────────────────────────


async def test_health_poll_recovers_unreachable(reset_agents):
    """First probe fails → reachable=False.  Second succeeds → True.

    We exercise the capabilities() path directly (which is what the
    background poll loop calls) rather than waiting on the poll's
    asyncio.sleep — sleep-real-seconds in unit tests is too brittle.
    """
    dc = reset_agents
    info = dc.get_agent(None)
    assert info is not None
    info.reachable = False  # simulate "we've never seen it"
    info.capabilities_cached_at = 0.0

    class _ToggleClient:
        """First call raises, subsequent calls succeed."""

        calls = 0

        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def get(self, *a, **kw):
            type(self).calls += 1
            if type(self).calls == 1:
                raise httpx.ConnectError("flake")
            # 2nd call: success.
            class _R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {
                        "agent_id": "default",
                        "platform": "macos",
                        "capabilities": {"applescript": True},
                        "version": "1.1.0",
                    }
            return _R()

    with patch("app.desktop_client.httpx.AsyncClient", _ToggleClient):
        # First poll: fails → unreachable.
        await dc.capabilities(force_refresh=True)
        assert dc.is_reachable() is False
        # Second poll: succeeds → reachable flips back.
        await dc.capabilities(force_refresh=True)
        assert dc.is_reachable() is True


# ── 3. multi-agent registry from env ───────────────────────────────────


async def test_agent_registry_parses_env(reset_agents):
    """DESKTOP_AGENTS JSON env → multi-entry registry; routing works."""
    dc = reset_agents
    os.environ["DESKTOP_AGENTS"] = (
        '[{"agent_id":"mac","url":"http://m:9877","token":"t1","default":true},'
        ' {"agent_id":"pc","url":"http://p:9877","token":"t2"}]'
    )
    dc.reload_agents_from_env()
    agents = dc.list_agents()
    assert len(agents) == 2
    ids = {a.agent_id for a in agents}
    assert ids == {"mac", "pc"}
    # Default should be "mac" — explicit default=True.
    assert dc.get_agent(None).agent_id == "mac"
    # Explicit lookup.
    assert dc.get_agent("pc").agent_id == "pc"
    assert dc.get_agent("unknown") is None


async def test_agent_registry_back_compat_single(reset_agents):
    """No DESKTOP_AGENTS, only DESKTOP_URL/TOKEN → single 'default' entry.

    Existing installs must keep working with zero config changes.
    """
    dc = reset_agents
    os.environ.pop("DESKTOP_AGENTS", None)
    os.environ["DESKTOP_URL"] = "http://localhost:9877"
    os.environ["DESKTOP_TOKEN"] = "abc"
    dc.reload_agents_from_env()
    agents = dc.list_agents()
    assert len(agents) == 1
    info = agents[0]
    assert info.agent_id == "default"
    assert info.url == "http://localhost:9877"
    assert info.token == "abc"
    assert info.default is True


# ── 4. routing picks the right agent ───────────────────────────────────


async def test_routing_picks_default(reset_agents):
    """No explicit agent_id + multiple agents → default flag wins."""
    dc = reset_agents
    os.environ["DESKTOP_AGENTS"] = (
        '[{"agent_id":"a","url":"http://a:9877","token":"t"},'
        ' {"agent_id":"b","url":"http://b:9877","token":"t","default":true}]'
    )
    dc.reload_agents_from_env()
    assert dc.get_agent(None).agent_id == "b"


async def test_routing_explicit_agent_id(reset_agents):
    """Explicit agent_id overrides default."""
    dc = reset_agents
    os.environ["DESKTOP_AGENTS"] = (
        '[{"agent_id":"a","url":"http://a:9877","token":"t","default":true},'
        ' {"agent_id":"b","url":"http://b:9877","token":"t"}]'
    )
    dc.reload_agents_from_env()
    assert dc.get_agent("b").agent_id == "b"
    assert dc.get_agent(None).agent_id == "a"  # default still picks a


# ── 5. resolve_default_app + cursor_activity helpers (Wave 3) ──────────


async def test_resolve_default_app_uses_capability_cache(reset_agents):
    """Pre-warmed ``default_apps`` on the capabilities cache → no HTTP.

    The agent's /v1/capabilities response includes a ``default_apps``
    map per category.  When the orchestrator already has it cached
    we should return that without paying another round-trip.
    """
    dc = reset_agents
    info = dc.get_agent(None)
    assert info is not None
    info.capabilities_cache = {
        "agent_id": "default", "platform": "macos",
        "capabilities": {"default_apps_resolver": True},
        "default_apps": {
            "mail": {
                "app_name": "Mail", "app_path": "/Applications/Mail.app",
                "bundle_id": "com.apple.mail", "scriptable": True,
            },
        },
    }

    class _BoomClient:
        """No HTTP should happen — fail loudly if it does."""

        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **kw):
            raise AssertionError("HTTP must not be called when cache is warm")

    with patch("app.desktop_client.httpx.AsyncClient", _BoomClient):
        out = await dc.resolve_default_app("mail")
    assert out is not None
    assert out["app_name"] == "Mail"
    assert out["bundle_id"] == "com.apple.mail"
    assert out["scriptable"] is True


async def test_resolve_default_app_404_returns_none(reset_agents):
    """Agent reports no default for a category → helper returns None.

    Not an exception — the caller (computer_use) prefers a clean
    "no app" branch to a try/except per call.
    """
    dc = reset_agents
    info = dc.get_agent(None)
    assert info is not None
    info.capabilities_cache = {}  # force the HTTP path

    class _NotFoundClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **kw):
            class _R:
                status_code = 404
                text = "no default"
                def raise_for_status(self): pass
                def json(self): return {}
            return _R()

    with patch("app.desktop_client.httpx.AsyncClient", _NotFoundClient):
        out = await dc.resolve_default_app("mail")
    assert out is None


async def test_cursor_activity_returns_payload(reset_agents):
    """``/v1/cursor_activity`` JSON is passed through to the caller."""
    dc = reset_agents

    class _OkClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **kw):
            class _R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {"x": 100, "y": 200, "idle_s": 42.5, "warm": True}
            return _R()

    with patch("app.desktop_client.httpx.AsyncClient", _OkClient):
        snap = await dc.cursor_activity()
    assert snap == {"x": 100, "y": 200, "idle_s": 42.5, "warm": True}
