"""
Tests for the Chrome CDP browser tools (#51).

Scope:
  * list_browser_tabs  — happy path + CDPunavailable + desktop offline
  * navigate_browser   — happy path (new tab) + CDPunavailable
  * read_browser_tab   — happy path (page_text → LLM Q&A) + CDPunavailable
                       + empty page text fallback

All mocked at the desktop_client layer — no real Chrome, no real
desktop-agent, no real LLM required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ── shared fixtures ────────────────────────────────────────────────────────


class _StubCtx:
    """Minimal AgentContext stub — matches the fields ToolCtx reads."""

    def __init__(self):
        self.client_id = "test-client"
        self.user_lang = "en"
        self.profile_id = 1
        self.is_authenticated = True
        self.progress_sink = None
        self.stream_sink = None


@pytest.fixture
def ctx():
    return _StubCtx()


_SAMPLE_TABS = [
    {"id": "A1", "title": "GitHub", "url": "https://github.com",
     "type": "page", "ws_url": "ws://127.0.0.1:9222/devtools/page/A1"},
    {"id": "B2", "title": "YouTube", "url": "https://youtube.com",
     "type": "page", "ws_url": "ws://127.0.0.1:9222/devtools/page/B2"},
]


def _reachable_patch():
    """Patch desktop_client.is_reachable to return True."""
    return patch("app.desktop_client.is_reachable", return_value=True)


def _unreachable_patch():
    """Patch desktop_client.is_reachable to return False."""
    return patch("app.desktop_client.is_reachable", return_value=False)


# ── list_browser_tabs ─────────────────────────────────────────────────────


async def test_list_tabs_happy_path(ctx):
    """Happy path: two open tabs → voice-friendly list in ToolResult.text."""
    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_list_tabs", new=AsyncMock(return_value=_SAMPLE_TABS)),
    ):
        from app.tools.browser_tabs import list_browser_tabs
        result = await list_browser_tabs(ctx=ctx)

    assert result.data.get("count") == 2
    assert "GitHub" in result.text
    assert "YouTube" in result.text


async def test_list_tabs_empty(ctx):
    """No tabs open → browser.no_tabs i18n message."""
    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_list_tabs", new=AsyncMock(return_value=[])),
    ):
        from app.tools.browser_tabs import list_browser_tabs
        result = await list_browser_tabs(ctx=ctx)

    assert "No Chrome tabs" in result.text
    assert result.data.get("tabs") == []


async def test_list_tabs_cdp_unavailable(ctx):
    """CDP 503 (Chrome not running with --remote-debugging-port) → friendly message."""
    from app.desktop_client import DesktopUnavailable
    with (
        _reachable_patch(),
        patch(
            "app.desktop_client.browser_list_tabs",
            new=AsyncMock(side_effect=DesktopUnavailable("Chrome not reachable on port 9222")),
        ),
    ):
        from app.tools.browser_tabs import list_browser_tabs
        result = await list_browser_tabs(ctx=ctx)

    # Should return the browser.cdp_unavailable i18n message, not raise.
    assert "Chrome" in result.text
    assert result.data.get("error") == "cdp_unavailable"


async def test_list_tabs_desktop_offline(ctx):
    """Agent unreachable → desktop.legacy_unavailable before we even try CDP."""
    with _unreachable_patch():
        from app.tools.browser_tabs import list_browser_tabs
        result = await list_browser_tabs(ctx=ctx)

    assert result.data.get("error") == "desktop_unavailable"


# ── navigate_browser ──────────────────────────────────────────────────────


async def test_navigate_browser_new_tab(ctx):
    """Bare URL is normalised (https:// added) and a new tab is opened."""
    nav_result = {"ok": True, "tab_id": "C3", "url": "https://github.com"}
    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_navigate", new=AsyncMock(return_value=nav_result)),
    ):
        from app.tools.browser_navigate import navigate_browser
        result = await navigate_browser("github.com", ctx=ctx)

    assert "github.com" in result.text.lower()
    assert result.data.get("tab_id") == "C3"


async def test_navigate_browser_adds_https(ctx):
    """Bare domain without scheme → https:// is prepended before the call."""
    captured = {}

    async def _mock_navigate(url, *, tab_id=None, agent_id=None):
        captured["url"] = url
        return {"ok": True, "tab_id": "D4", "url": url}

    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_navigate", new=_mock_navigate),
    ):
        from app.tools.browser_navigate import navigate_browser
        await navigate_browser("example.com", ctx=ctx)

    assert captured["url"].startswith("https://")


async def test_navigate_cdp_unavailable(ctx):
    """CDP offline during navigate → browser.cdp_unavailable message."""
    from app.desktop_client import DesktopUnavailable
    with (
        _reachable_patch(),
        patch(
            "app.desktop_client.browser_navigate",
            new=AsyncMock(side_effect=DesktopUnavailable("Chrome not reachable")),
        ),
    ):
        from app.tools.browser_navigate import navigate_browser
        result = await navigate_browser("https://example.com", ctx=ctx)

    assert result.data.get("error") == "cdp_unavailable"


# ── read_browser_tab ──────────────────────────────────────────────────────


async def test_read_browser_tab_happy_path(ctx):
    """page_text → LLM → answer returned in ToolResult.text."""
    page_info = {
        "text": "Python is a programming language created by Guido van Rossum.",
        "title": "Python Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Python",
        "tab_id": "A1",
    }
    llm_choice = {
        "message": {"content": "Python is a programming language.", "role": "assistant"},
        "finish_reason": "stop",
    }
    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_page_text", new=AsyncMock(return_value=page_info)),
        patch("app.tools.browser_read.chat", new=AsyncMock(return_value=llm_choice)),
    ):
        from app.tools.browser_read import read_browser_tab
        result = await read_browser_tab("who created Python?", ctx=ctx)

    assert "Python" in result.text
    assert result.data.get("url") == "https://en.wikipedia.org/wiki/Python"
    assert result.data.get("chars_read") > 0


async def test_read_browser_tab_empty_page(ctx):
    """Empty page text → browser.read_failed without calling LLM."""
    page_info = {"text": "", "title": "Blank", "url": "about:blank", "tab_id": "Z9"}
    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_page_text", new=AsyncMock(return_value=page_info)),
    ):
        from app.tools.browser_read import read_browser_tab
        result = await read_browser_tab("what is this?", ctx=ctx)

    assert result.data.get("error") == "empty_page"


async def test_read_browser_tab_cdp_unavailable(ctx):
    """CDP offline → browser.cdp_unavailable without touching LLM."""
    from app.desktop_client import DesktopUnavailable
    with (
        _reachable_patch(),
        patch(
            "app.desktop_client.browser_page_text",
            new=AsyncMock(side_effect=DesktopUnavailable("Chrome not reachable")),
        ),
    ):
        from app.tools.browser_read import read_browser_tab
        result = await read_browser_tab("summarise this page", ctx=ctx)

    assert result.data.get("error") == "cdp_unavailable"


async def test_read_browser_tab_desktop_offline(ctx):
    """Agent unreachable → desktop.legacy_unavailable before CDP call."""
    with _unreachable_patch():
        from app.tools.browser_read import read_browser_tab
        result = await read_browser_tab("what is this?", ctx=ctx)

    assert result.data.get("error") == "desktop_unavailable"


async def test_read_browser_tab_truncates_long_text(ctx):
    """Text longer than PAGE_TEXT_LIMIT is truncated before being sent to LLM."""
    from app.tools.browser_read import PAGE_TEXT_LIMIT

    long_text = "x" * (PAGE_TEXT_LIMIT + 5000)
    page_info = {"text": long_text, "title": "Big Page", "url": "https://big.example.com", "tab_id": "E5"}
    llm_choice = {
        "message": {"content": "Short answer.", "role": "assistant"},
        "finish_reason": "stop",
    }

    captured_messages = {}

    async def _mock_chat(messages, **kwargs):
        captured_messages["msgs"] = messages
        return llm_choice

    with (
        _reachable_patch(),
        patch("app.desktop_client.browser_page_text", new=AsyncMock(return_value=page_info)),
        patch("app.tools.browser_read.chat", new=_mock_chat),
    ):
        from app.tools.browser_read import read_browser_tab
        result = await read_browser_tab("what is here?", ctx=ctx)

    # The user message should contain the truncation marker.
    user_content = captured_messages["msgs"][1]["content"]
    assert "truncated" in user_content
    # The ToolResult should record the full chars_read, not the truncated length.
    assert result.data.get("chars_read") == len(long_text)
