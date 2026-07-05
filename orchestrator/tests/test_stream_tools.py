"""
Tests for the MJPEG live-streaming tools (#53).

Scope:
  * stream_camera  — happy path + camera unavailable + desktop offline
  * stream_tab     — happy path + CDP unavailable + desktop offline
  * media_sink     — URL and source are passed through correctly
  * URL building   — fps and agent_id encoded into stream URL

All mocked at the desktop_client layer — no real camera, no real
Chrome, no real desktop-agent required.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


# ── shared fixtures ────────────────────────────────────────────────────


class _StubCtx:
    """Minimal AgentContext stub — matches the fields ToolCtx reads."""

    def __init__(self):
        self.client_id = "test-client"
        self.user_lang = "en"
        self.profile_id = 1
        self.is_authenticated = True
        self.progress_sink = None
        self.stream_sink = None
        self.media_sink = None


@pytest.fixture
def ctx():
    return _StubCtx()


@pytest.fixture
def media_calls():
    """Capture (url, source) tuples pushed via media_sink."""
    calls = []

    async def _sink(url: str, source: str) -> None:
        calls.append((url, source))

    return calls, _sink


def _reachable_patch():
    return patch("app.desktop_client.is_reachable", return_value=True)


def _unreachable_patch():
    return patch("app.desktop_client.is_reachable", return_value=False)


def _camera_cap_patch(value: bool):
    return patch(
        "app.desktop_client.has_capability_cached",
        return_value=value,
    )


# ── stream_camera ─────────────────────────────────────────────────────


async def test_stream_camera_happy_path(ctx, media_calls):
    """Happy path: camera available → media_sink called + spoken reply."""
    calls, sink = media_calls
    ctx.media_sink = sink

    with (
        _reachable_patch(),
        _camera_cap_patch(True),
    ):
        from app.tools.stream_camera import stream_camera
        result = await stream_camera(fps=10, ctx=ctx)

    assert result.data.get("source") == "camera"
    assert "/api/stream/camera" in result.data.get("stream_url", "")
    assert "fps=10" in result.data.get("stream_url", "")
    assert len(calls) == 1
    url, source = calls[0]
    assert source == "camera"
    assert "/api/stream/camera" in url
    assert "fps=10" in url
    assert "Camera" in result.text or "Трансляция" in result.text or "Stream" in result.text


async def test_stream_camera_no_media_sink(ctx):
    """media_sink is None — tool completes without error (no-op sink)."""
    # ctx.media_sink stays None
    with (
        _reachable_patch(),
        _camera_cap_patch(True),
    ):
        from app.tools.stream_camera import stream_camera
        result = await stream_camera(ctx=ctx)

    assert result.data.get("source") == "camera"


async def test_stream_camera_unavailable(ctx):
    """Camera capability absent → camera.unavailable message, no media_sink call."""
    calls, sink = [], None

    async def _sink(url, source):
        calls.append((url, source))

    ctx.media_sink = _sink

    with (
        _reachable_patch(),
        _camera_cap_patch(False),
    ):
        from app.tools.stream_camera import stream_camera
        result = await stream_camera(ctx=ctx)

    assert result.data.get("error") == "camera_unavailable"
    assert len(calls) == 0


async def test_stream_camera_desktop_offline(ctx):
    """Agent unreachable → desktop.legacy_unavailable before capability check."""
    with _unreachable_patch():
        from app.tools.stream_camera import stream_camera
        result = await stream_camera(ctx=ctx)

    assert result.data.get("error") == "desktop_unavailable"


async def test_stream_camera_fps_clamped(ctx, media_calls):
    """fps > 30 is clamped to 30; fps < 1 is clamped to 1."""
    calls, sink = media_calls
    ctx.media_sink = sink

    with (
        _reachable_patch(),
        _camera_cap_patch(True),
    ):
        from app.tools.stream_camera import stream_camera
        result_high = await stream_camera(fps=999, ctx=ctx)
        result_low = await stream_camera(fps=0, ctx=ctx)

    assert "fps=30" in result_high.data.get("stream_url", "")
    assert "fps=1" in result_low.data.get("stream_url", "")


# ── stream_tab ────────────────────────────────────────────────────────


async def test_stream_tab_happy_path(ctx, media_calls):
    """Happy path: CDP available → media_sink called with tab URL."""
    calls, sink = media_calls
    ctx.media_sink = sink

    with (
        _reachable_patch(),
        patch("app.desktop_client.has_capability_cached", return_value=True),
    ):
        from app.tools.stream_tab import stream_tab
        result = await stream_tab(fps=5, ctx=ctx)

    assert result.data.get("source") == "tab"
    assert "/api/stream/tab" in result.data.get("stream_url", "")
    assert len(calls) == 1
    url, source = calls[0]
    assert source == "tab"
    assert "/api/stream/tab" in url


async def test_stream_tab_with_tab_id(ctx, media_calls):
    """tab_id is included in the stream URL."""
    calls, sink = media_calls
    ctx.media_sink = sink

    with (
        _reachable_patch(),
        patch("app.desktop_client.has_capability_cached", return_value=True),
    ):
        from app.tools.stream_tab import stream_tab
        result = await stream_tab(tab_id="ABC123", fps=5, ctx=ctx)

    stream_url = result.data.get("stream_url", "")
    assert "tab_id=ABC123" in stream_url
    url, _ = calls[0]
    assert "tab_id=ABC123" in url


async def test_stream_tab_cdp_unavailable(ctx):
    """CDP not running → browser.cdp_unavailable message, no media_sink."""
    calls = []

    async def _sink(url, source):
        calls.append((url, source))

    ctx.media_sink = _sink

    with (
        _reachable_patch(),
        patch("app.desktop_client.has_capability_cached", return_value=False),
    ):
        from app.tools.stream_tab import stream_tab
        result = await stream_tab(ctx=ctx)

    assert result.data.get("error") == "cdp_unavailable"
    assert len(calls) == 0


async def test_stream_tab_desktop_offline(ctx):
    """Agent offline → desktop.legacy_unavailable."""
    with _unreachable_patch():
        from app.tools.stream_tab import stream_tab
        result = await stream_tab(ctx=ctx)

    assert result.data.get("error") == "desktop_unavailable"


async def test_stream_tab_fps_clamped(ctx, media_calls):
    """tab fps > 15 is clamped to 15."""
    calls, sink = media_calls
    ctx.media_sink = sink

    with (
        _reachable_patch(),
        patch("app.desktop_client.has_capability_cached", return_value=True),
    ):
        from app.tools.stream_tab import stream_tab
        result = await stream_tab(fps=100, ctx=ctx)

    assert "fps=15" in result.data.get("stream_url", "")
