"""
``net.has_internet`` cache + offline gate.

A broken cache would flood every tool call with a probe (unbearable
latency on a flaky network) or, worse, cache a stale negative
forever after one bad probe.  We assert both directions.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app import net


@pytest.fixture(autouse=True)
def _reset_cache():
    """Wipe the module-level cache between tests."""
    net.invalidate_cache()
    yield
    net.invalidate_cache()


async def test_positive_cache_hits_once():
    """5 concurrent calls → exactly one probe."""
    mock_probe = AsyncMock(return_value=True)
    with patch.object(net, "_probe", mock_probe):
        results = await asyncio.gather(*[net.has_internet() for _ in range(5)])
    assert results == [True] * 5
    assert mock_probe.call_count == 1


async def test_negative_cache_hits_once():
    """Same for negative result — no probe stampede."""
    mock_probe = AsyncMock(return_value=False)
    with patch.object(net, "_probe", mock_probe):
        results = await asyncio.gather(*[net.has_internet() for _ in range(5)])
    assert results == [False] * 5
    assert mock_probe.call_count == 1


async def test_invalidate_forces_reprobe():
    """invalidate_cache() makes the next call hit the network again."""
    mock_probe = AsyncMock(return_value=True)
    with patch.object(net, "_probe", mock_probe):
        await net.has_internet()
        net.invalidate_cache()
        await net.has_internet()
    assert mock_probe.call_count == 2


async def test_probe_failure_returns_false():
    """The real _probe must swallow errors and return False — never raise.

    conftest.py points OFFLINE_PROBE_URL at 127.0.0.1:0 (closed port),
    so the probe ALWAYS fails in tests.  We verify the failure mode
    silently returns False — has_internet() can then cache that as
    "offline" without surfacing the connection error.
    """
    net.invalidate_cache()
    result = await net._probe()
    assert result is False
    # And has_internet itself stays False without raising.
    assert (await net.has_internet()) is False
