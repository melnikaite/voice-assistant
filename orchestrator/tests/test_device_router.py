"""
Tests for personal-device routing (#50) + WoL hardening (security pass).

Scope:
  * resolve_agent_id_for_profile — no pairing / unknown agent / online / stale+WoL
  * send_wol — MAC validation, public-IP rejection (egress hardening), magic packet
All offline: storage + desktop_client + the UDP send are mocked.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── resolve_agent_id_for_profile ─────────────────────────────────────


async def test_resolve_no_pairing_returns_none():
    """No profile_devices row → None (caller falls back to 'no device')."""
    with patch("app.device_router.get_default_device", AsyncMock(return_value=None)):
        from app.device_router import resolve_agent_id_for_profile
        out = await resolve_agent_id_for_profile(1, "macos_agent")
    assert out is None


async def test_resolve_agent_not_in_registry_returns_id_anyway():
    """Paired but agent never connected → still return the id (transport
    layer surfaces the clear 'unavailable' error)."""
    row = {"device_uid": "macbook", "wol_mac": None, "wol_target_ip": None}
    with (
        patch("app.device_router.get_default_device", AsyncMock(return_value=row)),
        patch("app.desktop_client.get_agent", MagicMock(return_value=None)),
    ):
        from app.device_router import resolve_agent_id_for_profile
        out = await resolve_agent_id_for_profile(1, "macos_agent")
    assert out == "macbook"


async def test_resolve_online_agent_no_wol():
    """Recently-seen agent → return id, no WoL packet."""
    row = {"device_uid": "macbook", "wol_mac": "AA:BB:CC:DD:EE:FF", "wol_target_ip": None}
    fresh = MagicMock()
    fresh.last_seen = time.time()  # just seen → not stale
    wol = AsyncMock()
    with (
        patch("app.device_router.get_default_device", AsyncMock(return_value=row)),
        patch("app.desktop_client.get_agent", MagicMock(return_value=fresh)),
        patch("app.device_router._wol_and_wait", wol),
    ):
        from app.device_router import resolve_agent_id_for_profile
        out = await resolve_agent_id_for_profile(1, "macos_agent")
        await asyncio.sleep(0)
    assert out == "macbook"
    wol.assert_not_called()


async def test_resolve_stale_agent_with_wol_fires_packet():
    """Stale agent + wol_mac configured → schedule a WoL wake."""
    row = {"device_uid": "macbook", "wol_mac": "AA:BB:CC:DD:EE:FF", "wol_target_ip": "192.168.1.255"}
    stale = MagicMock()
    stale.last_seen = 0  # never heartbeated → stale
    wol = AsyncMock()
    with (
        patch("app.device_router.get_default_device", AsyncMock(return_value=row)),
        patch("app.desktop_client.get_agent", MagicMock(return_value=stale)),
        patch("app.device_router._wol_and_wait", wol),
    ):
        from app.device_router import resolve_agent_id_for_profile
        out = await resolve_agent_id_for_profile(1, "macos_agent")
        await asyncio.sleep(0)  # let the created task evaluate the arg
    assert out == "macbook"
    wol.assert_called_once_with("AA:BB:CC:DD:EE:FF", "192.168.1.255", "macbook")


# ── send_wol ─────────────────────────────────────────────────────────


async def test_send_wol_rejects_bad_mac():
    """A malformed MAC raises ValueError (no packet sent)."""
    from app.device_router import send_wol
    with pytest.raises(ValueError):
        await send_wol("not-a-mac", "192.168.1.255")


async def test_send_wol_rejects_public_ip():
    """A public/routable target IP is refused — WoL is a LAN-only egress."""
    from app.device_router import send_wol
    with pytest.raises(ValueError):
        await send_wol("AA:BB:CC:DD:EE:FF", "8.8.8.8")


async def test_send_wol_valid_private_sends_magic_packet():
    """Valid MAC + private broadcast → the 6×0xFF + 16×MAC magic packet."""
    sent = MagicMock()
    with patch("app.device_router._send_udp", sent):
        from app.device_router import send_wol
        await send_wol("AA:BB:CC:DD:EE:FF", "192.168.1.255")
    sent.assert_called_once()
    packet, ip, port = sent.call_args[0]
    assert ip == "192.168.1.255"
    assert port == 9
    assert packet[:6] == b"\xff" * 6
    assert packet == b"\xff" * 6 + bytes.fromhex("AABBCCDDEEFF") * 16
