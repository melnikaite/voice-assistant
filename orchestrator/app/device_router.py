"""
Personal-device routing (#50).

Routes device-tier tool calls to the correct desktop-agent based on the
recognised speaker's ``profile_id``.

Flow
────
1.  ``resolve_agent_id_for_profile(profile_id, device_kind)``
    – Look up ``profile_devices`` for the profile.
    – Filter to entries matching ``device_kind`` (or any kind if None).
    – Prefer the ``is_default`` row; fall back to the most-recently-seen.
    – If the agent hasn't heartbeated within ONLINE_TIMEOUT_S, try WoL
      before returning (fire-and-forget; caller doesn't wait for wake).
    – Return ``None`` when no pairing row exists or no agent in registry.

2.  ``send_wol(wol_mac, target_ip)``
    – Emit a magic packet to the given broadcast / unicast address on
      UDP port 9 (the standard WoL port).

Online detection
────────────────
``AgentInfo.last_seen`` is updated by:
  • The background health-poll task (HTTP mode, every 30 s by default).
  • ``heartbeat`` frames from the agent (reverse-WSS mode).
  • Any successful tool call (desktop_client updates last_seen on each
    successful HTTP response).

An agent is considered "online" when ``last_seen > now - ONLINE_TIMEOUT_S``.
``reachable`` is a direct flag; we use ``last_seen`` for WoL decisions
because ``reachable`` only flips on the next poll iteration.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import struct
import time

from . import desktop_client
from .storage.profile_devices import get_default_device

log = logging.getLogger(__name__)

# An agent is considered online if it heartbeated or polled within this window.
ONLINE_TIMEOUT_S = 90.0

# After sending a WoL magic packet, wait this long before giving up and
# returning the agent_id anyway (the tool call will either succeed or fail
# with a normal DesktopUnavailable — the router's job is routing, not retry).
_WOL_WAIT_S = 12.0


async def resolve_agent_id_for_profile(
    profile_id: int,
    device_kind: str | None,
) -> str | None:
    """Return the agent_id to use for a device-tier tool call by this profile.

    Returns ``None`` when:
      • No ``profile_devices`` row exists for this profile + device_kind.
      • The row exists but the agent_id is not in the desktop_client registry
        at all (agent was never configured or never connected).

    When the agent is offline (stale last_seen) AND WoL is configured, sends
    a magic packet and returns the agent_id immediately — the tool call will
    either succeed (if wake was fast enough) or fail with a transport error.
    The caller is responsible for the failure path; the router only routes.
    """
    device_row = await get_default_device(profile_id, device_kind)
    if device_row is None:
        log.debug(
            "device_router: profile=%d device_kind=%s — no pairing found",
            profile_id, device_kind or "any",
        )
        return None

    agent_id: str = device_row["device_uid"]
    agent_info = desktop_client.get_agent(agent_id)
    if agent_info is None:
        # Paired but the agent has never connected to this orchestrator
        # (e.g. the machine is off and has no WoL config).  Return the
        # agent_id anyway — desktop_client will raise DesktopUnavailable
        # with a clear error message.
        log.debug(
            "device_router: profile=%d agent=%s not in registry — returning anyway",
            profile_id, agent_id,
        )
        return agent_id

    # WoL: if the agent hasn't been seen recently and we have WoL config,
    # kick the magic packet.  We return the agent_id immediately and let
    # the tool call time out or succeed on its own.
    now = time.time()
    last_seen = agent_info.last_seen
    is_stale = (last_seen == 0) or ((now - last_seen) > ONLINE_TIMEOUT_S)

    if is_stale:
        wol_mac = device_row.get("wol_mac")
        wol_ip = device_row.get("wol_target_ip") or "255.255.255.255"
        if wol_mac:
            log.info(
                "device_router: profile=%d agent=%s stale (last_seen=%.0f s ago), "
                "sending WoL to %s",
                profile_id, agent_id, (now - last_seen) if last_seen else float("inf"),
                wol_mac,
            )
            asyncio.create_task(_wol_and_wait(wol_mac, wol_ip, agent_id))
        else:
            log.debug(
                "device_router: profile=%d agent=%s offline, no WoL configured",
                profile_id, agent_id,
            )

    return agent_id


async def _wol_and_wait(wol_mac: str, target_ip: str, agent_id: str) -> None:
    """Send a WoL packet, then poll until the agent comes online."""
    try:
        await send_wol(wol_mac, target_ip)
    except Exception as exc:
        log.warning("device_router: WoL send failed for %s: %s", agent_id, exc)
        return
    # Poll for the agent to come back online.
    deadline = time.monotonic() + _WOL_WAIT_S
    while time.monotonic() < deadline:
        await asyncio.sleep(2.0)
        info = desktop_client.get_agent(agent_id)
        if info and info.reachable:
            log.info("device_router: WoL success — %s is online", agent_id)
            return
    log.info(
        "device_router: WoL timeout — %s didn't respond within %.0f s",
        agent_id, _WOL_WAIT_S,
    )


async def send_wol(wol_mac: str, target_ip: str, port: int = 9) -> None:
    """Send an IEEE 802.3 Wake-on-LAN magic packet.

    ``wol_mac`` — colon- or hyphen-separated hex MAC, e.g.
                  ``"AA:BB:CC:DD:EE:FF"`` or ``"AA-BB-CC-DD-EE-FF"``.
    ``target_ip`` — broadcast address (``"192.168.1.255"``) or
                    unicast IP of the host.  Broadcast gives the best
                    chance of waking a machine on the same subnet; unicast
                    works for routed WoL (requires directed-broadcast or
                    relay agent on the router).
    """
    # Normalise: strip separators and convert to 6-byte binary.
    mac_clean = wol_mac.replace(":", "").replace("-", "").replace(".", "")
    if len(mac_clean) != 12:
        raise ValueError(f"Invalid MAC address: {wol_mac!r}")
    mac_bytes = bytes.fromhex(mac_clean)

    # Reject public/routable targets.  WoL is a LAN operation — the
    # target must be a private, loopback, link-local, or broadcast
    # address.  Without this, a stored ``wol_target_ip`` becomes an
    # outbound-UDP-to-arbitrary-host primitive (weak SSRF / LAN probe).
    try:
        ip = ipaddress.ip_address(target_ip)
    except ValueError:
        raise ValueError(f"Invalid WoL target IP: {target_ip!r}")
    if ip.is_global:
        raise ValueError(
            f"WoL target must be a LAN/broadcast address, not public: {target_ip!r}"
        )

    # Magic packet = 6× 0xFF + 16× MAC.
    magic = b"\xff" * 6 + mac_bytes * 16

    log.info("device_router: WoL → %s:%d (%s)", target_ip, port, wol_mac)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send_udp, magic, target_ip, port)


def _send_udp(packet: bytes, target_ip: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.connect((target_ip, port))
        s.send(packet)
