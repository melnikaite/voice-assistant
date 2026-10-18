"""Operator CLI — host-side maintenance that must not be reachable over HTTP.

    uv run python -m app.cli reset-passphrase <profile_id|name>

Defaults DB_PATH / DATA_DIR_CONTAINER to the repo's ``data/`` (as
``start.sh`` does) so it hits the same files as the launchd service.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data"
os.environ.setdefault("DB_PATH", str(_DATA / "assistant.db"))
os.environ.setdefault("DATA_DIR_CONTAINER", str(_DATA))


async def _resolve_profile(ref: str) -> tuple[int, str] | None:
    from .storage import get_all_speaker_profiles

    profiles = [(p[0], p[1]) for p in await get_all_speaker_profiles()]
    if ref.isdigit():
        return next((p for p in profiles if p[0] == int(ref)), None)
    matches = [p for p in profiles if p[1].casefold() == ref.casefold()]
    if len(matches) > 1:
        raise SystemExit(f"name {ref!r} is ambiguous — pass the profile id: {[m[0] for m in matches]}")
    return matches[0] if matches else None


async def reset_passphrase(ref: str) -> int:
    from .storage import revoke_profile_sessions
    from .user_files import clear_passphrase

    profile = await _resolve_profile(ref)
    if profile is None:
        print(f"no profile matches {ref!r}", file=sys.stderr)
        return 1
    pid, name = profile
    cleared = await clear_passphrase(pid)
    revoked = await revoke_profile_sessions(pid)
    print(
        f"profile {pid} ({name}): code word {'cleared' if cleared else 'was not set'}, "
        f"{revoked} session(s) revoked — set a new one from the web UI login."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("reset-passphrase", help="clear a profile's code word and log it out everywhere")
    rp.add_argument("profile", help="profile id or name")
    args = parser.parse_args(argv)
    if args.cmd == "reset-passphrase":
        return asyncio.run(reset_passphrase(args.profile))
    return 2


if __name__ == "__main__":
    sys.exit(main())
