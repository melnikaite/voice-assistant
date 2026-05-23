#!/usr/bin/env bash
# Wipe xtts-server from the host.
#
# This intentionally does NOT touch the model files in
# ~/.cache/voice-assistant/xtts/ (1.8 GB) — re-downloading those would
# burn time and bandwidth.  Delete them manually if you really want to.
# Same for uv itself: only this project's venv is removed; uv stays
# installed unless you nuke it explicitly.

set -e
cd "$(dirname "$0")"

cat <<EOF
Removing from this folder:
  .venv/        — isolated xtts-server environment
  uv.lock       — dependency lockfile

LEFT INTACT (delete manually if you want to reclaim disk):
  ~/.cache/voice-assistant/xtts/   — XTTS-v2 models (~1.8 GB)
       rm -rf ~/.cache/voice-assistant/xtts

  uv (the manager itself) and its data:
       rm -rf ~/.local/bin/uv ~/.local/share/uv ~/.cache/uv

  Autostart (if you installed it — see README on launchd):
       launchctl unload ~/Library/LaunchAgents/com.voiceassistant.xtts.plist
       rm ~/Library/LaunchAgents/com.voiceassistant.xtts.plist

EOF

read -p "Continue? [y/N] " yn
case "$yn" in
    y|Y) ;;
    *) echo "cancelled"; exit 0 ;;
esac

rm -rf .venv uv.lock
echo "✅ xtts-server removed from this folder (model and uv left intact — see notes above)"
