#!/usr/bin/env bash
# Remove the launchd agent installed by install-autostart.sh.
#
# Stops desktop-agent (if running under launchd), unloads the plist,
# deletes the plist file.  Leaves the venv and uv alone — use
# ./uninstall.sh for those.

set -e

TARGET="$HOME/Library/LaunchAgents/com.voiceassistant.desktop.plist"

if [ ! -f "$TARGET" ]; then
    echo "Autostart not installed (no $TARGET)"
    exit 0
fi

launchctl unload "$TARGET" 2>/dev/null || true
rm -f "$TARGET"

echo "✅ Autostart removed."
echo
echo "   desktop-agent stopped (if it was running under launchd)."
echo "   venv and uv left intact — see uninstall.sh to remove those."
