#!/usr/bin/env bash
# Remove the launchd agent installed by install-autostart.sh.
#
# Stops xtts-server (if running under launchd), unloads the plist,
# deletes the plist file.  Leaves the venv, model, and uv alone —
# use ./uninstall.sh for those.

set -e

TARGET="$HOME/Library/LaunchAgents/com.voiceassistant.xtts.plist"

if [ ! -f "$TARGET" ]; then
    echo "Autostart not installed (no $TARGET)"
    exit 0
fi

launchctl unload "$TARGET" 2>/dev/null || true
rm -f "$TARGET"

echo "✅ Autostart removed."
echo
echo "   xtts-server stopped (if it was running under launchd)."
echo "   venv, model, and uv left intact — see uninstall.sh to remove those."
