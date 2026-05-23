#!/usr/bin/env bash
# Install a launchd agent so xtts-server starts at login.
#
# Renders com.voiceassistant.xtts.plist.template with this directory's
# absolute path baked in, copies it into ~/Library/LaunchAgents/, and
# tells launchd to load it.  After this, xtts-server starts at every
# login and auto-restarts on crash (but not on clean exit — so `kill`
# works as expected).
#
# Remove later via ./uninstall-autostart.sh.

set -e
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
START_SH="$SCRIPT_DIR/start.sh"
TEMPLATE="$SCRIPT_DIR/com.voiceassistant.xtts.plist.template"
TARGET="$HOME/Library/LaunchAgents/com.voiceassistant.xtts.plist"
LOG_DIR="$HOME/Library/Logs/voice-assistant"

if [ ! -x "$START_SH" ]; then
    echo "❌ start.sh not found or not executable at $START_SH"
    exit 1
fi

# Sanity-check that uv is on PATH — the plist bakes
# /opt/homebrew/bin and ~/.local/bin into the launchd env, so if uv
# isn't in either of those it'll fail to start without a clear error.
if ! command -v uv >/dev/null 2>&1; then
    echo "⚠ uv not found on PATH.  Install it first:"
    echo "    brew install uv      # or"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
UV_BIN="$(command -v uv)"
case "$UV_BIN" in
    /opt/homebrew/bin/*|$HOME/.local/bin/*|/usr/local/bin/*) ;;
    *)
        echo "⚠ uv found at $UV_BIN — but the plist looks for it in"
        echo "  /opt/homebrew/bin, ~/.local/bin, or /usr/local/bin."
        echo "  Edit PATH in com.voiceassistant.xtts.plist.template"
        echo "  if needed."
        ;;
esac

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$LOG_DIR"

# Render the template — replace the three placeholders.
sed \
    -e "s|__SCRIPT_PATH__|$START_SH|g" \
    -e "s|__SCRIPT_DIR__|$SCRIPT_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    "$TEMPLATE" > "$TARGET"

# If a previous agent is loaded, unload it first so the new plist takes effect.
launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"

echo "✅ Autostart installed."
echo
echo "   plist:    $TARGET"
echo "   uv:       $UV_BIN"
echo "   stdout:   $LOG_DIR/xtts.out.log"
echo "   stderr:   $LOG_DIR/xtts.err.log"
echo
echo "   Check:    launchctl list | grep xtts"
echo "   Logs:     tail -f $LOG_DIR/xtts.err.log"
echo "   Remove:   ./uninstall-autostart.sh"
