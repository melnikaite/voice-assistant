#!/usr/bin/env bash
# Launches desktop-agent in a self-contained uv venv.  Idempotent —
# uv only re-resolves if pyproject.toml changed.
#
# Usage:
#     ./start.sh              # foreground (Ctrl+C to stop)
#     ./start.sh &            # background — see uninstall.sh
#
# Bootstrap (one-time on the host):
#     curl -LsSf https://astral.sh/uv/install.sh | sh    # or brew install uv

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed.  Bootstrap with:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  or"
  echo "    brew install uv"
  exit 1
fi

# uv sync brings the .venv into existence (Python 3.12 from uv's own
# toolchain, fastapi + uvicorn + pyautogui + Pillow + platform-specific
# extras).  On a fresh install this is the only step that hits the network.
echo "→ ensuring venv…"
uv sync --quiet

echo "→ starting desktop-agent on ${DESKTOP_HOST:-127.0.0.1}:${DESKTOP_PORT:-9877}"
exec uv run python3 desktop-agent.py
