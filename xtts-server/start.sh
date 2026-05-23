#!/usr/bin/env bash
# Launch the XTTS-v2 host service via uv.
#
# uv = single-binary Python project manager from astral.sh.  It brings
# its OWN Python toolchain (independent of Homebrew / system Python),
# so we get a fully isolated venv without coupling to whatever Python
# happens to be on PATH today.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    cat <<'EOF' >&2

uv is not installed.  It's a Python environment manager that does NOT
depend on Homebrew's Python — so a system Python upgrade
(`brew upgrade python`) won't break xtts-server.

Pick either installer:

  curl -LsSf https://astral.sh/uv/install.sh | sh
  brew install uv

Then run ./start.sh again.

EOF
    exit 1
fi

# `uv sync` is idempotent:
#   • first run: installs Python 3.12 into ~/.local/share/uv (NOT touching
#     Homebrew / system Python), creates ./.venv, installs all deps from
#     pyproject.toml (~500 MB into the venv), writes uv.lock for
#     reproducible reinstalls.
#   • subsequent runs: ~1 s, just verifies the lockfile.
echo "→ uv sync — verifying environment …"
uv sync

# Run the server inside the project's managed venv.  exec replaces the
# shell process so Ctrl-C / signals go straight to uvicorn.
exec uv run xtts-server.py
