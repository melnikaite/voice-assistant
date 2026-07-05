#!/usr/bin/env bash
# Launches the orchestrator natively in a self-contained uv venv —
# same pattern as ../xtts-server and ../desktop-agent.  Idempotent:
# uv only re-resolves when pyproject.toml changed.
#
# Usage:
#     ./start.sh              # foreground (Ctrl+C to stop)
#     launchd                 # via com.voiceassistant.orchestrator
#
# The docker-compose path (../docker-compose.yml) remains as the
# optional Linux-server variant; this script is the macOS default.

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed.  Bootstrap with:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  or"
  echo "    brew install uv"
  exit 1
fi

# Force uv's OWN CPython toolchain, never the system/brew one — brew
# python paths change on any passing `brew upgrade` (see desktop-agent).
export UV_PYTHON_PREFERENCE=only-managed

# Operator overrides come from the repo-root .env (same file the
# docker-compose path reads); everything unset falls back to the
# defaults below, which mirror docker-compose.yml's documented values.
set -a
[ -f "$ROOT/.env" ] && source "$ROOT/.env"

# --- backends -----------------------------------------------------------
: "${LLM_URL:=http://localhost:1240}"
: "${LLM_MODEL:=gemma-4-e4b-it-qat-q4_0}"
# Disables Gemma's adaptive thinking on LocalAI/llama.cpp — a voice
# reply cannot afford sporadic multi-second CoT (see docker-compose.yml).
: "${LLM_REASONING_EFFORT:=none}"
: "${LLM_MAX_TOKENS:=131072}"
: "${LLM_TIMEOUT:=120}"
: "${WHISPER_URL:=http://localhost:1240}"
: "${WHISPER_MODEL:=whisper-large-q5_0}"
: "${WHISPER_TEMPERATURE:=0.0}"
: "${TTS_URL:=http://localhost:9876}"
: "${DESKTOP_URL:=http://localhost:9877}"
: "${DESKTOP_HEALTHPOLL_INTERVAL_S:=30}"

# --- storage / static ---------------------------------------------------
: "${DB_PATH:=$ROOT/data/assistant.db}"
: "${DATA_DIR_CONTAINER:=$ROOT/data}"
: "${DATA_DIR_HOST:=$ROOT/data}"
: "${STATIC_DIR:=$ROOT/frontend}"
: "${FASTEMBED_CACHE_PATH:=$HOME/.cache/voice-assistant/fastembed}"

# --- behaviour knobs (compose-documented values) -------------------------
: "${EMBEDDING_MODEL:=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
: "${MEMORY_TOP_K:=3}"
: "${MEMORY_SIMILARITY_THRESHOLD:=0.72}"
: "${MEMORY_MAX_AGE_DAYS:=30}"
: "${MAX_HISTORY_TURNS:=10}"
: "${HISTORY_RESUME_MAX_AGE_S:=1800}"
: "${CONTINUATION_TIMEOUT_S:=10}"
: "${SPEAKER_THRESHOLD:=0.75}"
: "${VAD_SILENCE_TIMEOUT_S:=1.8}"
: "${VAD_MIN_SPEECH_S:=0.3}"
: "${VAD_MAX_RECORD_S:=30}"
: "${WAKE_WORD_NAME:=hey_jarvis_v0.1}"
: "${WAKE_WORD_THRESHOLD:=0.5}"
: "${DDG_REGION:=de-de}"
: "${NEWS_DEFAULT_TOPICS:=tech, world news, AI}"
: "${NEWS_MAX_RESULTS:=6}"
: "${WEB_SEARCH_MAX_RESULTS:=8}"
: "${WEB_SEARCH_FETCH_PAGES:=true}"
: "${WEB_SEARCH_FETCH_TARGET:=5}"
: "${WEB_SEARCH_FETCH_PARALLELISM:=10}"
: "${WEB_SEARCH_FETCH_DEADLINE_S:=4}"
: "${WEB_SEARCH_PAGE_TIMEOUT_S:=8}"
: "${WEB_SEARCH_PAGE_MAX_CHARS:=3000}"
: "${TZ:=Europe/Berlin}"

# --- bind ----------------------------------------------------------------
# 0.0.0.0 so family devices on the LAN / tailnet reach the PWA, same as
# the container did.  The optional outer Basic Auth (#43) is the door.
: "${ORCH_HOST:=0.0.0.0}"
: "${ORCH_PORT:=8080}"
set +a

mkdir -p "$DATA_DIR_HOST" "$FASTEMBED_CACHE_PATH"

echo "→ ensuring venv…"
uv sync --quiet

echo "→ starting orchestrator on ${ORCH_HOST}:${ORCH_PORT}"
# --reload keeps the edit→takes-effect dev loop the container had; the
# watcher is scoped to app/ so data/ writes never trigger restarts.
exec uv run uvicorn app.main:app \
  --host "$ORCH_HOST" --port "$ORCH_PORT" \
  --reload --reload-dir app
