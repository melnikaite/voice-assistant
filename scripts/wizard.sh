#!/usr/bin/env bash
# scripts/wizard.sh — first-run setup wizard for voice-assistant
#
# Probes the host for all required backends, recommends a model if none
# is running, optionally writes .env, and offers to start the
# orchestrator.
#
# NEVER installs anything globally or touches existing venvs / PATH
# binaries.  If a backend is already running the script leaves it
# completely alone — most importantly the mlx-openai-server fork on
# :18000, which must not be replaced by an upstream version.
#
# Usage:
#   bash scripts/wizard.sh          # interactive
#   bash scripts/wizard.sh --check  # non-interactive probe (CI / health)

set -euo pipefail
cd "$(dirname "$0")/.."  # always run from repo root

# ── colour helpers ───────────────────────────────────────────────────

if [ -t 1 ]; then
  RED='\033[0;31m' YEL='\033[0;33m' GRN='\033[0;32m'
  CYN='\033[0;36m' BLD='\033[1m'    RST='\033[0m'
else
  RED='' YEL='' GRN='' CYN='' BLD='' RST=''
fi

ok()      { echo -e "${GRN}  ✓${RST} $*"; }
warn()    { echo -e "${YEL}  ⚠${RST} $*"; }
fail()    { echo -e "${RED}  ✗${RST} $*"; }
hint()    { echo -e "${CYN}    →${RST} $*"; }
section() { echo; echo -e "${BLD}── $* ──────────────────────────────────────────────────────${RST}"; }

# ── HTTP probe helpers ────────────────────────────────────────────────

# Returns 0 if the URL responds with any HTTP status, 1 if unreachable.
probe_reachable() { curl -sf --max-time 2 "$1" >/dev/null 2>&1; }

# Returns the raw JSON body or an empty string on any error.
probe_json() { curl -sf --max-time 3 "$1" 2>/dev/null || true; }

# ── Hardware detection ────────────────────────────────────────────────

ram_gb() {
  if [[ "$OSTYPE" == darwin* ]]; then
    local b; b=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    echo $(( b / 1073741824 ))
  else
    local k; k=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    echo $(( k / 1048576 ))
  fi
}

is_apple_silicon() { [[ "$OSTYPE" == darwin* && "$(uname -m)" == arm64 ]]; }

# ── LLM model recommendation ─────────────────────────────────────────
#
# Strategy (probe-and-adapt):
#   1. If llmfit is on PATH, try `llmfit fit --json` — use its pick.
#   2. Otherwise fall back to a RAM-based tier matrix drawn from the
#      llmfit fork's hf_models.json data (whisper data is for ASR not
#      LLM; we use the LLM side here).
#
# This function prints the model ID string and nothing else.

recommend_llm_model() {
  # 1. llmfit probe
  if command -v llmfit &>/dev/null; then
    local out
    out=$(llmfit fit --json 2>/dev/null || true)
    if [ -n "$out" ] && echo "$out" | grep -q '"model"'; then
      echo "$out" | python3 -c \
        "import sys,json; print(json.load(sys.stdin).get('model',''))" 2>/dev/null \
        && return
    fi
  fi

  # 2. RAM-tier fallback
  local gb; gb=$(ram_gb)
  if   (( gb >= 64 )); then echo "google/gemma-3-27b-it"
  elif (( gb >= 32 )); then echo "google/gemma-4-e4b"
  else                       echo "google/gemma-2-2b-it"
  fi
}

# ── Section: OS ───────────────────────────────────────────────────────

OS_OK=false

check_os() {
  section "System"
  case "$OSTYPE" in
    darwin*)
      ok "macOS $(sw_vers -productVersion 2>/dev/null)"
      OS_OK=true
      ;;
    linux*)
      ok "Linux $(uname -r | cut -d- -f1)"
      OS_OK=true
      ;;
    *)
      fail "Unsupported OS: $OSTYPE  (macOS and Linux only)"
      exit 1
      ;;
  esac
  local gb; gb=$(ram_gb)
  ok "RAM: ${gb} GB"
  is_apple_silicon && ok "Apple Silicon (MLX acceleration available)"
}

# ── Section: Docker ───────────────────────────────────────────────────

DOCKER_OK=false

check_docker() {
  section "Docker"
  if ! command -v docker &>/dev/null; then
    fail "docker not found"
    hint "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    return
  fi
  if ! docker info &>/dev/null 2>&1; then
    fail "Docker installed but not running — start Docker Desktop"
    return
  fi
  ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"

  # macOS: host networking requires Docker Desktop 4.34+
  if [[ "$OSTYPE" == darwin* ]]; then
    local ver
    ver=$(docker info --format '{{.ServerVersion}}' 2>/dev/null | grep -oE '^[0-9]+\.[0-9]+' || echo "0.0")
    local major; major=$(echo "$ver" | cut -d. -f1)
    if (( major < 26 )); then
      warn "Docker Desktop may be too old for host networking (need 4.34+)"
      hint "Settings → General → check version, update if needed"
    fi
    hint "Confirm: Settings → Resources → Network → 'Enable host networking' is ON"
  fi
  DOCKER_OK=true
}

# ── Section: ASR ─────────────────────────────────────────────────────
#
# NO-CLOBBER RULE: if anything is already listening on :18000 we leave
# it completely alone.  This preserves the mlx-openai-server fork (or
# any other Whisper-compatible server the user has running) without
# risk of an upstream-version overwrite.

ASR_URL="http://localhost:18000"
ASR_OK=false

check_asr() {
  section "ASR / Whisper  (:18000)"
  local body; body=$(probe_json "${ASR_URL}/v1/models")

  if [ -n "$body" ] && echo "$body" | grep -q '"object"'; then
    local ids
    ids=$(echo "$body" | python3 -c \
      "import sys,json
d=json.load(sys.stdin)
print(', '.join(m['id'] for m in d.get('data',[])))" 2>/dev/null || echo "(unknown)")
    ok "Running — model(s): ${ids}"
    ok "Preserved as-is (will not be overwritten)"
    ASR_OK=true
  else
    fail "No server on :18000"
    if is_apple_silicon; then
      hint "Start your mlx-openai-server fork:"
      hint "  source ~/.venvs/mlx-server/bin/activate"
      hint "  mlx-openai-server launch --config ~/.mlx-server/config.yaml"
    else
      hint "Start any OpenAI-Whisper-compatible server on :18000"
      hint "  faster-whisper-server, whisper.cpp, or vLLM with Whisper"
    fi
    hint "After starting it, re-run: bash scripts/wizard.sh"
  fi
}

# ── Section: LLM ─────────────────────────────────────────────────────

LLM_URL=""
LLM_MODEL=""
LLM_OK=false

_detect_first_model() {
  # Extract the first model ID from a /v1/models JSON response.
  python3 -c \
    "import sys,json
d=json.load(sys.stdin)
m=d.get('data',[])
print(m[0]['id'] if m else '')" 2>/dev/null || true
}

check_llm() {
  section "LLM backend"
  local body first_model

  # --- LM Studio (:1234) ---
  body=$(probe_json "http://localhost:1234/v1/models")
  if [ -n "$body" ] && echo "$body" | grep -q '"object"'; then
    first_model=$(echo "$body" | _detect_first_model)
    ok "LM Studio running on :1234"
    ok "Active model: ${first_model:-'(none loaded)'}"
    LLM_URL="http://localhost:1234"
    LLM_MODEL="${first_model:-$(recommend_llm_model)}"
    LLM_OK=true
    return
  fi

  # --- Ollama (:11434) ---
  body=$(probe_json "http://localhost:11434/v1/models")
  if [ -n "$body" ] && echo "$body" | grep -q '"object"'; then
    first_model=$(echo "$body" | _detect_first_model)
    ok "Ollama running on :11434"
    ok "Active model: ${first_model:-'(no model pulled)'}"
    LLM_URL="http://localhost:11434"
    LLM_MODEL="${first_model:-$(recommend_llm_model)}"
    LLM_OK=true
    return
  fi

  # --- Nothing found ---
  fail "No LLM backend found (tried :1234 LM Studio, :11434 Ollama)"
  local rec; rec=$(recommend_llm_model)
  hint "Recommended model for your hardware: ${rec}"
  hint "Option A — LM Studio: https://lmstudio.ai  →  load '${rec}'"
  hint "Option B — Ollama:    https://ollama.ai    →  ollama pull ${rec}"
  LLM_URL="http://localhost:1234"  # default; can edit docker-compose.yml
  LLM_MODEL="$rec"
  LLM_OK=false
}

# ── Section: TTS ─────────────────────────────────────────────────────

TTS_OK=false

check_tts() {
  section "TTS — XTTS-v2  (:9876)"
  if probe_reachable "http://localhost:9876/health"; then
    ok "xtts-server running"
    TTS_OK=true
  else
    warn "xtts-server not running (voice replies will fail)"
    hint "Start it:  cd xtts-server && ./start.sh"
    hint "(first run downloads the XTTS-v2 model — ~1.8 GB, cached afterwards)"
    TTS_OK=false
  fi
}

# ── Section: Desktop-agent ────────────────────────────────────────────

AGENT_OK=false

check_agent() {
  section "Desktop-agent  (:9877, optional)"
  if probe_reachable "http://localhost:9877/v1/capabilities"; then
    ok "Desktop-agent running"
    AGENT_OK=true
  else
    hint "Not running — computer_use / mail / calendar tools will be unavailable"
    hint "Start it:  cd desktop-agent && ./start.sh"
    AGENT_OK=false
  fi
}

# ── .env setup ────────────────────────────────────────────────────────
#
# Only creates .env if it doesn't exist yet.  Never overwrites an
# existing one — changes must stay explicit.

setup_env() {
  section ".env"
  if [ -f .env ]; then
    ok ".env already exists — leaving unchanged"
    return
  fi

  cp .env.example .env
  ok "Created .env from .env.example"

  # If Ollama was detected instead of the default LM Studio, append
  # overrides so docker-compose.yml picks up the right endpoint.
  if [ "$LLM_OK" = true ] && [ "$LLM_URL" = "http://localhost:11434" ]; then
    {
      echo ""
      echo "# Auto-configured by wizard (Ollama detected on :11434)"
      echo "LLM_URL=${LLM_URL}"
      echo "LLM_MODEL=${LLM_MODEL}"
    } >> .env
    ok "Added LLM_URL / LLM_MODEL for Ollama"
  fi
}

# ── Summary ───────────────────────────────────────────────────────────

print_summary() {
  section "Summary"
  echo
  local _ok="${GRN}✓ ready${RST}"
  local _warn="${YEL}⚠ needs attention${RST}"
  local _opt="${CYN}→ optional${RST}"

  printf "  %-20s %b\n" "ASR (:18000)"   "$([ "$ASR_OK"    = true ] && echo "$_ok" || echo "$_warn")"
  printf "  %-20s %b\n" "LLM (${LLM_URL})" "$([ "$LLM_OK" = true ] && echo "$_ok" || echo "$_warn")"
  printf "  %-20s %b\n" "TTS (:9876)"    "$([ "$TTS_OK"    = true ] && echo "$_ok" || echo "$_warn")"
  printf "  %-20s %b\n" "Desktop-agent"  "$([ "$AGENT_OK"  = true ] && echo "$_ok" || echo "$_opt")"
  printf "  %-20s %b\n" "Docker"         "$([ "$DOCKER_OK" = true ] && echo "$_ok" || echo "$_warn")"
  echo
}

# ── Optional start ────────────────────────────────────────────────────

maybe_start() {
  if [ "${CHECK_ONLY:-false}" = true ]; then
    return
  fi

  if [ "$DOCKER_OK" != true ] || [ "$ASR_OK" != true ] || [ "$LLM_OK" != true ]; then
    warn "Fix the items above, then re-run:  bash scripts/wizard.sh"
    hint "Or start manually:                 docker compose up -d"
    return
  fi

  echo
  printf "Start the orchestrator now? [Y/n] "
  read -r ans </dev/tty
  if [[ -z "$ans" || "${ans,,}" =~ ^y ]]; then
    echo
    docker compose up -d
    echo
    ok "Orchestrator started →  http://localhost:${HOST_PORT:-8080}"
    hint "Tail logs:  task logs   (or: docker compose logs -f orchestrator)"
    hint "Stop:       task down   (or: docker compose down)"
  fi
}

# ── main ─────────────────────────────────────────────────────────────

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

main() {
  echo
  echo -e "${BLD}voice-assistant  —  setup wizard${RST}"
  echo "────────────────────────────────────────────────────────────────"
  echo "  Probes host backends, writes .env if missing, optionally"
  echo "  starts the orchestrator.  Nothing is installed globally."
  echo "  Your mlx-openai-server fork on :18000 is never overwritten."
  echo

  check_os
  check_docker
  check_asr
  check_llm
  check_tts
  check_agent
  setup_env
  print_summary
  maybe_start
}

main "$@"
