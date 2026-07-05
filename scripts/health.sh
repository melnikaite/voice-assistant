#!/usr/bin/env bash
# scripts/health.sh — quick backend status overview
#
# Called by `task health`.  Probes all five backend endpoints and
# prints a one-line status for each.  Exits 0 always (so `task health`
# doesn't fail on partial-down state); pass --strict to exit 1 on any
# degraded backend (useful in CI).

set -uo pipefail
cd "$(dirname "$0")/.."

STRICT=false; [[ "${1:-}" == "--strict" ]] && STRICT=true

if [ -t 1 ]; then
  GRN='\033[0;32m' YEL='\033[0;33m' RED='\033[0;31m' BLD='\033[1m' RST='\033[0m'
else
  GRN='' YEL='' RED='' BLD='' RST=''
fi

FAILURES=0

probe() {
  # probe <label> <url> [<grep-pattern-for-"ok">]
  local label="$1" url="$2" pat="${3:-}"
  local body
  body=$(curl -sf --max-time 3 "$url" 2>/dev/null || true)

  if [ -z "$body" ]; then
    printf "  ${RED}✗${RST}  %-26s unreachable\n" "$label"
    (( FAILURES++ )) || true
    return
  fi

  if [ -n "$pat" ] && ! echo "$body" | grep -q "$pat"; then
    printf "  ${YEL}⚠${RST}  %-26s unexpected response\n" "$label"
    (( FAILURES++ )) || true
    return
  fi

  # Extract a short status hint from JSON, but use bare string matching
  # to avoid SystemExit-caught-by-bare-except issues.
  local hint
  hint=$(echo "$body" | python3 -c \
    "import sys, json
try:
    d = json.load(sys.stdin)
    # Orchestrator /health
    if 'status' in d:
        print(d['status'])
        sys.exit(0)
    # /v1/models
    ids = [m['id'] for m in d.get('data', [])]
    if ids:
        print(', '.join(ids[:2]))
        sys.exit(0)
    print('ok')
except Exception:
    print('ok')" 2>/dev/null || echo "ok")

  printf "  ${GRN}✓${RST}  %-26s %s\n" "$label" "$hint"
}

echo
echo -e "${BLD}── voice-assistant health ──────────────────────────────────────${RST}"
echo

probe "Orchestrator (:${HOST_PORT:-8080})"  "http://localhost:${HOST_PORT:-8080}/health"  '"status"'
probe "LocalAI LLM+ASR (:${BACKEND_PORT:-1240})" "http://localhost:${BACKEND_PORT:-1240}/v1/models" '"object"'
probe "TTS / xtts-server (:9876)"           "http://localhost:9876/v1/health"              ''
# The agent's API is token-gated — any HTTP status (401 included) proves
# the process is up, so probe by status code instead of body.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:9877/v1/capabilities" 2>/dev/null || echo 000)
if [ "$code" != "000" ]; then
  printf "  ${GRN}✓${RST}  %-26s ok (http %s)\n" "Desktop-agent (:9877)" "$code"
else
  printf "  ${RED}✗${RST}  %-26s unreachable\n" "Desktop-agent (:9877)"
  (( FAILURES++ )) || true
fi

echo

if [ "$STRICT" = true ] && [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
