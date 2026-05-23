# desktop-agent

Host-side HTTP gateway for desktop automation — the orchestrator can't
reach the host's accessibility tree from inside Docker, so this daemon
runs natively in the user's GUI session and exposes a small localhost
API over HTTP.

The daemon is OS-agnostic by design: per-platform work lives in a
`Backend` subclass (`MacOSBackend`, `WindowsBackend`, `LinuxBackend`),
picked at startup.  The orchestrator stays platform-blind — it asks
`/v1/capabilities` what's available and registers the matching tools.

## Why

- macOS AppleScript / Apple Events needs the GUI session (and
  Accessibility + Automation permissions granted to the host process).
- pyautogui / pywinauto / xdotool all need access to the display,
  which a container does not have.
- Querying the OS default-app registry (NSWorkspace / xdg-mime /
  winreg) requires a logged-in user session.
- We want one universal tool surface in the orchestrator regardless of
  the host OS; this daemon abstracts per-OS engine choice.

## Endpoints (all under `/v1/`)

| Method | Path | Notes |
|--------|------|-------|
| `GET`  | `/v1/health` | Liveness + agent_id, platform, capabilities, version. **No auth.** |
| `GET`  | `/v1/capabilities` | Agent id + platform + feature flags + version.  Auth required. |
| `GET`  | `/v1/platform` | OS / Python / preferred engine (legacy; prefer `/v1/capabilities`). |
| `POST` | `/v1/applescript` | Run AppleScript via `/usr/bin/osascript`. macOS only. |
| `POST` | `/v1/pyautogui` | Structured cross-platform input (click/type/scroll/hotkey). |
| `POST` | `/v1/key` | Keyboard shortcut, e.g. `["cmd","space"]`. |
| `GET`  | `/v1/screenshot` | Full-screen PNG capture. |
| `GET`  | `/v1/default_app?category=mail\|browser\|calendar\|files` | Bundle id + display name of the OS default app for the category. macOS via JXA/NSWorkspace; stubs on Linux/Windows. |
| `GET`  | `/v1/cursor_activity` | Cursor position + idle seconds — used by the orchestrator to refuse vision-driven UI ops while the user is actively typing. |
| `POST` | `/v1/audit` | Free-form audit log entry (intent-only events). |

All authed endpoints require header `X-Desktop-Token: <secret>`. The
secret is read from env `DESKTOP_TOKEN`; if absent, the daemon
generates one and stashes it in `~/.cache/voice-assistant/desktop-token`
on first run. Copy that value into `docker-compose.yml` next to
`TTS_URL` / `WHISPER_URL`.

Every accepted execution is appended as one JSON-line to
`~/.cache/voice-assistant/desktop-audit.log` (filename kept stable
across the `desktop-server` → `desktop-agent` rename so existing
history isn't orphaned). Reviewable any time.

## Capabilities advertised by `/v1/capabilities`

| Feature | macOS | Linux | Windows |
|---------|-------|-------|---------|
| `screenshot`            | pyautogui | pyautogui (X11)      | pyautogui |
| `applescript`           | yes       | no                   | no        |
| `pyautogui`             | yes       | yes (X11)            | yes       |
| `hotkey`                | pyautogui | pyautogui or xdotool | pyautogui |
| `default_apps_resolver` | JXA / NSWorkspace | no (xdg-mime TBD) | no (winreg TBD) |
| `cursor_activity`       | pyautogui | pyautogui            | pyautogui |

Per-app mail/browser/calendar logic does NOT live here.  Wave 3
collapsed it into the orchestrator's universal `computer_use` tool,
which composes `default_apps_resolver` + `applescript` (fast path) or
`screenshot` + vision LLM (fallback) per intent.

## Engines by OS

| OS | Engines populated by `uv sync` | Required system tools |
|----|--------------------------------|-----------------------|
| macOS  | pyautogui + pyobjc | `osascript` (built-in) |
| Linux  | pyautogui + python-xlib | `xdotool` (install via apt/brew) |
| Windows | pyautogui + pywinauto + pywin32 | — |

## Bootstrap

```bash
# uv is the only system dep
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: brew install uv

cd desktop-agent
./start.sh
```

`uv sync` runs once on first start; later starts re-use the cached
venv. Stop with `Ctrl+C`. To run permanently, install autostart (see
below).

## macOS permissions (one-time)

The first AppleScript call that tries to drive an app (e.g. `tell
application "Calendar" …`) will trigger macOS to ask for Automation
permission for *that* target app. Approve and the next call works
silently.

`pyautogui` needs Accessibility permission for the running process:

1. Settings → Privacy & Security → Accessibility
2. Add the `uv` binary (or the Python that uv invokes — `~/.local/share/uv/python/cpython-3.12*/bin/python3`)
3. Toggle on

If actions silently no-op, this is usually why.

## Autostart (macOS)

Templates live next to this README — same pattern as `xtts-server`:

```bash
./install-autostart.sh       # creates ~/Library/LaunchAgents/com.voiceassistant.desktop.plist
./uninstall-autostart.sh     # removes it
```

The plist label stays `com.voiceassistant.desktop` (without the
`-agent` suffix) so older installations don't end up with a stale
plist they'd have to manually unload — same launchd identifier,
new contents.

After installation the daemon launches on every login and restarts if
it crashes. Logs land in `~/Library/Logs/voice-assistant/desktop.{out,err}.log`.

## Testing

```bash
# Health (no auth)
curl -s http://localhost:9877/v1/health | jq

# Capabilities (auth required)
TOKEN=$(cat ~/.cache/voice-assistant/desktop-token)
curl -s -H "X-Desktop-Token: $TOKEN" http://localhost:9877/v1/capabilities | jq

# Read a clock-app-style script
curl -s -H "X-Desktop-Token: $TOKEN" -X POST http://localhost:9877/v1/applescript \
  -H 'Content-Type: application/json' \
  -d '{"script":"tell application \"Calendar\" to get name of every calendar"}' | jq

# Which app would the OS launch for a mailto: link?
curl -s -H "X-Desktop-Token: $TOKEN" \
  "http://localhost:9877/v1/default_app?category=mail" | jq

# How long has the user been idle?
curl -s -H "X-Desktop-Token: $TOKEN" \
  http://localhost:9877/v1/cursor_activity | jq
```

## Security notes

- **Localhost-only bind by default** — `127.0.0.1:9877`. Override via `DESKTOP_HOST` env if you must, but consider why.
- **Shared-secret auth** — defends against an unrelated local process from another user account driving the daemon. Not a hardened TLS-mTLS story; that's outside scope for a single-user home assistant.
- **Auditable** — every accepted call is appended to a file you can `tail` or `grep`. Nothing is silent.
- **No `eval`** — every pyautogui call is one of a handful of named primitives.
- **App allowlist** — enforced by the orchestrator's `desktop` tool, NOT this daemon. The daemon is a trusted-client gateway; per-app policy belongs to the LLM tool layer (so it can be edited in `settings.json` without touching the daemon).
- **Read-only stance for `computer_use`** — the orchestrator-side tool runs every intent at `risk="read"` and refuses mutation verbs at three layers: AppleScript category allowlist, vision-rail destructive-text blacklist, and tool-risk classification.  The daemon itself doesn't audit verbs — it trusts the orchestrator to filter — so any direct caller of `/v1/applescript` could in principle send a destructive script.  Treat `DESKTOP_TOKEN` accordingly.

## Remote agent (non-localhost binding)

When the orchestrator and agent run on the SAME host (the default), the
agent binds to `127.0.0.1` and only the local UID can talk to it. When
the agent runs on a DIFFERENT host (your laptop's agent serving a
phone-tethered orchestrator, a kitchen Raspberry Pi, etc.), the agent
must bind to a routable address:

```bash
DESKTOP_HOST=0.0.0.0 ./start.sh
```

**Security implications** (in order of how much they matter):

1. **The shared token is the entire firewall now.** Anyone on the same
   network who guesses (or sniffs) `DESKTOP_TOKEN` can drive the
   daemon — full AppleScript, full screen capture, full mouse control. The
   default token is 32 bytes of randomness, but if it ever leaks into
   a log, screenshot or chat message, rotate immediately. Treat the
   contents of `~/.cache/voice-assistant/desktop-token` as you'd treat
   an SSH private key.
2. **The wire is plaintext HTTP.** If your network is shared (coffee
   shop, dorm), the token + everything you do over the daemon is
   visible on the wire. Use a TLS terminator (caddy, traefik) in
   front, OR run the connection inside a VPN that does TLS for you.
3. **Listening on `0.0.0.0` exposes the daemon to every interface**
   the host has, including any that you forgot were public. Bind to a
   specific interface (e.g. `DESKTOP_HOST=100.64.0.5` for a Tailscale
   IP) rather than the wildcard whenever possible.

The recommended NAT-traversal story is **Tailscale or Wireguard**
rather than implementing our own punchthrough. Both give you a
private, mTLS-ish overlay where the agent advertises `:9877` only to
your tailnet — the public internet sees nothing. Setup is one
`tailscale up` per host, and the agent's bind address can stay on the
tailnet IP. If you can't (or won't) run a VPN, the next option is the
reverse-WSS mode below.

## Reverse-WSS mode (agent dials the orchestrator)

When the agent is behind a NAT you can't reverse (and a VPN isn't
acceptable), flip the polarity: the agent dials the orchestrator over
WSS and stays connected. The orchestrator pushes RPC calls down the
same socket. No inbound port on the agent's side; no NAT punchthrough.

```bash
DESKTOP_MODE=reverse \
ORCHESTRATOR_URL=wss://my-orch.example.com/v1/agent/connect \
DESKTOP_TOKEN=<shared secret> \
./start.sh
```

- `DESKTOP_MODE=reverse` switches the agent from HTTP-server to
  WSS-client. The default `DESKTOP_MODE=server` keeps the existing
  HTTP behaviour — nothing changes for current installs.
- `ORCHESTRATOR_URL` must point at the orchestrator's
  `/v1/agent/connect` endpoint, scheme `ws://` for plain WS or
  `wss://` for TLS. Most production setups put this behind a TLS
  terminator (caddy, traefik) and use `wss://`.
- The agent identifies itself with `DESKTOP_AGENT_ID` (defaults to
  the host's hostname). To register it cleanly with the orchestrator,
  list it in the orchestrator's `DESKTOP_AGENTS` env so the operator
  can flag a `default=True` entry.
- On disconnect the agent reconnects with exponential backoff (1 s,
  2 s, 4 s, 8 s, 16 s, capped at 30 s). The orchestrator-side
  registry holds the most-recent capability snapshot until the agent
  re-handshakes.

The wire protocol is v1, documented at the top of
`orchestrator/app/agent_proxy.py`. No schema versioning yet — bump the
`version` field on a breaking change.

## Environment knobs (summary)

| Var | Default | Notes |
|-----|---------|-------|
| `DESKTOP_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` only for remote agents and see the security section above. |
| `DESKTOP_PORT` | `9877` | TCP port for the HTTP server. Ignored in reverse mode. |
| `DESKTOP_TOKEN` | random + saved | Shared secret. Auto-generated on first run if unset; copy into orchestrator env. |
| `DESKTOP_AGENT_ID` | hostname | Stable identifier the orchestrator routes by (multi-agent mode). |
| `DESKTOP_MODE` | `server` | `server` (HTTP, default) or `reverse` (WSS client). |
| `ORCHESTRATOR_URL` | empty | Required in reverse mode — e.g. `wss://my-orch.tailnet.ts.net/v1/agent/connect`. |
| `DESKTOP_AUDIT_LOG` | `~/.cache/voice-assistant/desktop-audit.log` | Path for the per-call audit trail. |
