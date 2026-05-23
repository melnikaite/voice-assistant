# desktop-agent — contributor notes

Service overview lives in [`README.md`](README.md).  This file is for
folks (LLM or human) coming in to modify the agent.

## Mental model

The desktop-agent is **a trusted-client gateway**, not a policy layer.
It exposes primitives (AppleScript, pyautogui, screenshot, default-app
lookup, cursor activity) over HTTP and a reverse-WSS variant; the
orchestrator decides what to allow per-tool.  This split lets us edit
policy without touching the daemon, and lets a future MS-Teams agent
or Windows agent expose the SAME primitives without inheriting macOS
verb assumptions.

## File map

```
desktop-agent/
├── desktop-agent.py             # Single-file service (FastAPI + Backend ABC)
├── pyproject.toml               # uv-managed deps (pyautogui + pyobjc on Mac, etc.)
├── start.sh                     # `uv run` wrapper
├── install-autostart.sh         # launchd plist installer (macOS)
├── uninstall-autostart.sh
├── com.voiceassistant.desktop.plist.template
└── README.md                    # User-facing docs
```

## Key code in `desktop-agent.py`

| Section | What lives there |
|---|---|
| `Backend(ABC)` | The contract for an OS-specific implementation |
| `MacOSBackend` | macOS via osascript + pyautogui + NSWorkspace (JXA) |
| `LinuxBackend` | Stub — pyautogui + xdotool TBD |
| `WindowsBackend` | Stub — pywinauto TBD |
| `_pick_backend()` | Boots the right subclass for `sys.platform` |
| `_handlers` HTTP routes | One endpoint per primitive (`/v1/applescript`, `/v1/pyautogui`, …) |
| Auth (`X-Desktop-Token`) | Shared secret read from env or persisted under `~/.cache/voice-assistant/desktop-token` |
| Audit log | Every accepted call appended to JSONL at `~/.cache/voice-assistant/desktop-audit.log` |
| `_reverse_loop()` | The WSS client for NAT'd agents (`DESKTOP_MODE=reverse`) |

## Adding a new backend (Linux / Windows)

1. Subclass `Backend` and implement every abstract method.  Look at
   `MacOSBackend` for the wire contracts.
2. Add a branch in `_pick_backend()`.
3. Update the capabilities table in `/v1/capabilities` so the
   orchestrator advertises the right tools for that platform.
4. Add the system-deps to `pyproject.toml`.  Use uv's
   `dependency-groups` so macOS users don't pull `pywin32`.

The orchestrator's `computer_use` tool now gates AppleScript on
capabilities (`has_capability_cached("applescript")`), so a Linux
backend that returns `applescript: false` automatically routes through
the vision-loop fallback.  No orchestrator changes needed.

## Adding a new primitive (rare)

Primitives should be small, OS-portable, and unambiguous.  Adding one:

1. Add the abstract method to `Backend`.
2. Implement on every backend (or leave a `raise NotImplementedError`
   on the platforms that can't serve it — the capability advertised in
   `/v1/capabilities` will tell the orchestrator to skip).
3. Add the HTTP route + the WSS RPC method.  The reverse-WSS protocol
   is documented at the top of `orchestrator/app/agent_proxy.py`.
4. Update the capabilities map.

## Gotchas

- **Accessibility permission applies to the launching process**, not
  the script.  If you run via `uv run` and Accessibility is granted to
  `uv`, pyautogui works.  If you grant it to a different Python, the
  permissions silently break.
- **Automation prompts trigger per target app**.  First AppleScript
  that touches Calendar.app prompts; second is silent.  Don't catch
  the permission-denied error — let it fail loudly so the user knows.
- **`network_mode: host` on the orchestrator** means the agent's
  `127.0.0.1` is reachable from inside the container without
  `host.docker.internal`.  This is why `DESKTOP_URL=http://localhost:9877`
  just works.
- **Reverse-WSS keeps the agent connected**.  The orchestrator-side
  `/v1/agent/connect` endpoint is the docking point; the agent's
  client is `_reverse_loop()` with exponential backoff.  Don't add
  retry logic outside that one place.
- **Audit log persists across daemon restarts**.  Don't truncate it —
  it's the only forensic trail when the LLM mis-clicks.
