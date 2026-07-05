# Deployment

End-to-end walkthrough, `git clone` to "talking to it". macOS first; Linux notes inline. Orchestrator runs in Docker; everything else (ASR, LLM, TTS, desktop-agent) runs natively on the host.

## 0. Quick path — setup wizard

If you already have the host services running (LLM, Whisper server, optionally TTS and desktop-agent), the wizard covers §1–§5 in one shot:

```bash
brew install go-task          # task runner — one-time
task wizard                   # probe → write .env → optionally start
```

The wizard probes all five backends, reports what's ready and what needs attention, and offers to run `docker compose up -d` when everything is green. **It never installs or modifies globally-installed tools** — your existing Whisper server fork on `:18000` is detected and left exactly as-is.

Other useful shortcuts:

```bash
task health     # quick status of all 5 backends
task upgrade    # git pull + rebuild + restart
task logs       # tail orchestrator logs
task test       # run the pytest suite inside the container
```

Read on for the full manual walkthrough, or use this as a reference for what each service does.

---

## 1. Prerequisites

| What                  | Version       | Why                                                |
|-----------------------|---------------|----------------------------------------------------|
| Docker                | macOS: Desktop 4.34+ with host networking; Linux: native | WebRTC needs ICE candidates on the host's interfaces; bridge-net 172.x.x.x is unreachable from the browser. `docker-compose.yml:5-13`. |
| `uv`                  | latest        | Hermetic Python toolchain for the host-side services (XTTS, desktop-agent). |
| LLM provider          | any OpenAI-compatible chat endpoint | LM Studio (default), Ollama, vLLM, llama.cpp. |
| Whisper provider      | any OpenAI-compatible transcription endpoint on `:18000` | Apple Silicon → mlx-whisper. Linux/CUDA → vLLM Whisper or `whisper.cpp` server. |

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

System Python 3.12 is NOT required — uv ships its own toolchain.

**macOS Docker Desktop**: Settings → Resources → Network → toggle **"Enable host networking"**. Without it WebRTC falls back to ICE relay candidates that the browser can't reach. **Linux Docker**: host networking is the default.

## 2. Pick your LLM provider

Default config targets LM Studio on `:1234` running Gemma 4 E4B. Switch by overriding `LLM_URL` and `LLM_MODEL` (optionally `LLM_VISION_URL` / `LLM_VISION_MODEL`).

| Provider     | Endpoint default       | Tool-call support              | Vision (image input) |
|--------------|------------------------|--------------------------------|----------------------|
| LM Studio    | `http://localhost:1234`| Yes (model-dependent)          | Yes — load a multimodal model like Gemma 4 |
| Ollama       | `http://localhost:11434`| Yes (model-dependent)         | Some models (`llava`, `bakllava`) |
| vLLM         | configurable           | Yes                            | Yes if the served model is multimodal |
| llama.cpp    | `http://localhost:8080`| Yes (server build)             | LLaVA via `--mmproj` |

Easiest vision path is a single multimodal model on LM Studio (default). To split text/vision across providers, set `LLM_TEXT_URL` / `LLM_TEXT_MODEL` / `LLM_VISION_URL` / `LLM_VISION_MODEL` (`docker-compose.yml:46-54`).

The model MUST support OpenAI-style tool calling for the agent loop. If it emits free text instead of `tool_calls`, the orchestrator falls back gracefully but most agent features degrade.

## 3. Start the host services

Bring them up in this order. For autostart see §12.

### 3.1 Whisper (`:18000`)

Apple Silicon:

```bash
uv tool install mlx-whisper
mlx-whisper-server --port 18000 --model mlx-community/whisper-large-v3-mlx
```

Linux / CUDA: any OpenAI-Whisper-compatible server on `:18000` — `whisper.cpp`'s `server` binary or vLLM with a Whisper model. Only `/v1/audio/transcriptions` + `/v1/models` are needed.

### 3.2 LLM (`:1234` for LM Studio default)

LM Studio:

1. Download from https://lmstudio.ai.
2. Download a multimodal model (default: `google/gemma-4-e4b`).
3. Local Server → Start Server (defaults to `:1234`).
4. Turn off "JIT model loading" for a warm cache on first run.

Ollama:

```bash
brew install ollama          # or: download from ollama.com
ollama serve
ollama pull llama3.1:8b
```

In `docker-compose.yml`: `LLM_URL: http://localhost:11434`, `LLM_MODEL: llama3.1:8b`.

### 3.3 XTTS (`:9876`)

```bash
cd xtts-server
./start.sh
```

First run downloads ~1.8 GB of weights into `~/.cache/voice-assistant/xtts/`. See `xtts-server/README.md` for custom-speaker config and autostart.

### 3.4 desktop-agent (`:9877`)

```bash
cd desktop-agent
./start.sh
```

First run generates a shared secret at `~/.cache/voice-assistant/desktop-token`. **Copy that value** — needed in step 4. Without the agent the `computer_use` and related tools degrade to "not reachable". See `desktop-agent/README.md` for endpoints, permissions, Backend ABC pattern.

## 4. Configure

```bash
cp .env.example .env
# Edit: set DESKTOP_TOKEN, change HOST_PORT if 8080 is busy
```

`.env.example` covers what you'll actually want to override per machine. Every other env var has a sensible default in `docker-compose.yml`; only set it in `.env` to depart from defaults.

## 5. Start the orchestrator

```bash
docker compose up -d
```

First boot: builds the orchestrator image (~2 min), creates `data/assistant.db` (SQLite, WAL) and `data/vapid_private.pem` (Web Push key), downloads the fastembed model on first ASR/memory call. Subsequent boots ~5 s.

Open `http://localhost:8080`. Health-check:

```bash
docker compose logs orchestrator | tail -50
curl -s http://localhost:8080/api/config | jq
```

## 6. Browser automation (optional)

To enable "open this URL", "read the current page", "what tabs are open" voice commands, Chrome must be started with remote debugging enabled:

```bash
# macOS
open -a "Google Chrome" --args --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222

# Windows (PowerShell)
Start-Process chrome -ArgumentList "--remote-debugging-port=9222"
```

Override the port via `CHROME_DEBUG_PORT` env on the desktop-agent side.  The daemon probes every 30 s — Chrome can be opened or closed mid-session and the capability flag updates automatically within one poll cycle.

## 6b. Global dictation hotkey (optional)

A system-wide keyboard shortcut lets you trigger voice dictation from any application — without switching to the browser tab.

**Default combo:** `Ctrl+Shift+Space` on all platforms.  Override via `HOTKEY_COMBO` env on the desktop-agent.

**How it works:**
1. desktop-agent listens for the combo with pynput (daemon thread)
2. On press: POSTs to the orchestrator's `/api/hotkey/ptt` endpoint
3. Orchestrator fans a `ptt_trigger` event to every connected browser session
4. Browser toggles PTT — first trigger starts recording, second trigger (or 10-second timeout) stops

**Behaviour** — toggle mode:
- Press once → PTT starts (browser shows "TALK" button pressed)
- Press again → PTT stops immediately
- No second press → auto-release after 10 seconds

**Setup:**
```bash
# Use the default Ctrl+Shift+Space (no config needed)
# Or override in desktop-agent/.env or your shell profile:
export HOTKEY_COMBO="<cmd>+<shift>+space"       # macOS-preferred
export ORCHESTRATOR_WEBHOOK_URL="http://localhost:8080"  # default — change if needed

cd desktop-agent && ./start.sh
```

**macOS permission note:** pynput uses Quartz and requires Accessibility access — same as pyautogui (already granted if desktop-agent works). Symptom of missing permission: `hotkey: failed to start listener`.

**Disable** the listener: `HOTKEY_ENABLED=0 ./start.sh`.

Check status:
```bash
TOKEN=$(cat ~/.cache/voice-assistant/desktop-token)
curl -s -H "X-Desktop-Token: $TOKEN" http://localhost:9877/v1/hotkey/status | jq
```

## 7. Live streaming to family devices (optional)

Once the desktop-agent is running you can stream the host camera or any Chrome tab to any device on your LAN (tablet, TV, phone) — no extra software needed.

**Voice commands:**
```
"покажи камеру"             → /api/stream/camera
"stream this tab"           → /api/stream/tab
"покажи вкладку на телевизоре"
```

**How it works:** The orchestrator proxies a MJPEG stream from the desktop-agent through `/api/stream/camera` or `/api/stream/tab`. The frontend renders it inline as a `<img>` tag. Any family device with a `va_session` cookie (i.e. logged in to the assistant) can open the stream URL directly too:

```
http://<orchestrator-host>:8080/api/stream/camera?fps=15
http://<orchestrator-host>:8080/api/stream/tab?fps=5
```

**Stream sources and caps:**

| Source | Default fps | Max fps | Notes |
|--------|------------|---------|-------|
| Camera | 15 | 30 | OpenCV device 0 |
| Tab | 5 | 15 | CDP `Page.captureScreenshot` loop |

Stop the stream by clicking the **Stop** button that appears in the frontend, or just close the tab on the viewing device.

**Relay vs P2P:** The orchestrator runs on your LAN — all stream traffic stays local. Remote devices (outside your home network) see the same stream via the orchestrator's public URL (if you've exposed it with a reverse proxy; see §10 for HTTPS setup).

## 8. macOS permissions (one-time)

Three buckets:

- **Microphone** — granted via the browser the first time you click "Start" and hit the `getUserMedia` prompt.
- **Accessibility** — pyautogui needs this. Settings → Privacy & Security → Accessibility → add the `uv` binary (or `~/.local/share/uv/python/cpython-3.12*/bin/python3`). Symptom of missing permission: clicks/keys silently no-op.
- **Automation** — each AppleScript-target app triggers a separate prompt the first time ("Allow Terminal to control Calendar?"). Approve once.

Full details in `desktop-agent/README.md`'s "macOS permissions" section.

## 9. Wake-word

Default phrase **"Hey Jarvis"**. Model lives at `frontend/models/hey_jarvis_v0.1.onnx`, loaded by the browser via `frontend/wake.js`.

Swap by:

1. **Pre-trained openWakeWord model.** "alexa", "hey mycroft", "hey rhasspy" are available. Drop the `.onnx` into `frontend/models/`, set `WAKE_WORD_NAME` in `docker-compose.yml` (without `.onnx`). Restart orchestrator.
2. **Train your own.** ~30 min on a free Colab T4. Walkthrough in `xtts-server/README.md::Training a custom wake-word`.

Threshold: `WAKE_WORD_THRESHOLD` (default `0.5`). Higher = fewer false wakes; lower = more sensitive.

## 10. Web Push

Lets the assistant notify you when the tab is closed — voicemail, reminders. Requires:

- HTTPS or `localhost`. Push API refuses `http://` on any other host.
- A VAPID key pair. Auto-generated on first boot into `data/vapid_private.pem`; public half served at `/api/push/vapid_public_key`.

Click the bell icon to grant permission and register the subscription. `frontend/sw.js` handles the push event, shows a system notification, routes the click back into the open tab.

For HTTPS via reverse proxy (caddy / nginx / traefik), point it at `:8080` and forward WebSocket frames with `Connection: Upgrade`. No extra auth wiring needed.

## 11. Multi-agent setup

For multiple agents — home Mac + work PC reachable from the same orchestrator — set `DESKTOP_AGENTS`:

```yaml
DESKTOP_AGENTS: |
  [{"agent_id":"macbook","url":"http://localhost:9877",
    "token":"<token-from-macbook>","default":true},
   {"agent_id":"work-pc","url":"http://work.tailnet.ts.net:9877",
    "token":"<token-from-work-pc>"}]
```

Full schema: `docker-compose.yml:175-189`. The orchestrator caches each agent's `/v1/capabilities` for 60 s and renders "@macbook" / "@work-pc" inline in the UI.

For NAT traversal (work PC behind corporate firewall), the recommended path is Tailscale or WireGuard. If a VPN isn't possible, the agent supports **reverse-WSS mode** where it dials the orchestrator — see `desktop-agent/README.md::Reverse-WSS mode`.

## 12. Autostart

macOS launchd templates:

```bash
cd orchestrator && ./install-autostart.sh
cd xtts-server && ./install-autostart.sh
cd desktop-agent && ./install-autostart.sh
```

Each drops a plist into `~/Library/LaunchAgents/`, wires auto-restart, points stdout/stderr at `~/Library/Logs/voice-assistant/`, bakes in PATH for `uv`. Disable with `./uninstall-autostart.sh`.

For Linux, write a systemd user unit pointing at `./start.sh` in each directory.

After installing the orchestrator agent, `task up` / `task down` / `task restart` control it.  Upgrading code does **not** restart a running daemon — `task upgrade` bundles pull + `uv sync` + restart.  To remove, run `./uninstall-autostart.sh` **before** deleting the checkout, or launchd will keep respawning a missing binary.

On a Linux server the docker-compose variant applies instead: `docker compose up -d` in your boot script, with `restart: unless-stopped` already set.

## 13. Backup / migration

Everything stateful lives under `data/`:

- `assistant.db` — SQLite WAL. Sessions, utterances, token usage, memory, reminders, voicemail, pending actions, push subs, auth.
- `vapid_private.pem` — Web Push signing key.
- `voice_messages/` — voicemail audio (one WAV per message).
- `custom_voices/` — user-recorded XTTS voice references.

```bash
task down                       # docker compose stop orchestrator on Linux
tar -czf voice-assistant-backup-$(date +%Y%m%d).tar.gz data/
task up                         # docker compose start orchestrator on Linux
```

Restore: stop, untar, start. Schema is idempotent (every migration is `CREATE IF NOT EXISTS`); a backup from an older version gains new columns on first boot of the new orchestrator.

XTTS model (`~/.cache/voice-assistant/xtts/`, 1.8 GB) is reproducible — not in the backup. Same for the fastembed model.

## 14. Troubleshooting

| Symptom                                         | Likely cause                                      | Fix                                                              |
|-------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|
| "No audio coming out"                           | xtts-server not running                           | `cd xtts-server && ./start.sh`. Check `curl :9876/v1/health`.    |
| "Desktop agent isn't reachable"                 | desktop-agent not running or token mismatch       | `curl :9877/v1/health`. 401 → token in compose ≠ `~/.cache/voice-assistant/desktop-token`. |
| LLM refuses tool calls / never invokes a tool   | Model doesn't support OpenAI tool-call format     | Swap to a model that does (Gemma 4, Mistral Large, Llama 3.1+).  |
| WebRTC connect fails / mic light blinks once    | Host networking not enabled in Docker Desktop     | Settings → Resources → Network → Enable host networking.         |
| Wake-word never triggers                        | Threshold too high, or `.onnx` not loaded         | Browser console for `wake models loaded`. Lower `WAKE_WORD_THRESHOLD`. |
| Web Push subscribe fails                        | Not on HTTPS or `localhost`                       | Reverse-proxy with TLS, or test on `localhost`.                  |
| ASR returns empty / garbled                     | Whisper not running or wrong model loaded         | `curl :18000/v1/models`. Restart with correct model id matching `WHISPER_MODEL`. |
| `pyautogui` clicks silently no-op (macOS)       | Accessibility permission missing                  | System Settings → Privacy → Accessibility → add `uv` / Python.    |
| `AppleScript` "Application can't run scripts"   | Automation permission missing for that app        | First AppleScript call against the app triggers a prompt; approve once. |
| Logs show `i18n: unknown key %r`                | Missing translation for current locale            | Add the key to `orchestrator/app/i18n.py::CATALOG`. `docs/adding-a-locale.md`. |
| "LLM hit max_tokens"                            | Model thought too long; finished without content  | Lower `reasoning_effort` for that tool, or raise context budget. |
| Container restart loop                          | Missing env var or DB schema mismatch             | `docker compose logs orchestrator`.                              |

For anything not listed: `docker compose logs -f orchestrator` shows the full pipeline trace (ASR ms, agent loop steps, tool calls, finish reasons, token usage).
