# voice-assistant

A self-hosted, offline-first voice assistant for your home — wake-word,
streaming ASR, agentic LLM with tools, streaming TTS, semantic memory,
speaker ID, personal item store, and desktop automation.  Runs entirely
on your own hardware; no cloud API keys; works without internet for
most tasks.

> **Status**: actively developed.  Single-user / single-household scope
> by design (one Mac + browser).  Multi-agent mode supports a primary
> Mac plus remote agents (e.g. a work PC) over Tailscale.

## What it does

- 🎙️ **Browser wake-word** ("Hey Jarvis" out of the box, retrain
  with [openWakeWord](https://github.com/dscripka/openWakeWord)).
- 📝 **Streaming transcription** via whisper.cpp (Metal/CUDA, served
  by LocalAI) or any OpenAI-compatible Whisper server (mlx-whisper,
  faster-whisper).
- 🧠 **Local LLM agent loop** via any OpenAI-compatible endpoint —
  default is [LocalAI](https://localai.io) (llama.cpp on Metal/CUDA)
  running Gemma 4 E4B QAT: multimodal, so vision rides on the same
  endpoint, and ASR shares the same server.  Works with Ollama, vLLM,
  llama.cpp, LM Studio out of the box.
- 🛠️ **Tools** — calculator, weather, web search, news, translate,
  reminders/timers, semantic memory, computer control via
  AppleScript or vision-loop, and Mail/Calendar bridges through the
  desktop-agent.
- 🔊 **Streaming TTS** via local
  [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) — record a
  6-second sample in the browser to clone any household member's
  voice; the assistant picks the right voice automatically when it
  speaks on someone's behalf.
- 👥 **Speaker ID** — d-vectors with
  [resemblyzer](https://github.com/resemble-ai/Resemblyzer), so the
  assistant knows who's speaking and routes replies / voice messages
  to the right person.
- 📦 **Personal item store** — drop links, notes, screenshots,
  shorts/videos into per-user categories; hybrid search (BM25 +
  semantic via RRF), LLM-powered auto-sort with preview, soft-delete
  trash with 7-day GC.  Items are reachable from both the UI and the
  agent (the LLM can read and write your store).
- 📬 **Household voicemail** — leave voice messages for another
  household member; speaker-ID routes them, and the recipient gets a
  Web Push with one-tap playback / reply.
- 🔐 **Three-layer read-only gate** on desktop automation —
  destructive verbs (delete/send/empty/move) refused at the prompt
  layer, the static classifier, and the vision-loop click filter.
- 📊 **Token usage observability** — per-tool / per-user counters,
  daily aggregates, live `/api/stats` dashboard so you can see where
  your LLM budget is going.
- 📱 **Web Push** notifications when you're away from the tab.

Designed to be **privately useful, not feature-equivalent with Siri**:
all data stays on your hardware; no telemetry; no accounts; pluggable
locally-hosted models.

## Quickstart

Prereqs: [`uv`](https://docs.astral.sh/uv/) — every service runs
natively in its own uv venv (no Docker needed); LocalAI (recommended —
one server covers LLM + ASR) or any OpenAI-compatible chat endpoint
(Ollama, vLLM, LM Studio).  Docker is only used by the optional
Linux-server path (`docker-compose.yml`).

**Fastest path** — if you have [go-task](https://taskfile.dev) installed
(`brew install go-task`), the setup wizard probes all backends and
writes `.env` for you:

```bash
git clone <repo> voice-assistant
cd voice-assistant
task wizard          # probe → configure → optionally start
```

**Manual path:**

```bash
git clone <repo> voice-assistant
cd voice-assistant

# 1. Start the LLM+ASR server (any OpenAI-compatible).  Default is
#    LocalAI at http://localhost:1240 with a multimodal model:
#      brew install localai
#      local-ai backends install llama-cpp && local-ai backends install whisper
#      local-ai models install gemma-4-e4b-it-qat-q4_0
#      local-ai models install whisper-large-q5_0
#      local-ai run --address 127.0.0.1:1240 --models-path ~/.localai/models
#    Override LLM_URL / LLM_MODEL / WHISPER_URL in docker-compose.yml or .env.

# 2. Start the host-side TTS service (XTTS-v2, ~1.8 GB model
#    downloaded on first run, cached at ~/.cache/voice-assistant/xtts/).
cd xtts-server && ./start.sh &        # uv handles deps; see its README

# 3. Start the host-side desktop-agent (AppleScript / pyautogui bridge).
cd desktop-agent && ./start.sh &      # same pattern

# 4. Start the orchestrator (FastAPI + frontend, native uv venv).
cd orchestrator && ./start.sh &      # same uv pattern as the others

# Open http://localhost:8080
```

**Common task shortcuts** (requires `brew install go-task`):

```bash
task up         # start the orchestrator (launchd)
task down       # stop it
task logs       # tail orchestrator logs
task health     # probe all backends at once
task upgrade    # git pull + uv sync + restart
task test       # run pytest (native uv venv)
```

For a complete walk-through including macOS permissions
(Accessibility, Automation), wake-word training, and the multi-agent
setup, see [`docs/deployment.md`](docs/deployment.md).

## Repository layout

```
voice-assistant/
├── orchestrator/          # FastAPI brain — agent loop, tools, storage
│   ├── app/               # Source (see orchestrator/CLAUDE.md)
│   ├── tests/             # pytest, 275 tests — `task test` (uv venv)
│   └── start.sh           # native service entry (uv); Dockerfile = Linux path
├── desktop-agent/         # Host-side HTTP/WSS gateway for desktop automation
├── xtts-server/           # Host-side TTS service (Coqui XTTS-v2)
├── frontend/              # Browser PWA — wake-word, mic, UI, push
├── docs/                  # Architecture, cookbooks, deployment
├── data/                  # Local DB + VAPID key + custom voices (gitignored)
├── docker-compose.yml     # Orchestrator service + env vars (well-documented)
└── README.md / CLAUDE.md  # This file + LLM contributor entry point
```

## Architecture in one screen

```
┌─────────────────┐                    ┌────────────────────┐
│   Browser PWA   │  WS audio + ctrl   │   orchestrator     │
│  (wake-word,    │ ─────────────────► │   (FastAPI, agent  │
│   mic VAD,      │ ◄───────────────── │    loop, tools,    │
│   image attach) │     TTS stream     │    storage)        │
└─────────────────┘                    └─────────┬──────────┘
                                                 │
   ┌──────────────────────────────┬──────────────┴────────┐
   │ LocalAI (host, :1240)        │ xtts-server           │
   │ llama.cpp + whisper.cpp      │ (host, :9876)         │
   │ ASR + LLM (text+vision+tools)│ TTS XTTS-v2 streaming │
   │ or any OpenAI-compatible     │                       │
   │ (Ollama / vLLM / mlx-whisper)│                       │
   └──────────────────────────────┴───────────────┬───────┘
                                                  │
                                       ┌──────────┴──────────┐
                                       │ desktop-agent       │
                                       │ (host, :9877)       │
                                       │ AppleScript /       │
                                       │ pyautogui / vision  │
                                       └─────────────────────┘
```

For data flow, the agent-loop state machine, and the three-layer
read-only defence on `computer_use`, read
[`docs/architecture.md`](docs/architecture.md).

## Where to look for X

| You want to … | Go to |
|---|---|
| Understand the project | [`README.md`](README.md), [`docs/architecture.md`](docs/architecture.md) |
| Run locally | This README's Quickstart, then [`docs/deployment.md`](docs/deployment.md) |
| Add a new LLM tool | [`docs/adding-a-tool.md`](docs/adding-a-tool.md) |
| Add a new UI/voice language | [`docs/adding-a-locale.md`](docs/adding-a-locale.md) |
| Work on the orchestrator | [`orchestrator/CLAUDE.md`](orchestrator/CLAUDE.md) |
| Work on desktop automation | [`desktop-agent/README.md`](desktop-agent/README.md) |
| Work on the frontend | [`frontend/CLAUDE.md`](frontend/CLAUDE.md) |
| Contribute / open a PR | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| LLM agent doing changes | [`CLAUDE.md`](CLAUDE.md) (root, navigation map) |

## Privacy and offline-first

Every default points at localhost:

- Whisper / LLM / TTS / desktop-agent all on `localhost` ports.
- Embedding model (paraphrase-multilingual-MiniLM-L12) downloads
  once via fastembed, then runs offline.
- DuckDuckGo / Open-Meteo / news fetch DO require internet — those
  tools degrade to a graceful "offline" answer when the network is
  out.  Everything else (LLM turns, memory, reminders, computer
  control, voice messages) works without internet.

No telemetry, no cloud accounts, no LLM API keys.  The only secrets
are local: `data/vapid_private.pem` (Web Push, auto-generated) and
the `DESKTOP_TOKEN` shared secret (auto-generated; copy into env).

## Related work

[Open WebUI](https://github.com/open-webui/open-webui) — closest self-hosted alternative.

## License

[MIT](LICENSE) for the code.  Bundled / downloaded models carry their
own licenses — most notably the XTTS-v2 weights are Coqui CPML
(non-commercial).  See LICENSE for the full breakdown.

## Acknowledgements

This project leans on the work of: OpenAI Whisper, Apple's MLX team,
Coqui XTTS-v2, openWakeWord, fastembed/Qdrant, resemblyzer, LM Studio
and Ollama for OpenAI-compatible local serving, and the FastAPI /
httpx / pyautogui ecosystems.
