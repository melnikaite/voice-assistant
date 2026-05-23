# orchestrator

The FastAPI brain of voice-assistant. Everything stateful lives here — audio-turn pipeline, agent loop, every tool, the SQLite store, and the browser frontend it serves.

For contributor patterns see [`CLAUDE.md`](CLAUDE.md). For the project as a whole see the [root README](../README.md).

## What it does

- **FastAPI app** (`app/main.py`) — boots schema, VAPID keypair, embedding model, speaker encoder, TTS client, desktop-agent registry, scheduler, pending-action executor in `lifespan`; serves HTTP + WS.
- **WebSocket endpoint** (`app/ws.py`) — one socket per browser tab, WebRTC signalling + JSON control. Audio runs over a peer WebRTC connection.
- **Per-turn pipeline** (`app/pipeline.py`) — ASR + speaker ID in parallel, semantic-memory lookup, passphrase / voicemail / vision short-circuits, then the agent loop. Hookable for testing.
- **Agent loop** (`app/agent.py`) — OpenAI-style tool-use. 4-step cap, terminal-tool short-circuit, risk gating via `pending_actions`.
- **Tools** (`app/tools/`) — calculator, weather, web_search, news, translate, set_reminder, my_history, memory, settings, inbox, desktop, computer_use, look_at_screen, general_answer. Auto-registered via `@tool(...)`.
- **Storage** (`app/storage/`) — SQLite with WAL + thread-local connections.
- **REST surface** (`app/main.py`) — `/api/stats` token-usage dashboard, push subs, voicemail inbox, speaker enrollment, custom-voice recording, auth.

## Layout

```
orchestrator/
├── Dockerfile
├── pyproject.toml         # test + lint deps (runtime deps in Dockerfile)
├── tests/                 # pytest suite — offline, runs in container or via uv
└── app/
    ├── main.py            # FastAPI app, REST routes, lifespan
    ├── ws.py              # WS endpoint + per-tab Session FSM
    ├── pipeline.py        # audio-in → text-out, per-turn pipeline
    ├── agent.py           # LLM tool-use loop (4-step bound)
    ├── llm.py             # back-compat thin wrapper around agent
    ├── llm_utils.py       # OpenAI-style /chat/completions client
    ├── asr.py             # Whisper HTTP client
    ├── tts.py             # XTTS-server HTTP client + streaming
    ├── webrtc.py          # aiortc peer + outbound TTS track
    ├── vad.py             # Silero VAD wrapper
    ├── speaker.py         # resemblyzer GE2E encoder + identify()
    ├── memory.py          # fastembed embeddings + cosine retrieval
    ├── i18n.py            # CATALOG[key][lang] + pick_lang/t helpers
    ├── messages.py        # legacy hard-coded reply constants
    ├── intents.py         # zero-LLM local intents (replay / new_topic)
    ├── vision.py          # one-shot multimodal call
    ├── vision_loop.py     # agentic screenshot → plan → click loop
    ├── vision_primitives.py
    ├── net.py             # has_internet() probe + cache
    ├── desktop_client.py  # multi-agent registry + RPC to desktop-agent
    ├── agent_proxy.py     # reverse-WSS server for NAT-traversed agents
    ├── push.py            # VAPID + pywebpush fan-out
    ├── scheduler.py       # apscheduler — reminders fire here
    ├── pending_executor.py# polls pending_actions, runs approved ones
    ├── registry.py        # client_id → Session lookup
    ├── search.py          # DuckDuckGo + trafilatura fetcher
    ├── user_files.py      # per-profile settings.json + memory.md
    ├── tools/
    │   ├── __init__.py    # auto-discovery via pkgutil.iter_modules
    │   ├── base.py        # @tool decorator, dispatch, ToolCtx, risk enum
    │   └── <tool>.py × 15
    └── storage/
        ├── db.py          # thread-local sqlite3 + WAL pragmas
        ├── schema.py      # init_schema() + idempotent ALTER blocks
        ├── sessions.py
        ├── utterances.py
        ├── reminders.py
        ├── speaker_profiles.py
        ├── custom_voices.py
        ├── token_usage.py
        ├── pending_actions.py
        ├── auth_sessions.py
        ├── push_subscriptions.py
        └── voice_messages.py
```

## Running locally

Runs as a Docker container with host networking:

```bash
docker compose up -d orchestrator
docker logs -f va-orchestrator
```

Host requirements (defaults in parens):

- LLM at `LLM_URL` (`http://localhost:1234`, LM Studio). Multimodal model recommended.
- `mlx-whisper` (or any OpenAI-Whisper-compatible server) at `WHISPER_URL` (`http://localhost:18000`).
- `xtts-server` at `TTS_URL` (`http://localhost:9876`).
- `desktop-agent` at `DESKTOP_URL` (`http://localhost:9877`) with `DESKTOP_TOKEN`. Optional — without it the computer-control tools return "desktop agent isn't reachable".

Hot reload is on (`uvicorn --reload`).

## Running tests

Offline by design:

```bash
# Inside the running container (matches CI):
docker exec va-orchestrator pytest /app/tests -q

# Or on the host with uv:
cd orchestrator
pip install -e ".[test]"
pytest
```

`tests/conftest.py` swaps storage onto `/tmp/voice-assistant-test.db` (NOT `:memory:` — see [`CLAUDE.md`](CLAUDE.md) Gotchas), wipes between tests, stubs the network probe to a dead address. `make_agent_ctx` fixture builds `AgentContext` objects so tool-level tests don't need a real WS session.

## Endpoints

| Method | Path                                | Notes |
|--------|-------------------------------------|-------|
| `GET`  | `/`                                 | Browser frontend (static mount from `frontend/`) |
| `GET`  | `/api/config`                       | Wake-word knobs surfaced to the browser at boot |
| `GET`  | `/health`                           | Liveness + Whisper / LLM backend probes |
| `GET`  | `/api/stats?range=day\|week\|month` | Token-usage dashboard |
| `WS`   | `/ws?client_id=<uuid>`              | Main voice endpoint — WebRTC signalling + JSON state |
| `WS`   | `/v1/agent/connect`                 | Reverse-WSS for NAT-traversed desktop-agents |
| `POST` | `/dev/respond`                      | Bypass ASR — feed text straight to the agent loop (dev/test) |
| `POST` | `/api/speakers/enroll`              | Enroll / update a speaker profile |
| `GET`  | `/api/speakers?client_id=<uuid>`    | List enrolled profiles |
| `PATCH`| `/api/speakers/{id}`                | Set a per-speaker TTS voice override |
| `DELETE`|`/api/speakers/{id}`                | Drop a profile |
| `GET`  | `/api/voices`                       | XTTS built-ins + user custom voices |
| `POST` | `/api/custom_voices/record`         | Save a user-recorded reference WAV (≥ 2 s) |
| `DELETE`|`/api/custom_voices/{id}`           | Drop a custom voice |
| `POST` | `/api/auth/setup_passphrase`        | First-time passphrase setup |
| `POST` | `/api/auth/login`                   | Verify passphrase, mint cookie session |
| `POST` | `/api/auth/logout`                  | Revoke session |
| `POST` | `/api/auth/rotate_passphrase`       | Change passphrase (requires login) |
| `GET`  | `/api/me`                           | Logged-in profile (auth required) |
| `GET`  | `/api/users/{pid}/memory`           | Per-profile memory.md (auth required) |
| `PUT`  | `/api/users/{pid}/memory`           | Replace memory.md |
| `GET`  | `/api/users/{pid}/settings`         | Per-profile settings.json |
| `PUT`  | `/api/users/{pid}/settings`         | Replace settings (excluding passphrase hash) |
| `GET`  | `/api/users/{pid}/pending`          | Pending-action queue |
| `POST` | `/api/pending/{id}/approve`         | Approve a queued high-write action |
| `POST` | `/api/pending/{id}/reject`          | Reject |
| `GET`  | `/api/push/vapid_public_key`        | Public VAPID key for `pushManager.subscribe` |
| `POST` | `/api/push/subscribe`               | Register a PushSubscription (auth required) |
| `DELETE`|`/api/push/subscribe`               | Unregister by endpoint |
| `GET`  | `/api/users/{pid}/voicemail`        | Inbox list |
| `GET`  | `/api/users/{pid}/outgoing_voicemail`| Sent voicemail + replies |
| `GET`  | `/api/voicemail/{id}/audio`         | Stream WAV bytes |
| `POST` | `/api/voicemail/{id}/listened`      | Mark as listened |
| `POST` | `/api/voicemail/{id}/reply`         | Save a textual reply |
| `DELETE`|`/api/voicemail/{id}`               | Delete |
| `GET`  | `/api/agents`                       | Registered desktop-agents + reachability + default flag |

WS protocol (message types in both directions) is documented at the top of [`app/ws.py`](app/ws.py).

## Storage

SQLite at `$DB_PATH` (default `/data/assistant.db`, host `./data/assistant.db`). Opened lazily, one connection per pool thread; WAL + 5 s `busy_timeout` lets many readers and one writer coexist with no app-level locks. See [`app/storage/db.py`](app/storage/db.py).

Schema in [`app/storage/schema.py`](app/storage/schema.py): three phases on every boot — `CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ADD COLUMN`, `CREATE INDEX IF NOT EXISTS`. Adding a column = one-line `_add_columns` entry.

Tables: `sessions`, `utterances`, `reminders`, `speaker_profiles`, `custom_voices`, `token_usage`, `pending_actions`, `auth_sessions`, `push_subscriptions`, `voice_messages`.

Per-profile `settings.json` and `memory.md` are flat files under `$DATA_DIR_CONTAINER/users/<profile_id>/` (not SQLite) so a user can hand-edit them.

## Configuration

Every env var is documented inline in [`docker-compose.yml`](../docker-compose.yml). Treat that file as the canonical reference for `LLM_URL`, `WHISPER_URL`, `TTS_URL`, `DESKTOP_URL`, `DB_PATH`, `WAKE_WORD_NAME`, `MEMORY_*`, and the rest.
