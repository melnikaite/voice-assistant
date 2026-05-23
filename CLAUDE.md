# Contributor entry point (LLM / human)

You're reading the **navigation map**.  This file does NOT explain the
project — `README.md` does.  Read that first if you're new.

This file's job is to get you to the RIGHT doc in one hop, so you can
answer a question without grepping the whole repo.

## Where to look for X

### Architecture / design questions

| Question | File |
|---|---|
| What is this project, at a glance? | [`README.md`](README.md) |
| How are the services wired together? | [`docs/architecture.md`](docs/architecture.md) — data flow, state machine, decisions |
| How does the agent loop pick tools? | [`docs/architecture.md`](docs/architecture.md) → "Agent loop" |
| How is the read-only stance on desktop-control enforced? | [`docs/architecture.md`](docs/architecture.md) → "Read-only defence" |
| How does multi-agent routing work? | [`desktop-agent/README.md`](desktop-agent/README.md) → "Reverse-WSS mode" |
| Why these specific tech choices (XTTS, mlx-whisper, Gemma)? | [`README.md`](README.md) → Acknowledgements + [`docs/architecture.md`](docs/architecture.md) → Decisions |

### "How do I add X" cookbooks

| Task | File |
|---|---|
| Add a new LLM tool | [`docs/adding-a-tool.md`](docs/adding-a-tool.md) |
| Add a new language to the UI / voice replies | [`docs/adding-a-locale.md`](docs/adding-a-locale.md) |
| Wire a new local LLM provider | [`docs/architecture.md`](docs/architecture.md) → "LLM provider abstraction" |
| Add a desktop-agent backend (e.g. Windows) | [`desktop-agent/desktop-agent.py`](desktop-agent/desktop-agent.py) — `Backend` ABC + existing subclasses |
| Train a custom wake-word | [`xtts-server/README.md`](xtts-server/README.md) — "Training a custom wake-word" |

### Service-specific docs

| Service | Where to start |
|---|---|
| Orchestrator (FastAPI brain) | [`orchestrator/README.md`](orchestrator/README.md), then [`orchestrator/CLAUDE.md`](orchestrator/CLAUDE.md) |
| Desktop-agent (host gateway) | [`desktop-agent/README.md`](desktop-agent/README.md) |
| XTTS-server (host TTS) | [`xtts-server/README.md`](xtts-server/README.md) |
| Frontend (browser PWA) | [`frontend/README.md`](frontend/README.md), then [`frontend/CLAUDE.md`](frontend/CLAUDE.md) |

### Workflow

| Question | File |
|---|---|
| How do I run tests? | [`CONTRIBUTING.md`](CONTRIBUTING.md) → "Running tests" |
| How do I send a PR? | [`CONTRIBUTING.md`](CONTRIBUTING.md) → "PR workflow" |
| Code style / conventions | [`CONTRIBUTING.md`](CONTRIBUTING.md) → "Code style" |

## Mental model in 60 seconds

- **Single brain, many sensors**: the orchestrator is a FastAPI server.
  Everything else (ASR, LLM, TTS, desktop control) is a separate
  process the orchestrator talks to over HTTP.  Each external piece
  can be swapped (different LLM provider, different TTS, …) without
  touching the brain.
- **Agent loop**: every user utterance becomes a chat completion with
  a tool catalog.  The LLM picks 0–1 tools; tool output either is the
  spoken reply (`terminal=True`) or gets fed back so the LLM can
  chain (`terminal=False`).  Loop bound: 4 iterations.
- **Tools are decorator-registered**: drop a file under `orchestrator/app/tools/`,
  decorate with `@tool(...)`, restart — see [`docs/adding-a-tool.md`](docs/adding-a-tool.md).
- **Storage is local SQLite**: thread-local connections, WAL mode,
  schema in `orchestrator/app/storage/schema.py`.  Migrations are
  idempotent CREATE-IF-NOT-EXISTS — bump schema version when adding.
- **Frontend is vanilla JS**: no build step, no framework.  Service
  worker handles wake-word and offline push.  i18n via a tiny custom
  helper (`frontend/i18n.js`).

## House rules

1. **No telemetry, no cloud-by-default**.  Anything that calls out
   over the public internet must (a) be a user-initiated action, and
   (b) degrade gracefully when offline.  Tools follow this — see
   `web_search`, `weather`, `news` for the pattern.
2. **English in code, comments, prompts**.  Locale-specific strings
   live in `orchestrator/app/i18n.py` (`CATALOG[key][lang]` map).  The
   only exception: the user-facing voice / text content the assistant
   SPEAKS — that's i18n-routed.
3. **Read-only by default for `computer_use`**.  Destructive verbs
   are refused at three layers — prompt, static classifier,
   vision-click filter.  Don't add a fourth bypass.
4. **Tests must run offline**.  No HTTP fetch in tests; mock the
   adapter layer.  See `orchestrator/tests/conftest.py` for the
   fixture pattern.
5. **Secrets never in tracked files**.  The only acceptable shape is
   `os.environ.get("VAR_NAME")` with a sensible default, plus a
   `.env.example` documenting the var.
