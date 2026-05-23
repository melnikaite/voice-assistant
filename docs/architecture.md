# Architecture

Deep reference for the voice-assistant codebase. Companion to the
top-level [`README.md`](../README.md).

File:line citations are breadcrumbs accurate as of this commit, not contracts.

---

## 1. Component map

```
                  ┌──────────────────────────────────────┐
                  │ Browser (PWA)                        │
                  │  • openWakeWord (ONNX)               │
                  │  • mic capture + VAD bypass          │
                  │  • WebRTC peer (RTCPeerConnection)   │
                  │  • Service Worker + Web Push         │
                  └──────────────┬───────────────────────┘
                                 │ WS (signalling + JSON)
                                 │ RTC (mic up, TTS down)
                                 ▼
   ┌──────────────────────────────────────────────────────────┐
   │ orchestrator  (Docker, network_mode: host, :8080)        │
   │   FastAPI app                                            │
   │   ├─ ws.py            FSM, WebRTC bridge                 │
   │   ├─ pipeline.py      one turn end-to-end                │
   │   ├─ agent.py         tool-using LLM loop (≤4 steps)     │
   │   ├─ tools/           pkgutil-discovered registry        │
   │   ├─ vision_loop.py   phase-2 fallback for computer_use  │
   │   ├─ desktop_client/  multi-agent registry + reverse-WSS │
   │   └─ storage/         SQLite (WAL, thread-local conns)   │
   └────┬────────┬────────┬────────┬────────────────────────┬─┘
        │ HTTP   │ HTTP   │ HTTP   │ HTTP / reverse-WSS     │
        ▼        ▼        ▼        ▼                        ▼
   ┌──────────┐ ┌──────┐ ┌─────────────┐ ┌──────────────────┐ ┌──────────────┐
   │ mlx-     │ │ LM   │ │ xtts-server │ │ desktop-agent    │ │ DuckDuckGo / │
   │ whisper  │ │Studio│ │  (host,     │ │  (host, :9877)   │ │ Open-Meteo / │
   │ (host,   │ │(host,│ │   :9876)    │ │  AppleScript /   │ │ news (over   │
   │ :18000)  │ │:1234)│ │  XTTS-v2    │ │  pyautogui /     │ │ internet)    │
   │ ASR      │ │ LLM  │ │  streaming  │ │  screenshot      │ └──────────────┘
   └──────────┘ └──────┘ └─────────────┘ └──────────────────┘
```

| Component        | Runs where      | Why there                                                                                                       |
|------------------|-----------------|-----------------------------------------------------------------------------------------------------------------|
| `frontend/`      | Browser         | Wake-word + mic must live where the user is; Service Worker needs `https://` or `localhost`; no build step.     |
| `orchestrator/`  | Docker (host net)| Reproducible build. Host networking required for WebRTC ICE — bridge candidates aren't routable. `docker-compose.yml:1-22`. |
| `xtts-server/`   | Host (uv venv)  | PyTorch needs MPS/CUDA; macOS Docker can't pass through Apple GPU (3-5× slower in container). |
| `mlx-whisper`    | Host            | MLX is Apple-only, needs direct Metal access. Any OpenAI-Whisper-compatible server works (env `WHISPER_URL`). |
| LLM server       | Host            | Default LM Studio. Anything OpenAI-compatible (Ollama, vLLM, llama.cpp, cloud) works — see §8.                  |
| `desktop-agent/` | Host (uv venv)  | AppleScript/Apple Events need GUI session + Accessibility/Automation grants; pyautogui needs display. Not reachable from Docker. |

Orchestrator never talks to the user directly. Browser is the only client; everything else is a backend it composes.

---

## 2. Data flow per turn

Typical wall-clock budgets on an M-series Mac with Gemma 4 E4B local:

- ASR + speaker-ID in parallel (~250 ms). Speaker-ID threshold 0.75 (env: `SPEAKER_THRESHOLD`); same speaker typically 0.80+, different speakers below 0.70.
- Agent loop — LLM tool selection + tool dispatch (~1-3 s; dominant cost).
- TTS first chunk (~300 ms TTFB after the tool returns).
- Vision-loop fallback adds up to 6 × ~10 s per planning step (§6).

Short-circuits skip the agent loop: `intents.py` catches "повтори"/"новая тема" with zero LLM cost (pipeline.py:663), voicemail-leave skips the agent (pipeline.py:449), an attached image goes straight to vision (pipeline.py:619). Streaming text tools (`web_search`) start TTS mid-LLM-generation via `on_response_chunk` (ws.py:405) — time-to-first-audio ~1.5 s instead of 5 s.

---

## 3. Agent loop

`run_agent()` in `orchestrator/app/agent.py:237` drives one user turn until:

1. The model emits free text (no `tool_calls`) — that text is the reply (agent.py:311-326).
2. A tool with `terminal=True` runs — its `text` IS the reply, no follow-up LLM call (agent.py:378-382).
3. `MAX_AGENT_STEPS = 4` exceeded — fall back to last tool's text or a fixed message (agent.py:388-395).

| Tool          | `terminal` | Why                                                                                |
|---------------|------------|------------------------------------------------------------------------------------|
| `general_answer` | `False`  | May signal `data["unknown"] = True` → loop continues, model usually picks `web_search`. |
| `web_search`     | `True`   | Already streamed audio chunk-by-chunk; further LLM call would just paraphrase.     |
| `computer_use`   | `True`   | Stdout (or "Done.") is the natural reply.                                          |
| `reminders`, `weather`, `calculator`, … | `True` | Value tools; one call, one answer.                                |

`_format_tool_message` (agent.py:219-234) injects a hint when `data["unknown"]` is set so Gemma reliably picks a follow-up tool. `general_answer` short-circuits via a forced-tool inner LLM call (`respond_with_confidence`) that returns a structured `(answer, confident)` pair — see `tools/general_answer.py:1-77`. Confidence is structural, never parsed from natural-language hedging.

The 4-step cap fits the realistic chain `general_answer → web_search` plus one hop. Each step is ~1.2 s of LLM time and 200-1000 tokens.

Risk gating: `_execute_one` (agent.py:398-473). A `risk="high_write"` call without an active passphrase auth window gets enqueued into `pending_actions`, loop returns a terminal "deferred" reply — see §4.

---

## 4. Tool registry

Each tool is an async function decorated with `@tool(...)` from `orchestrator/app/tools/base.py:145`. The decorator registers the handler into `TOOL_REGISTRY` at import time.

Auto-discovery in `tools/__init__.py:24-33`:

```python
for _mod_info in pkgutil.iter_modules(__path__):
    if _mod_info.name.startswith("_") or _mod_info.name == "base":
        continue
    importlib.import_module(f"{__name__}.{_mod_info.name}")
```

Drop `orchestrator/app/tools/my_tool.py`, decorate with `@tool(...)`, restart. The decorator inspects the signature; tools that take `ctx: AgentContext` get it injected (`base.py:174-177`). Walk-through: [`docs/adding-a-tool.md`](adding-a-tool.md).

A tool returns `ToolResult(text, data=None, tts_voice_override=None)` (`base.py:60-70`). `text` is what the user hears; `data` is structured payload for logs and UI. `tts_voice_override` is used only by `inbox_summary` to read voicemails in the sender's voice.

### Three risk levels

Declared per tool; gated in `agent._execute_one`:

| Risk          | Gate                                                                                 | Example tools                                              |
|---------------|--------------------------------------------------------------------------------------|------------------------------------------------------------|
| `read`        | Always runs.                                                                         | `weather`, `calculator`, `web_search`, `general_answer`, `look_at_screen`, `computer_use`. |
| `low_write`   | Always runs (reversible / low-stakes writes).                                        | `set_reminder`, music control, "like" actions.             |
| `high_write`  | Requires `ctx.is_authenticated` (passphrase within auth window, default 5 min). Otherwise enqueued in `pending_actions`. | `memory_write`, `update_settings`, "send", "delete", anything destructive. |

`computer_use` is `risk="read"` because it never mutates — its three-layer defence (§5) refuses destructive scripts outright. The `desktop` tool is the explicit mutation path: inspects its arguments and chooses risk per call.

---

## 5. Read-only defence on `computer_use`

`computer_use` is the most powerful tool: free-form goal → AppleScript. Contract is "read or system-state only" — no deletes, no sends, no shell escapes. Three independent gates; each alone would suffice, defence in depth means a creative LLM that bypasses two still hits the third.

After one spoken passphrase, all `high_write` tools run for ~5 min (`AUTH_WINDOW_S` in `pipeline.py`); the timer resets on each successful passphrase.

### Layer 1: generator prompt

System prompt (`tools/computer_use.py:81-126`) forbids destructive verbs and reserves a sentinel:

> NEVER output destructive verbs: delete, remove, empty, send, save,
> do shell script, close, quit, make new, duplicate, move.
> If the goal CANNOT be expressed in a single AppleScript snippet,
> output exactly one word: UNKNOWN

10 few-shot examples (volume / keyboard layout / URL open / app activate / scripted reads) plus two negatives ("delete all photos from 2020" → UNKNOWN). Examples beat prose for Gemma 4 E4B.

### Layer 2: static `_classify_applescript_risk`

`tools/desktop.py:124-158` runs two checks:

1. **Category-strict reject.** When the caller passes `category="computer_use"`, any substring match against `_FORBIDDEN_IN_READONLY` (`desktop.py:108-114`) — `delete`, `remove`, `empty`, `send`, `make new`, `do shell script`, `set read`, `удалить`, `переместить` — returns the `DENIED_READONLY_VIOLATION` sentinel. Short-circuits with `t("computer_use.readonly_violation", lang)` (`computer_use.py:291-303`).
2. **Risk upgrade.** `_DESTRUCTIVE_RE` (`desktop.py:79-97`) — word-bounded matches on `\bdelete\b`, `set (content|body|name|location|...)\b`, `do shell script`, `shut down`, etc. — bumps `declared` to `high_write`. `computer_use` refuses anything above `read` (computer_use.py:304-321).

Substring matching is intentionally conservative: false positives are annoying (deferred to queue), false negatives are dangerous. `_FORBIDDEN_IN_READONLY` includes Cyrillic verbs in case a clever generator switches alphabet.

### Layer 3: vision-loop `_validate_action`

When the generator returns `UNKNOWN` (or M1.3's capability gate skips AppleScript on Linux/Windows), `computer_use` drops into the vision loop (§6). Every planned `click` carries a `target_text` field; `vision_loop._validate_action` runs `vision_primitives.is_destructive_text(target)` (`vision_loop.py:178`) and rewrites the action to `{"type": "fail", "reason": "destructive_click_refused: <target>"}` if the visible label hits the blacklist. A model proposing "Send" or "Empty Trash" cannot — the click never reaches `desktop-agent`.

### Capability gate (M1.3)

`computer_use.py:263-272` reads `desktop_client.has_capability_cached("applescript", ...)`. Linux/Windows agents advertise `applescript: False` in `/v1/capabilities`; cache=NO skips the generator (saves a 1-2 s LLM round-trip) and goes straight to vision-loop fallback. Cache-miss (`None`) falls through optimistically; 30 s health-poll backfills.

---

## 6. Vision-loop fallback (phase 2)

`orchestrator/app/vision_loop.py` takes over when the AppleScript generator returns UNKNOWN or the agent platform lacks AppleScript. Each round: one screenshot → multimodal-LLM-plans-next-action → execute on `desktop-agent`. Up to `MAX_STEPS = 6` rounds (vision_loop.py:70), then fail out.

### Action JSON contract

Planner emits one action per call as strict JSON (no markdown fences). Validated by `_validate_action` (vision_loop.py:154-231):

| Type     | Required fields                     | Notes                                                                 |
|----------|-------------------------------------|-----------------------------------------------------------------------|
| `click`  | `x`, `y` (int), `target_text` (str) | `target_text` is mandatory — destructive blacklist runs against it. Coords bounded `[0, 8192]`. |
| `type`   | `text` (str)                        | Capped at `MAX_TYPE_LEN = 200` (vision_loop.py:88).                   |
| `key`    | `keys` (list of str)                | e.g. `["cmd", "space"]`. Preferred over clicks when both work.        |
| `wait`   | `ms` (int)                          | Capped at 5000 ms.                                                    |
| `scroll` | `x`, `y`, `dy` (int)                | `dy` clamped to `[-2000, 2000]`.                                      |
| `done`   | `result` (str)                      | Terminal success. `result` is the spoken reply.                       |
| `fail`   | `reason` (str)                      | Terminal failure.                                                     |

### Refusals and transport errors

`run_vision_loop` returns `{"ok": bool, "error": <key>?, "result": ..., "steps": [...]}`. Error keys map 1:1 to i18n keys in `computer_use._LOOP_ERROR_I18N` (computer_use.py:383-390):

| Error key            | Meaning                                                                          |
|----------------------|----------------------------------------------------------------------------------|
| `user_active`        | Cursor-activity probe says user is at the keyboard (< 5 s idle). Refuses fast — vision-driven mouse moves fighting the user is the worst failure mode. |
| `screenshot_failed`  | `desktop-agent` returned empty bytes.                                            |
| `plan_unparseable`   | Model returned non-JSON or shape-invalid output.                                 |
| `max_steps`          | Budget exhausted without `done`.                                                 |
| `planner_fail`       | Model emitted `{"type": "fail", ...}` — typically because the path required a destructive click. |
| `transport`          | `desktop_client.DesktopUnavailable` mid-loop.                                    |

Cursor-activity refusal enforced at loop start AND between every step (vision_loop.py:381-393). Threshold 5 s. The "warm=False" case (no cursor data yet on agent boot) fails open so a fresh start doesn't lock out the user forever (`_user_is_active` vision_loop.py:336-352).

---

## 7. Storage model

SQLite at `$DB_PATH` (default `/data/assistant.db` in the container). Single-file, single-host, single-user. Schema in `orchestrator/app/storage/schema.py`, applied idempotently every boot (CREATE TABLE → ALTER TABLE ADD COLUMN → CREATE INDEX) so older DBs walk up to current shape with no manual migration.

### Concurrency

`orchestrator/app/storage/db.py` opens one connection per Python thread (`threading.local`). Each fresh connection enables four PRAGMAs (db.py:80-87):

| PRAGMA              | Why                                                             |
|---------------------|-----------------------------------------------------------------|
| `journal_mode=WAL`  | Multiple concurrent readers + one writer at the engine level.   |
| `synchronous=NORMAL`| Loses at most the last committed txn on hard power loss; big write speedup vs `FULL`. |
| `busy_timeout=5000` | Sleep + retry on SQLITE_BUSY rather than raise; no app-level locks. |
| `foreign_keys=ON`   | Harmless; flips on if/when FK constraints get declared.         |

All SQL runs on `asyncio.to_thread`, so each pooled worker caches one connection for its lifetime. Legacy `_lock` symbol is a no-op context manager (`db.py:117-131`) kept for unmigrated `with _lock:` blocks.

### Tables

Defined in `storage/schema.py:16-272`:

| Table                | Purpose                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| `sessions`           | One row per WebSocket connection. `client_id` is the stable per-browser identifier.       |
| `utterances`         | One row per turn — transcript, tool used, response, ASR/LLM ms, embedding BLOB. Source of semantic memory. |
| `reminders`          | Future-fire reminders; `fired` + `delivered` flags drive the scheduler.                   |
| `pending_actions`    | Queue of `high_write` tool calls deferred for passphrase approval. TTL via `expires_at`.  |
| `voice_messages`     | Voicemail audio + transcripts. Inbound (`to_profile_id`) and outbound by direction.       |
| `speaker_profiles`   | Enrolled household members: name + d-vector centroid + per-speaker TTS voice override.    |
| `custom_voices`      | User-recorded XTTS-cloning reference WAVs.                                                |
| `push_subscriptions` | Web Push subscriptions (one per browser/device).                                          |
| `token_usage`        | Per-LLM-call row — prompt, completion, reasoning tokens, attributed to `tool_name` + `client_id`. Feeds `/api/stats`. |
| `auth_sessions`      | UI cookie-session store (HttpOnly, server-side revocable).                                |

### TTLs

| Table                  | TTL    | Cleanup                                                        |
|------------------------|--------|----------------------------------------------------------------|
| `auth_sessions`        | 5 min  | read-path filter on `expires_at`                               |
| `pending_actions`      | 5 min  | read-path filter on `expires_at`                               |
| `items` (deleted)      | 7 days | scheduled `purge_expired_trash` every 6 h                      |
| `utterances`, `token_usage` | —  | append-only, no GC                                             |

No periodic GC on the expires-at rows — read paths filter so stale rows never surface; at one-user scale the tables don't grow fast enough to need pruning. Sweep helpers (`_sweep_expired_sync` etc.) exist — wire them into `scheduler.add_job` if that changes (`main.py:131-137`).

### When to add a table

- Distinct lifecycle (`pending_actions` expires; `utterances` never does).
- Different access pattern (`token_usage` is append-then-aggregate-by-day; `utterances` is append-then-vector-search-by-recency).
- Different ownership boundary (`voice_messages.to_profile_id` is FK-cascaded; `utterances` is session-scoped).

Don't add one for one-off settings (use `user_files.py` JSON files) or small caches (in-process dicts work — `desktop_client.AgentInfo`).

---

## 8. LLM provider abstraction

`orchestrator/app/llm_utils.py:chat()` is the ONLY place a chat HTTP call leaves the process. Every other module — `agent.py`, vision, tools, `general_answer`'s inner call — goes through it. Abstraction is "OpenAI-compatible `/v1/chat/completions`".

### Endpoint split

`llm_utils.py:13-25`:

| Var                | Purpose                                                                       |
|--------------------|-------------------------------------------------------------------------------|
| `LLM_URL`          | Single-provider default. Required.                                            |
| `LLM_MODEL`        | Model id for single-provider default. Required.                               |
| `LLM_TEXT_URL`     | Text endpoint override (defaults to `LLM_URL`).                               |
| `LLM_TEXT_MODEL`   | Text model override (defaults to `LLM_MODEL`).                                |
| `LLM_VISION_URL`   | Vision endpoint override (defaults to `LLM_URL`).                             |
| `LLM_VISION_MODEL` | Vision model override (defaults to `LLM_MODEL`).                              |

Leave the split commented out and Gemma 4 (multimodal) on LM Studio serves both. Override only when you want different providers for text vs vision. `chat()` accepts `endpoint_url=` and `model=` per-call (`llm_utils.py:146-148`); `vision.py` routes vision calls to `LLM_VISION_URL` automatically. `chat_stream()` is the streaming counterpart, used by `web_search` for sentence-by-sentence TTS (`llm_utils.py:219-319`).

### Compatibility matrix

| Server          | Streaming      | Tool calls | Notes                                                                          |
|-----------------|---------------|-----------|--------------------------------------------------------------------------------|
| LM Studio       | yes           | yes       | Default. Reports `usage` in final SSE chunk via `stream_options.include_usage`. |
| Ollama          | yes           | yes (recent) | `LLM_URL=http://localhost:11434`. Models without tool support fall back to free-text + parse. |
| vLLM            | yes           | yes       | Run an OpenAI-format model (Llama 3.1, Qwen, Mistral).                         |
| llama.cpp       | yes           | model-dependent | `llama-server` binary; same `/v1/...` shape.                              |
| OpenAI cloud    | yes           | yes       | `LLM_URL=https://api.openai.com`. Costs money; useful for dev on a low-power box. |
| Any reverse proxy | passthrough | passthrough | The orchestrator only sees OpenAI shape on the wire.            |

`chat()` records every call to `token_usage`, attributed to `tool_name` so the dashboard shows "X% to web_search, Y% to agent_loop tool selection" (`llm_utils.py:90-131`).

---

## 9. Multi-agent (desktop-agent) routing

The orchestrator can talk to multiple `desktop-agent` daemons — typical setup is home Mac + work PC on the same tailnet. Configured by `DESKTOP_AGENTS` env (`desktop_client.py:179-237`):

```jsonc
DESKTOP_AGENTS: |
  [{"agent_id":"macbook","url":"http://localhost:9877",
    "token":"abc","default":true},
   {"agent_id":"work-pc","url":"http://work.tailnet.ts.net:9877",
    "token":"xyz"}]
```

`DESKTOP_URL` / `DESKTOP_TOKEN` are the single-agent fallback (`docker-compose.yml:167-187`).

### `AgentInfo` registry

One `AgentInfo` per agent (`desktop_client.py:135-167`): `agent_id`, `url`, `token`, `default` flag, `mode` (`http` or `reverse`), capabilities cache + timestamp, reachable flag, last_seen.

Resolution for `get_agent(None)` (`desktop_client.py:250-272`):
1. Agent flagged `default=True`.
2. Most-recently-reachable agent (highest `last_seen`).
3. First in env order.

Tools taking an `agent_id` argument (LLM-visible) route to a named agent; omitting it selects the default. Frontend's "Connected devices" panel reads `/api/agents` (`main.py:997-1036`).

### Capability cache (TTL 60 s, polled every 30 s)

`AgentInfo.capabilities_cache` holds the most recent `/v1/capabilities` response (`desktop_client.py:365-428`). On cold start the first `capabilities()` call pays an HTTP round-trip; subsequent calls within `_CAPS_TTL_S = 60.0` (`desktop_client.py:125`) return the cached value.

`_health_poll_loop` (`desktop_client.py:476-509`) runs every `DESKTOP_HEALTHPOLL_INTERVAL_S = 30` s and refreshes every HTTP-mode agent's caps. Reverse-mode agents are skipped — liveness IS the WSS. Poll keeps the `reachable` flag warm so tools fast-fail with a friendly message instead of a 3 s HTTP connect timeout.

`has_capability_cached()` (`desktop_client.py:437-456`) returns `True` / `False` / `None` (cache miss) — used by `computer_use`'s M1.3 fast-skip (§5).

### Reverse-WSS for NAT'd agents

`orchestrator/app/agent_proxy.py` flips polarity: agent dials orchestrator over WSS (`/v1/agent/connect`) and stays connected. Orchestrator pushes RPC calls down the same socket (`desktop_client._reverse_call`, `desktop_client.py:575-585`).

Wire protocol v1 (newline-delimited JSON, `agent_proxy.py:13-33`):

```
agent → orchestrator on connect:
    {"type":"hello", "agent_id":"...", "token":"...", "capabilities":{...}, "version":"1.1.0"}
orchestrator → agent:
    {"type":"hello_ack", "session_id":"..."}    or  {"type":"reject", "reason":"auth|version|protocol"}
orchestrator → agent (RPC):
    {"type":"call", "call_id":"<uuid>", "method":"screenshot", "params":{...}}
agent → orchestrator (RPC response):
    {"type":"result", "call_id":"<uuid>", "ok":true,  "data":{...}}
    {"type":"result", "call_id":"<uuid>", "ok":false, "error":"..."}
keepalive:
    {"type":"ping"}  ↔  {"type":"pong"}                          (every 30 s)
```

`AgentConnection` (`agent_proxy.py:93-239`) owns one live WSS, a `recv_loop` task, a `_pending: dict[call_id, Future]` map. Caller posts a frame, awaits the matching future; recv loop resolves it. Concurrent calls interleave freely (WSS is full-duplex; `call_id` correlates). Process crash loses in-flight calls — fine (re-issue from LLM). On reconnect the prior `AgentConnection` is evicted (`agent_proxy.py:305-309`) so a flapping link doesn't accumulate zombies.

---

## 10. i18n

`orchestrator/app/i18n.py` owns a single `CATALOG[key][lang] = text` dict plus `t(key, lang, **fmt)`. Code/comments/prompts/module docs are all English so any contributor can read them. What the user *hears or sees* is rendered in their own language.

### Resolution

`pick_lang(settings_lang, detected_lang)` (`i18n.py:311-326`) picks per turn:

1. Speaker's `settings.language` if set to `en|ru|de`.
2. Whisper's detected language for THIS turn (when wired up — currently passes `None`).
3. `DEFAULT_LANG = "en"`.

`AgentContext.user_lang` (set in `pipeline.py:764-773`) carries the resolved value into every tool. Tools call `t(key, ctx.user_lang, **fmt)` (`i18n.py:329-348`) — no literal user-facing strings live in tool source.

Format placeholders are Python `str.format` — keep them named (`{summary}`, not `{0}`) so translations can reorder.

### Adding a locale

1. Add the code to `SUPPORTED_LANGS` (`i18n.py:48`).
2. For each entry in `CATALOG`, add the new key alongside `en/ru/de`. Missing keys fall back to English (`i18n.py:340-341`).
3. Add a Whisper-detected-language → catalog-code mapping if Whisper reports a code we don't recognise.

No registry edits, no per-locale module, no compilation.

---

## 11. Privacy / offline-first

Everything important runs on `localhost`:

| Service          | Local? | Falls back to                                                            |
|------------------|--------|--------------------------------------------------------------------------|
| Wake-word        | yes (browser ONNX) | —                                                              |
| ASR              | yes (mlx-whisper)  | Any OpenAI-Whisper-compatible HTTP endpoint                    |
| LLM (text+vision)| yes (LM Studio default) | Any OpenAI-compatible endpoint                            |
| TTS              | yes (XTTS-v2)      | —                                                              |
| Semantic memory  | yes (fastembed MiniLM-L12) | Downloads once (~220 MB ONNX), then offline            |
| Speaker ID       | yes (resemblyzer)  | —                                                              |
| Storage          | yes (SQLite)       | —                                                              |
| Desktop control  | yes (desktop-agent) | —                                                             |
| Reminders        | yes (in-process scheduler) | —                                                      |

Three tools call out to the internet:

- `web_search` → DuckDuckGo + per-result page fetch (trafilatura)
- `weather` → Open-Meteo
- `news` → DuckDuckGo News + Open-Meteo geolocation

All three degrade to a localised "<Tool> needs internet, we're offline right now" via `net.py` + `i18n.offline_for_tool` (`i18n.py:356-364`). Everything else (LLM turns, memory, reminders, computer control, voicemail between household members) works without internet.

No telemetry. No cloud accounts. No LLM API keys by default. Only local secrets are `data/vapid_private.pem` (Web Push, auto-generated) and `DESKTOP_TOKEN` (auto-generated by `desktop-agent`).

---

## 12. Architectural decisions log

One-line rationale per decision.

- **SQLite over Postgres.** Single-user single-host; no operator burden — accept the limit explicitly. WAL + busy_timeout gets multi-reader / single-writer at near-zero cost.
- **Vanilla JS over React/Vue.** Static files via FastAPI `StaticFiles` (`main.py:1056`). No build step, edit-refresh works without a bundler. CDN-vendored libraries (`frontend/vendor/`) live alongside source so the whole thing works offline.
- **Tool decorator + auto-discovery over manifest.** `@tool(...)` + `pkgutil.iter_modules` (`tools/__init__.py:24-33`). Adding a tool = drop a file, restart. A manifest would need to stay in sync with code anyway.
- **Three-layer read-only defence over single allowlist.** Generator prompt + static substring/regex classifier + vision-loop click filter (§5). Each layer independent; bypassing one doesn't bypass the others. LLM-as-judge classifiers add another round-trip + another LLM to trust.
- **LM Studio default but provider-agnostic.** OpenAI-compatible `/v1/chat/completions` is the lingua franca; the orchestrator only sees that shape. Endpoint URL + model are env knobs.
- **WebRTC for audio over binary WebSocket frames.** RTCPeerConnection both ways. Browser's WebRTC voice engine sees TTS as reference audio and runs AEC3 — speaker→mic feedback is cancelled for free, enabling full-duplex barge-in. WS stays as signalling + JSON-events (`ws.py:1-34`).
- **Host xtts-server over in-container TTS.** PyTorch reaches MPS/CUDA; container can't (`docker-compose.yml:128-143`). 3-5× faster inference, RTF ~0.3 vs ~4.
- **Reverse-WSS over orchestrator-initiated VPN.** Agent dials orchestrator (`agent_proxy.py`, §9). One env flip and the home agent serves a phone-tethered orchestrator across a coffee-shop NAT. Tailscale is still recommended for the static case (`desktop-agent/README.md:183-189`).
- **Voice ID as auth, passphrase as escalation.** Speaker recognition (resemblyzer d-vectors) identifies WHO without a credential exchange — enough for `read` and `low_write`. `high_write` requires the passphrase to open a 5-minute auth window (`pipeline.py:60`). Passphrase-on-every-command defeats the "voice is the UI" point; voice-ID-only fails on guests.

---

## Where to look next

- [Adding a tool](adding-a-tool.md)
- [`orchestrator/CLAUDE.md`](../orchestrator/CLAUDE.md) — contributor map inside the orchestrator service.
- [`desktop-agent/README.md`](../desktop-agent/README.md) — bootstrap, per-OS capabilities, autostart, reverse-WSS.
- [`xtts-server/README.md`](../xtts-server/README.md) — TTS service bootstrap, model location, wake-word retraining.
