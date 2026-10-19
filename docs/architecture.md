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
   │ orchestrator  (host, uv venv, launchd, :8080)            │
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
   ┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐ ┌──────────────┐
   │ LocalAI         │ │ xtts-server │ │ desktop-agent    │ │ DuckDuckGo / │
   │ (host, :1240)   │ │  (host,     │ │  (host, :9877)   │ │ Open-Meteo / │
   │ whisper.cpp ASR │ │   :9876)    │ │  AppleScript /   │ │ news (over   │
   │ + llama.cpp LLM │ │  XTTS-v2    │ │  pyautogui /     │ │ internet)    │
   │ (text+vision)   │ │  streaming  │ │  screenshot      │ └──────────────┘
   └─────────────────┘ └─────────────┘ └──────────────────┘
```

| Component        | Runs where      | Why there                                                                                                       |
|------------------|-----------------|-----------------------------------------------------------------------------------------------------------------|
| `frontend/`      | Browser         | Wake-word + mic must live where the user is; Service Worker needs `https://` or `localhost`; no build step.     |
| `orchestrator/`  | Host (uv venv)  | Native uv service since 2026-06-12 (`orchestrator/start.sh`, launchd). WebRTC ICE binds host interfaces directly — no Docker port-proxy layer. `docker-compose.yml` remains the optional Linux-server path. |
| `xtts-server/`   | Host (uv venv)  | PyTorch needs MPS/CUDA; macOS Docker can't pass through Apple GPU (3-5× slower in container). |
| ASR server       | Host            | Default LocalAI (whisper.cpp Metal/CUDA, same server as the LLM). Any OpenAI-Whisper-compatible server works (env `WHISPER_URL` — e.g. mlx-whisper, faster-whisper). |
| LLM server       | Host            | Default LocalAI :1240 (llama.cpp Metal/CUDA; also serves whisper.cpp ASR). Anything OpenAI-compatible (Ollama, vLLM, llama.cpp, LM Studio, cloud) works — see §8. |
| `desktop-agent/` | Host (uv venv)  | AppleScript/Apple Events need GUI session + Accessibility/Automation grants; pyautogui needs display. Not reachable from Docker. |

Orchestrator never talks to the user directly. Browser is the only client; everything else is a backend it composes.

---

## 2. Data flow per turn

Typical wall-clock budgets on an M-series Mac with a local 4–8B model (Qwen3-VL-8B by default):

- ASR + speaker-ID in parallel (~250 ms; the speaker embedding itself is ~14 ms for a 2 s utterance on CPU). Speaker-ID threshold 0.55 (env: `SPEAKER_THRESHOLD`) — see [Speaker ID](#speaker-id).
- Agent loop — LLM tool selection + tool dispatch (~1-3 s; dominant cost).
- TTS first chunk (~300 ms TTFB after the tool returns).
- Vision-loop fallback adds up to 6 × ~10 s per planning step (§6).

Short-circuits skip the agent loop: `intents.py` catches "повтори"/"новая тема" with zero LLM cost (pipeline.py:663), voicemail-leave skips the agent (pipeline.py:449), an attached image goes straight to vision (pipeline.py:619). Streaming text tools (`web_search`) start TTS mid-LLM-generation via `on_response_chunk` (ws.py:405) — time-to-first-audio ~1.5 s instead of 5 s.

---

### Speaker ID

[`app/speaker.py`](../orchestrator/app/speaker.py) embeds each utterance with **WeSpeaker CAM++** (VoxCeleb-trained, 512-dim, ONNX on onnxruntime, ~14 ms median for a 2 s clip on Apple-silicon CPU). Front-end: 80-bin Kaldi fbank (25/10 ms, no dither, `kaldi-native-fbank`) on int16-scaled PCM, then per-utterance mean normalisation — identical to the WeSpeaker reference pipeline (cosine 1.0 against torchaudio's fbank). Model file and download: `docs/deployment.md` § 9b. It replaced resemblyzer (GE2E): on public LibriSpeech/MLS/OpenSLR crops (en/de/ru, 80 speakers) CAM++ EER is 0.3–0.7 % at 4 s and 1.7–3.0 % at 1.8 s, against 1.7–2.1 % / 3.4–6.7 %.

**Threshold `SPEAKER_THRESHOLD=0.55`.** CAM++ scores live on a lower scale than resemblyzer's (its EER thresholds are 0.34–0.51), so the old 0.75 would reject most genuine matches. Matching the wrong profile is the costly error (another person's memory, devices, voicemail); a miss only falls back to the guest path. 0.55 on the bench:

| trial                          | FAR     | FRR    |
|--------------------------------|---------|--------|
| 4 s enrollment vs 1.8 s query  | 0.04 %  | 8.7 %  |
| 4 s vs 4 s                     | 0.10 %  | 1.7 %  |
| 1.8 s vs 1.8 s                 | 0.01 %  | 24 %   |

0.50 halves the short-query FRR (4.9 %) at ~4× the FAR (0.15 %); 0.60 pushes FAR to 0.01 % at 16 % FRR. These are strangers on read speech — family members on one microphone are closer, so re-check on real voices before changing it: each member enrolls one 5 s sample in a quiet room, then speaks ~10 ordinary commands of 1.5–3 s from the usual positions; the log line `speaker=… score=…` gives genuine scores, and commands from member B scored against A's profile (or a guest) give impostor scores. The threshold should sit below nearly all genuine scores and clearly above the highest impostor score.

**Model compatibility.** `speaker_profiles.embedding_model` records which model produced each centroid (`wespeaker-campplus-voxceleb`; NULL = resemblyzer, enrolled before the column existed). Embeddings from different models are different vector spaces and no raw audio is stored, so old profiles cannot be converted: speaker ID skips them, `GET /api/speakers` returns `needs_reenroll: true`, and the Voice profiles panel asks for a new recording. Enrolling under the same name then replaces the old embedding (sample count restarts at 1); further recordings with the same model are running-mean averaged as before.

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

### Step-up gate (`private=True`, #55)

Orthogonal to the risk levels above: a tool flagged `@tool(..., private=True)` reads data sensitive enough to need a **second factor that isn't a spoken word** (so a bystander can't shoulder-surf it). `my_history` is the first such tool.

When a `private` tool is called without an active grant, `agent._execute_one`:
1. Mints a single-use token in `step_up_grants` bound to `(profile_id, client_id)`.
2. Web-Pushes a "tap to confirm" notification to the speaker's devices (token rides the encrypted payload — never a URL).
3. Returns "tap your device to confirm, then ask again" and ends the turn.

The user taps → the service worker POSTs the token to `/api/step-up/approve` → the orchestrator consumes the grant and elevates **only** the originating session (verified by `client_id` + `profile_id`) → a `step_up_granted` WS event opens a 5-min window so the re-asked turn carries `ctx.step_up_auth=True`. Distinct from the passphrase window (`high_write`): step-up proves *device possession*, not a *shared secret*.

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

Defined in `storage/schema.py`:

| Table                  | Purpose                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------|
| `sessions`             | One row per WebSocket connection. `client_id` is the stable per-browser identifier.       |
| `utterances`           | One row per turn — transcript, tool used, response, ASR/LLM ms, embedding BLOB. Source of semantic memory + the per-tool latency/error figures in `/api/stats`. |
| `reminders`            | Future-fire reminders; `fired` + `delivered` flags drive the scheduler.                   |
| `pending_actions`      | Queue of `high_write` tool calls deferred for passphrase approval. TTL via `expires_at`.  |
| `voice_messages`       | Voicemail audio + transcripts. Inbound (`to_profile_id`) and outbound by direction.       |
| `speaker_profiles`     | Enrolled household members: name + speaker-embedding centroid + the model that produced it (`embedding_model`) + per-speaker TTS voice override. |
| `custom_voices`        | User-recorded XTTS-cloning reference WAVs.                                                |
| `push_subscriptions`   | Web Push subscriptions (one per browser/device).                                          |
| `token_usage`          | Per-LLM-call row — prompt, completion, reasoning tokens, attributed to `tool_name` + `client_id`. Feeds `/api/stats`. |
| `auth_sessions`        | UI cookie-session store (HttpOnly, server-side revocable).                                |
| `items`, `categories`, `category_shares` | Personal item store: links/notes/media + hierarchical folders + household sharing grants. |
| `datasets`, `dataset_entries` | Structured JSON datasets owned by a category, one JSON document per entry — see [§14](#14-structured-datasets). |
| `profile_devices`      | (#50) Maps a `speaker_profile` → the desktop-agent (`device_uid`) that runs its device-tier tools; optional WoL MAC/IP + `is_default`. |
| `device_pairing_codes` | (#50) Ephemeral single-use 6-digit codes (CSPRNG) binding an agent to a profile over reverse-WSS. 5-min TTL. |
| `telegram_links`       | Telegram account → `speaker_profile` (UNIQUE `telegram_user_id`, plus `chat_id`). One account per profile; re-linking replaces. |
| `telegram_link_tokens` | Single-use CSPRNG link tokens minted in Settings for the logged-in profile. 5-min TTL. |
| `step_up_grants`       | (#55) Push-to-device approval tokens for `private=True` tools. Single-use, 90-s TTL, stores `(profile_id, client_id)` so only the requesting session is elevated. |

### TTLs

| Table                  | TTL    | Cleanup                                                        |
|------------------------|--------|----------------------------------------------------------------|
| `auth_sessions`        | 5 min  | read-path filter on `expires_at`                               |
| `pending_actions`      | 5 min  | read-path filter on `expires_at`                               |
| `device_pairing_codes` | 5 min  | read-path filter + purge-on-create                            |
| `telegram_link_tokens` | 5 min  | read-path filter + purge-on-create                            |
| `step_up_grants`       | 90 s   | single-use (deleted on consume) + `purge_step_up_grants`      |
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

Leave the split commented out and the one multimodal model on LocalAI (Qwen3-VL-8B by default) serves both. Override only when you want different providers for text vs vision. `chat()` accepts `endpoint_url=` and `model=` per-call (`llm_utils.py:146-148`); `vision.py` routes vision calls to `LLM_VISION_URL` automatically. `chat_stream()` is the streaming counterpart, used by `web_search` for sentence-by-sentence TTS (`llm_utils.py:219-319`).

### Compatibility matrix

| Server          | Streaming      | Tool calls | Notes                                                                          |
|-----------------|---------------|-----------|--------------------------------------------------------------------------------|
| LocalAI         | yes           | yes       | Default (:1240, llama.cpp + whisper.cpp backends). Set `LLM_REASONING_EFFORT=none` — Gemma's adaptive thinking otherwise fires sporadically (seconds of CoT before a voice reply). |
| LM Studio       | yes           | yes       | Works (it's llama.cpp underneath). Reports `usage` in final SSE chunk via `stream_options.include_usage`. |
| Ollama          | yes           | yes (recent) | `LLM_URL=http://localhost:11434`. Models without tool support fall back to free-text + parse. |
| vLLM            | yes           | yes       | Run an OpenAI-format model (Llama 3.1, Qwen, Mistral).                         |
| llama.cpp       | yes           | model-dependent | `llama-server` binary; same `/v1/...` shape.                              |
| OpenAI cloud    | yes           | yes       | `LLM_URL=https://api.openai.com`. Costs money; useful for dev on a low-power box. |
| Any reverse proxy | passthrough | passthrough | The orchestrator only sees OpenAI shape on the wire.            |

`chat()` records every call to `token_usage`, attributed to `tool_name` so the dashboard shows "X% to web_search, Y% to agent_loop tool selection" (`llm_utils.py:90-131`).

### Models evaluated

Everything below was measured on one M-series Mac / 64 GB, through LocalAI on `:1240`, with the
production `SYSTEM_PROMPT` and the production tool catalog (`_schemas_with_clock`).  Numbers are
medians of three runs unless noted.  Read this before trying "a better model" — most of the
plausible candidates have already been eliminated, and for reasons that are not obvious from
their benchmark cards.

#### Usable

| | **Gemma 4 E4B QAT q4_0** (removed 2026-10) | **Qwen3-VL-8B-Instruct Q4_K_M** (default) |
|---|---|---|
| Install | `local-ai models install …` (already present) | `local-ai models install qwen3-vl-8b-instruct` |
| Backend | llama-cpp + mmproj | llama-cpp + mmproj (same one) |
| Licence | Gemma Terms | Apache-2.0 |
| Recommended context | **131072** — costs almost nothing, see below | **32768** |
| Resident at that context | **8.3 GB** | **11.1 GB** |
| Tool selection (34 tools) | 19/20 | **20/20** |
| Tool arguments | 7/7 | 7/7 |
| End-to-end `run_agent` | 6/7 | 6/7 |
| Screen-text recognition | 6/8 (misread "jarvis" as "jarlis") | **8/8** |
| Click targeting | 152 px mean | **5.3 px mean** |
| Voice TTFT (no catalog) | 0.75 s | 0.74 s |
| Tool-selection turn | **1.1–1.2 s** | 1.5–1.6 s |
| Turn that generates prose | **8–18 s** | 11–29 s |
| Obeys "never free text" | no — answers chatty turns directly | **yes** |

Qwen3-VL is the only model, the default in `orchestrator/start.sh`, for being better at every
accuracy axis measured, decisively so on anything involving the screen.  Gemma 4 E4B was the
fallback until 2026-10 and was removed from LocalAI once nothing else on the host used it; it is
1.5–2x faster on turns that generate speech with a nearly free long context, so it is the first
thing to re-measure if latency matters more than the screen.

#### Context sizing — why the two numbers differ so much

Per-token KV cost is an architectural property, not a tuning knob:

* Gemma 4 E4B — 42 layers, **2** KV heads, and **35 of 42 layers are sliding-attention with a 512-token window**.  ≈14 KB/token, and most layers stop growing entirely.  128K context costs ~1.9 GB.
* Qwen3-VL-8B — 36 layers, all full attention, 8 KV heads, head_dim 128.  ≈**144 KB/token**, 10x more.  Measured: 10.4 GB @8192, 11.1 @32768, 14.9 @65536, 23.9 @131072.  Its 262144 ceiling would need ~38 GB of KV alone — not usable here.

What the app actually needs: tool catalog ≈8.5K tokens, `browser_read` caps page text at
`PAGE_TEXT_LIMIT = 8_000` chars (≈2K tokens), `web_search` bodies at 300 chars, history at
`MAX_HISTORY_TURNS = 10` (20 messages), one screenshot ≈1.3K tokens.  **Worst realistic turn is
12–14K tokens**, so 32768 is already 2x headroom.  Window size does **not** affect speed —
llama.cpp allocates KV up front and attends only over real tokens (measured: 1.49–1.61 s at
32768 vs 1.47–1.79 s at 65536).  It is purely a memory dial.

#### Rejected, with the reason

| Model | Why not |
|---|---|
| [Mage-VL](https://huggingface.co/microsoft/Mage-VL) 8-bit (MS, 5B, Codec-ViT + Qwen3-4B) | 30 px pointing — 5x worse than Qwen3-VL. No GGUF, so it needs the `mlx-vlm` backend **plus** an in-place `pip install mlx-vlm==0.6.9` into `~/.localai/backends/metal-mlx-vlm/venv` (the backend pins 0.4.4, which has no `mage_vl`); any backend reinstall silently reverts that and the model stops loading. Lightest of the lot (5.4 GB) and fastest (4.5 s), which is its only argument. |
| [Holo-3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B) Q8_0 | Pointing is fine (6.2 px) and it is cheap (7.7 GB), but it **never reports a missing control** — 0 of 3 — it substitutes the nearest plausible one. Unusable in a closed loop that must know when it cannot see the target. |
| [Holo-3.1-0.8B](https://huggingface.co/Hcompany/Holo-3.1-0.8B) Q8_0 | The tempting "tiny grounding specialist at 2.2 GB". Pointing 55 px, and worse, it **mixes pixel and normalized coordinate spaces between calls**, so it cannot be calibrated. Reads text acceptably (6/8) but its errors are spatial-association ones — exactly the wrong trade for a grounding role. |
| Holo-3.1-35B-A3B | 22.2 GB of weights (MoE: all experts must be resident, only compute is 3B) for a non-primary role. Not evaluated — the cost ruled it out first. |
| Gemma 4 12B QAT q4_0 (2026-10) | Ties Qwen3-VL on tool use (20/20/20 selection ru/de/en, 21/21 arguments, 21/21 end-to-end), reads the screen (8/8) and reports missing controls (3/3) at 9.6 GB — but clicks land **32 px** off against 3 px. Worse, its sliding-window attention makes llama.cpp re-process the whole ~8.5K-token tool catalog every turn unless the server runs with `--swa-full`, which LocalAI does not pass: 50–90 s per turn under load, past the 90 s agent timeout. |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) Q4_K_M (2026-10) | No official GGUF; converted with llama.cpp's converter, vision projector included, so it does take images. Lighter (7.9 GB) and clicks well (~5 px), but regresses on tool use: 19/19/19 selection, and it **translates in free text instead of calling `translate`** in every run (18/21 arguments); 2/3 on missing controls. |
| Qwen3-VL-8B **Q8_0** | Deliberately not evaluated: its only plausible benefit was fixing the timer-argument failure, and that turned out to be a schema problem (below), so no outcome of the test would have changed the decision. |
| Not installed: Qwen3-VL 2B/4B/30B-A3B/32B and InternVL3.5 14B/30B-A3B (both in the LocalAI gallery), UI-TARS-2B-SFT, ShowUI-2B, OS-Atlas, `locate-anything-3b` | Left on the shelf once Qwen3-VL-8B cleared every bar. Listed so the search does not start from scratch. |

#### Findings that outlived the model choice

1. **Coordinates come back normalized to [0,1000], not pixels** — in every model tried, and telling them "the image is 1440x900" does not change it. Multiply by `W/1000`, `H/1000`. Read as pixels the same replies look 100–390 px wrong, which reads as a bad model and isn't.
2. **A model may decline to emit optional fields at all.** Qwen3-VL called the old multiplexed `reminders` tool with `action`+`trigger` and nothing else, because everything else was optional — 0/6 regardless of field name, type, format, property order, `if/then` schemas, or how loudly the prose demanded it. Putting the field in `required` fixed it outright (6/6). Grammar-constrained calling also fixes it but is not needed. This is why scheduling is now four tools — see `set_reminder.py`.
3. **Spoken durations belong in a fixed machine format, not as computed seconds.** `set_timer` takes ISO-8601 (`PT1H30M`); transcription is reliable where arithmetic is not, and it is what Alexa/Lex normalise spoken durations to. Both models: 6/6 with ISO-8601, including "два с половиной часа" → `PT2H30M`.
4. **Refusal is a model property, not a prompt property**, and single samples mislead — one early run had Mage-VL inventing a click, the opposite of its stable 3/3 behaviour. Repeat before believing any of this.
5. **§6's `_validate_action` catches destructive verbs, not confident nonsense.** A not-visible path has to be added whichever model is chosen, and can only be delegated to the model when that model actually refuses.
6. **Topically varied history breaks referential follow-ups in BOTH models.** "А какая там погода?" after 8 *identical* filler turns resolves fine; after 8 *varied* ones both models answer in free text and call no tool. Depth alone is not the trigger. Model-independent; partly fixed by conversation focus — see §13 and `scripts/context_probe.py`.
7. **`web_search` replies leak the source language** — asked in Russian, the assistant answered in German and English, against `SYSTEM_PROMPT`'s "reply in the user's language". Also model-independent.
8. The `general_answer(unknown=true)` → `web_search` chain **never fired** in end-to-end runs: both models route straight to `web_search`. The chaining machinery is correct but effectively unexercised.
9. **Gemma 4 thinks by default on a plain llama-server** and returns the reasoning as the answer, which gets cut at the token limit: both Gemmas scored 0–2/8 on screen text until the server ran with `--reasoning off`. Check this before blaming a Gemma for bad answers outside LocalAI.

The harness is [`scripts/llm_bench/`](../orchestrator/scripts/llm_bench/bench_llm.py): a synthetic
screenshot with eight controls at known pixel centres, the production prompt and catalog, three
repeats, and — for anything about the agent loop — `run_agent` itself with tool dispatch stubbed,
rather than a bare `/v1/chat/completions` call. Several conclusions in this section reversed when
the probe moved closer to production shape. Re-run on 2026-10 (llama.cpp b11719, each model on its
own server at 32K), the two usable models reproduced: Qwen3-VL 20/19/20 selection, 21/21 arguments,
8/8 screen text, 3.0 px clicks, 3/3 missing controls; Gemma 4 E4B 20/20/20, 21/21, 7/8, 37 px, 0/3.
Latency was not re-measured validly (the runs overlapped another load), so the timing rows above
stand.

### TTS evaluated

Measured 2026-10 on the same Mac: 10 sentences per language with names, numbers and dates (raw
digits, as the orchestrator sends them), WER via parakeet with digits expanded on both sides,
speaker similarity by ECAPA (CAM++ ranks them the same) against one 8 s reference clip of XTTS's
built-in voice — which flatters XTTS on English, so read the ru/de columns.

| | WER ru / de / en | Similarity ru / de | First audio | RTF | Peak memory |
|---|---|---|---|---|---|
| **XTTS-v2** (`xtts-server`, streaming) — kept | **7.1 / 8.0 / 9.5 %** | 0.56 / 0.61 | **1.5 s** | 1.6 | 2.9 GB |
| Chatterbox Multilingual, mlx-audio | 29.9 / 7.7 / 9.6 % | 0.61 / 0.58 | 2.5 s | 0.8 | 4.7 GB |
| Chatterbox, official PyTorch on MPS | 37 / 53 / 11 % | 0.65 / 0.78 | 22–31 s (no streaming) | 4.6 | 14 GB |
| Fun-CosyVoice3 0.5B, official, CPU only | 6.1 / 9.4 / 7.9 % | 0.59 / 0.60 | 24–28 s | 5–6 | 5.9–7.6 GB |
| Fun-CosyVoice3, mlx-audio-plus fork | 14 / 55 / 50 % | 0.48 / 0.47 | 3–7 s | 0.8 | 4.6 GB |

XTTS stays: it is the only one with sub-2 s first audio and good Russian. Chatterbox reads Russian
digits in English ("вантайон") and the MPS build appends babble; CosyVoice3 matches XTTS on accuracy
and cloning but has no MPS path and is ~15x slower here, and needs its cross-lingual mode for ru/de
(zero-shot with an English reference transcript carries the accent: 47 / 36 % WER). The one mlx
port of it produces broken audio.

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

Tools taking an `agent_id` argument (LLM-visible) route to a named agent; omitting it selects the default. Frontend's "Connected devices" panel reads `/api/agents`.

### Personal-device routing + Wake-on-LAN (#50)

For an *identified speaker*, device-tier tools shouldn't hit the default agent — they should hit **that person's own machine**. `device_router.resolve_agent_id_for_profile(profile_id, device_kind)` (`device_router.py`) looks up `profile_devices`, returns the paired `device_uid`, and `agent._execute_one` injects it as `agent_id` when the LLM didn't name one. No pairing → a friendly "no personal device" reply. Pairing happens over reverse-WSS with an ephemeral CSPRNG `device_pairing_codes` entry (no Settings UI round-trip).

If the paired agent is stale (`last_seen` older than `ONLINE_TIMEOUT_S`) and a `wol_mac` is configured, the router fires a Wake-on-LAN magic packet (`send_wol`) and returns the `agent_id` immediately — the tool call wakes the box or fails with a transport error. WoL targets are validated as **LAN/broadcast only** (`ipaddress.is_global` rejected) so a stored `wol_target_ip` can't become an arbitrary-host UDP egress primitive.

Lock-aware: tools declaring `locked_compat="needs_unlock"` are refused when the agent reports `locked=True`; the default `"always"` runs regardless (AppleScript/COM/D-Bus don't need an active UI session).

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
| ASR              | yes (LocalAI whisper.cpp) | Any OpenAI-Whisper-compatible HTTP endpoint             |
| LLM (text+vision)| yes (LocalAI default) | Any OpenAI-compatible endpoint                              |
| TTS              | yes (XTTS-v2)      | —                                                              |
| Semantic memory  | yes (fastembed MiniLM-L12) | Downloads once (~220 MB ONNX), then offline            |
| Speaker ID       | yes (CAM++ ONNX)   | —                                                              |
| Storage          | yes (SQLite)       | —                                                              |
| Desktop control  | yes (desktop-agent) | —                                                             |
| Reminders        | yes (in-process scheduler) | —                                                      |

Three tools call out to the internet:

- `web_search` → DuckDuckGo + per-result page fetch (trafilatura)
- `weather` → Open-Meteo
- `news` → DuckDuckGo News + Open-Meteo geolocation

All three degrade to a localised "<Tool> needs internet, we're offline right now" via `net.py` + `i18n.offline_for_tool` (`i18n.py:356-364`). Everything else (LLM turns, memory, reminders, computer control, voicemail between household members) works without internet.

The optional Telegram channel (`orchestrator/app/telegram.py`) also talks to the internet (`api.telegram.org`, long polling). It is off unless `TELEGRAM_BOT_TOKEN` is set, and polling failures back off and retry without affecting the rest of the orchestrator.

No telemetry. No cloud accounts. No LLM API keys by default. Only local secrets are `data/vapid_private.pem` (Web Push, auto-generated) and `DESKTOP_TOKEN` (auto-generated by `desktop-agent`).

---

## 12. Architectural decisions log

One-line rationale per decision.

- **SQLite over Postgres.** Single-user single-host; no operator burden — accept the limit explicitly. WAL + busy_timeout gets multi-reader / single-writer at near-zero cost.
- **Vanilla JS over React/Vue.** Static files via FastAPI `StaticFiles` (`main.py:1056`). No build step, edit-refresh works without a bundler. CDN-vendored libraries (`frontend/vendor/`) live alongside source so the whole thing works offline.
- **Tool decorator + auto-discovery over manifest.** `@tool(...)` + `pkgutil.iter_modules` (`tools/__init__.py:24-33`). Adding a tool = drop a file, restart. A manifest would need to stay in sync with code anyway.
- **Three-layer read-only defence over single allowlist.** Generator prompt + static substring/regex classifier + vision-loop click filter (§5). Each layer independent; bypassing one doesn't bypass the others. LLM-as-judge classifiers add another round-trip + another LLM to trust.
- **Orchestrator native via uv, Docker demoted to the optional Linux path (2026-06-12).** The original rationale for Docker (reproducible build, host networking for WebRTC ICE) aged out: `uv.lock` gives reproducibility, and on macOS "host networking" is actually an OrbStack/Docker-Desktop proxy layer in front of every port — one more failure mode (helper processes own the sockets) for zero benefit. Native = WebRTC binds real interfaces, one service model for the whole stack (uv venv + LaunchAgent), tests run with plain `uv run pytest`, and the install story needs no Docker at all. `docker-compose.yml` stays for headless Linux servers.
- **LocalAI default (was LM Studio) but provider-agnostic.** OpenAI-compatible `/v1/chat/completions` is the lingua franca; the orchestrator only sees that shape. Endpoint URL + model are env knobs. LocalAI won the default slot 2026-06-12 because one brew-installable server covers LLM + ASR on Metal AND the Linux/CUDA distribution path (spike: `docs/spikes/2026-06-12-localai-spike.md`).
- **WebRTC for audio over binary WebSocket frames.** RTCPeerConnection both ways. Browser's WebRTC voice engine sees TTS as reference audio and runs AEC3 — speaker→mic feedback is cancelled for free, enabling full-duplex barge-in. WS stays as signalling + JSON-events (`ws.py:1-34`).
- **Host xtts-server over in-container TTS.** PyTorch reaches MPS/CUDA; container can't (`docker-compose.yml:128-143`). 3-5× faster inference, RTF ~0.3 vs ~4.
- **Reverse-WSS over orchestrator-initiated VPN.** Agent dials orchestrator (`agent_proxy.py`, §9). One env flip and the home agent serves a phone-tethered orchestrator across a coffee-shop NAT. Tailscale is still recommended for the static case (`desktop-agent/README.md:183-189`).
- **Voice ID as auth, passphrase as escalation.** Speaker recognition (CAM++ embeddings) identifies WHO without a credential exchange — enough for `read` and `low_write`. `high_write` requires the passphrase to open a 5-minute auth window (`pipeline.py:60`). Passphrase-on-every-command defeats the "voice is the UI" point; voice-ID-only fails on guests.
- **Step-up via push-to-device, not a spoken OTP (#55).** Reading private data (`my_history`) needs a second factor a bystander can't overhear. A spoken/heard OTP leaks to anyone in the room; instead we Web-Push a "tap to confirm" to the speaker's own device. Proves *possession of the phone*, not a shared secret. Token travels only in the encrypted push payload + POST body — never a URL — and elevates only the requesting session (§4 step-up gate).
- **Outer Basic Auth installed at app construction, not in lifespan.** Starlette builds its middleware stack on the first request, which is *after* lifespan startup — so `add_middleware` from inside lifespan silently never wraps requests (or raises). The anti-scanner "door before the house" (#43) is therefore added at module-construction time, reading instance settings synchronously (it already requires a restart to toggle).
- **Observability from existing rows, no new telemetry pipeline (#46).** `/api/stats` aggregates `token_usage` (cost/tokens) and `utterances` (per-tool latency + error rate) plus a live in-process snapshot (sessions, agents, uptime). No metrics daemon, no external sink — consistent with offline-first; the dashboard is owner-gated behind the `va_session` cookie.
- **Mixed-utterance attribution via windowed-partials clustering, not diarization (#59).** For clips with at least four 1.6 s windows (0.8 s hop, ≈ 4 s) the encoder also embeds each window in one batched call; clustering those windows (seeded 2-means, min-pairwise-cosine homogeneity gate `SPEAKER_MIXED_HOMOGENEITY=0.20`, with blip / same-profile / centroid-proximity guards) detects "two voices in one VAD segment" at zero new dependencies. On 8 s bench clips: same-voice min-pairwise median ≈ 0.49, two-voice ≈ 0.02 (p99 0.25); at 0.20, 3.8 % of one-voice clips over-fire and 5.4 % of two-voice clips are missed — the guards absorb over-fires. Mixed turns carry no `profile_id`: the cookie fallback is muted, the passphrase auth window is NOT inherited, memory context goes household-wide, and device-tier tools ask whose device instead of guessing (`clarify.which_device`); the existing continuation window catches the answer as a fresh single-speaker turn with its own clean speaker-ID. Real diarization (LocalAI `/v1/audio/diarization`) is Linux-only today — it slots in behind the same `_resolve_speaker` seam later (see `docs/spikes/2026-06-12-localai-spike.md`).

---

## 13. Known defects (TODO)

Reproduced and characterised, not yet fixed.  Both are **model-independent** — they were
measured on Gemma 4 E4B and Qwen3-VL-8B alike, so swapping models will not clear them and
neither should be used as an argument for or against a model.

### Partly fixed: referential follow-ups after topically varied history

`"А какая там погода?"` after a turn about Munich resolves with a shallow or repetitive history
but fails after 8+ topically varied turns: the model answers in free text or picks a recent
filler subject.  Root cause: history kept only text, so the entity a follow-up needs lived in
tool arguments the model never saw again.  A second symptom on the same path: "добавь туда
молоко" was answered "Добавил" with no tool call at all.

Fix: `app/conversation.py` — a focus line built from previous tool calls (`@tool(focus=…)`,
`data["focus"]`) placed right before the question, plus the last `CONVERSATION_KEEP_TURNS`
turns verbatim and a background list of older subjects.  The "list" slot falls back to the
profile's last-used category (`speaker_profiles.last_category_id`).  Measured with
`orchestrator/scripts/context_probe.py` (qwen3-vl-8b, varied filler, N=4/8/16 × 3): 9/36 →
21/36, claimed-without-tool replies 17 → 1.  Query rewriting, replaying tool calls and a prose
summary were measured and rejected (no gain or +2–4 s on the voice path).

The reminder follow-up ("напомни об этом за час") needed two more things: `set_timer` no
longer fills the `reminder` slot (a fresh "таймер" label displaced the listed appointment),
and `list_reminders` puts the absolute fire time into the focus value so "an hour before"
is computable — qwen 0/9 → 6/9, fire time checked by the probe.  The population case
dropped at N=8 (6/6 → 0/6) for the same reason: the stale "таймер" slot was the only thing
rendering the focus line, and that instruction alone made the model resolve "там".  Subjects
from `general_answer` have no slot, and at N=8 the filler's last place is Ottawa, so the
case was passing by accident.

Still open: focus expires after `FOCUS_TTL_TURNS`, so at N=16 subjects survive only via the
summary and are mostly lost.  Raising the TTL to 20 was measured and rejected: shopping_list
gained at N=16 (0→3/3) but the `stale_place` case lost N=16 (3→0/3), so the TTL stays 10.

Still open: a stale slot hijacks a pronoun whose referent is the previous turn.  The
`stale_place` probe case (weather for Munich, fillers, then a `general_answer` turn about
Lisbon, then "А какая там погода?") passes only at N=16, where the Munich slot has expired;
at N=4/8 weather is called for Munich or answered in free text (0/6).  Three fixes were
measured (qwen3-vl-8b, varied filler, N=4/8/16 × 3) and none moved it, nor did writing the line
in the user's language (which fixed `web_search` reply language):

| variant | weather | shopping_list | reminder | population | others | stale_place |
|---|---|---|---|---|---|---|
| phase2 (shipped) | 9 | 3 | 6 | 3 | 21/36 | 3/9 |
| `topic` slot from `general_answer`/`web_search` | 6 | 0 | 6 | 3 | 15/36 | 3/9 |
| reworded focus line ("only if the last turn names no other target") | 9 | 2 | 3 | 3 | 17/36 | 3/9 |
| both | 6 | 3 | 3 | 3 | 15/36 | 3/9 |
| focus line in Russian (the turn's language) | 9 | 0 | 6 | 3 | 18/36 | 3/9 |
| Russian line + "only if the previous reply named no other place" + one example | 9 | 1 | 6 | 3 | 19/36 | 3/9 |

The topic slot mostly holds a filler question, and at N=16 it displaced the place ("там" was
passed literally as the location).  The English rule in the reworded line was ignored.  At N=4/8 the model still
answers "В Мюнхене" whatever the line says, so the slot itself, not the wording, wins.  Probe
variants `topic`, `reword`, `topic_reword`, `ru_line`, `ru_reword` in `scripts/context_probe.py`.

### `web_search` reply language

The summariser takes its reply language from `cx.user_lang`, not from the question.  The
`query` it sees is written by the agent LLM, often in English or in the locale language
even when the user spoke Russian, and "reply in the language of the question" then
followed the query.  The user message ends with `search.reply_language` from the locale
catalog ("Ответь по-русски…"), written in the reply language itself: an English rule loses
to a few KB of foreign source text.  No extra LLM call.

`scripts/web_search_lang_probe.py` checks it against hand-written results without
DuckDuckGo: 18 cases, ru/de/en questions over foreign-language pages, including the
query-in-another-language ones that leaked every time before the fix.

## 14. Structured datasets

A category can own several **datasets** — "Utilities" → "Bills", "Meter readings" — for facts
the user wants to count rather than search.  Code: `app/datasets.py` (schema, validation,
periods, forecast; pure), `app/storage/datasets.py` (rows, access, `json_extract` aggregation),
`app/tools/datasets.py` (the `datasets` tool and the context block).

**Shape.** A dataset stores a name, a description, a JSON Schema subset (`object` /
`properties` / `required`; leaf types `string`, `number`, `integer`, `boolean`, `array`; string
formats `month` = `YYYY-MM` and `date`; an `x-unit` annotation) and two aggregation hints,
`amount_field` and `period_field`, guessed from the schema when not given.  Each entry is one
denormalized JSON document in `dataset_entries.data`, so adding never rewrites the dataset and
SQLite computes over fields directly.  Nesting is unrestricted; dotted names (`meter.from`)
address nested fields everywhere.

**Access** follows the owning category and is re-checked on every read and write: the
category owner has full rights; a `category_shares` grant on the category *or any ancestor*
gives `read` (stats, forecast, view) or `write` (also add / delete entries).  Schema changes and
dataset deletion are owner-only.  A deleted category hides its datasets; dataset deletion is
soft (`deleted_at`).

**Context.** When a category is in focus (the Conversation `list` slot, else the profile's
last category), `pipeline.py` appends one bracketed line after the focus line: that category's
datasets with their fields (`*` = required, type, unit), the most frequent stored values of up
to two required text fields (so filters match what is stored), today's date, and a one-line
rule for `add`.  Measured cost: ~80 prompt tokens for one dataset; categories without datasets
cost nothing.

**Asking for missing data.** `add` validates before saving.  Missing or invalid required
fields → nothing is saved; the tool returns `data["hold"]` with the partial entry, the
Conversation keeps it for `HELD_TTL_TURNS` turns, and the pipeline hands it back as
`AgentContext.held` (and names the missing fields in the context line).  The next `add` merges
the user's answer into the held entry.  Because an 8B model readily fills a month the user
never said, `add` checks the model's values against `AgentContext.transcript`: a month/date is
dropped when the utterance names no time at all, an optional text value is dropped when it
shares no word stem with the utterance, and relative periods ("за прошлый месяц", "letzten
Monat", "yesterday") are computed by code, overriding the model's arithmetic.

**New fields.** A value with no matching field is never saved silently: `add` enqueues an
`extend` call (schema extension + the entry) in `pending_actions` and says so; it runs when
the user approves it like any deferred action.  `create`, `extend` and `delete` are
`high_write`, so a model-drafted schema also goes through the pending queue unless the speaker
is authenticated; the `@tool(summary=…)` hook makes the queued readback list the drafted
fields instead of `k=v` pairs.

**Numbers are code.** `stats` takes `field`, `op` (sum/avg/min/max/count), `group_by`
(month/quarter/year or any field), equality `filters`, a period keyword (`this_year`,
`last_month`, …) or explicit bounds, and `include_subtree` (same-named datasets in sub-folders)
and runs one SQL aggregate.  `forecast` sums the field per month and fits a linear trend plus
month-of-year seasonal indices (shrunk by k/(k+1) for k observations), with a ≈90 % range from
the residual spread; under 12 months of data it answers that there is not enough.

**UI.** Selecting a category in the item store shows its datasets as read-only tables with a
per-entry delete for profiles with write access.

**Measured** with `orchestrator/scripts/datasets_probe.py` against qwen3-vl-8b on LocalAI
(create, add, missing→ask→answer, stats, forecast with too little data, new field, no share).

## 15. wwyd persona

[wwyd](https://github.com/melnikaite/wwyd) is a separate self-hosted MCP server that holds
one person's style, habits and decision rules as small *claims* and answers "how would the
owner reply here?" (`predict`).  The assistant is a client of it.  Code: `app/wwyd.py`
(OAuth + MCP client, the only place that sees tokens), `app/storage/wwyd.py`,
`app/routes/wwyd.py`, `app/tools/wwyd.py`, `app/wwyd_capture.py`.

**Connection — per profile.**  Settings takes the wwyd URL and runs OAuth 2.1 as a public
client on the server side: protected-resource + authorization-server discovery, dynamic
client registration, PKCE S256, scope `owner`, RFC 8707 `resource` = `<url>/mcp`.  The
callback (`/api/wwyd/callback`) requires the same cookie profile that started the flow and a
single-use `state` (10-min TTL).  Access tokens (1 h) are refreshed before expiry and once on
a 401; wwyd rotates the refresh token on every use.  An owner-scope API key can be pasted
instead.  Tokens live in `wwyd_connections` like the other local secrets (plaintext in the
local DB), are never returned by the API and never logged.  Disconnect revokes at wwyd's
`/revoke` (RFC 7009: the grant with `client_id` only, or the API key itself as `token`) and
deletes the row.  Both connect paths read `serverInfo.version` from `initialize` and refuse
anything outside wwyd 0.2.x (pre-1.0 wwyd breaks on minor bumps); the contract followed is
wwyd's `docs/clients.md`, and unknown response fields are ignored.
Plain `http://` is accepted for loopback only.  wwyd itself is only linked ("Open wwyd"),
never embedded.

**Transport.**  Minimal MCP over Streamable HTTP with `httpx` (initialize →
`notifications/initialized` → one `tools/call` → session DELETE; JSON or SSE replies) rather
than the MCP SDK, to avoid a dependency for four tools.

**Who is who.**  wwyd has no contact identities, only owner-defined *audiences* (slugs with a
description).  The mapping lives here: `wwyd_connections.audience_map` maps
`profile:<id>` (another household member, by voice or their linked Telegram) and `guest`
(unrecognised voice) to an audience; unmapped askers get wwyd's `default` alias (the persona's default audience, whatever it is named).  Names
never go to wwyd.

**Tool visibility.**  Both tools are `tier="api"`; the pipeline sets
`AgentContext.services` per turn (`tools/wwyd.py::wwyd_services`):

| Service | Granted when | Tool |
|---|---|---|
| `wwyd` | one identified speaker (voice match, typed turn or Telegram link — not a cookie under an unmatched voice), not mixed, with wwyd connected | `wwyd_remember` |
| `wwyd_ask` | anyone on the server has wwyd connected; never on mixed turns | `how_would_answer` |

**Read path — `how_would_answer`.**  Target = the named person, else the speaker, else the
only connected profile.  The owner asking about themself names the audience and gets
`include_private`; anyone else is answered as their mapped audience (the model's `audience`
argument is ignored) and always hears wwyd's `disclosure` first.  wwyd's answer is always
sendable; when it comes with `depends_on` (plain words: what the real answer depends on),
the target's own paired Mac is asked for busy times — a fixed read-only AppleScript, start/end
for 7 days, never titles, attendees or places — and the local LLM turns them into a one-line
conclusion ("free tomorrow from 14:00 to 16:00") or NONE.  Only that conclusion goes back to
`predict` as `context`; without a Mac, or when the calendar does not settle it, the first
answer stands.  Whether to auto-reply or
draft stays with the assistant; this tool only speaks the prediction.

**Write path, explicit — `wwyd_remember`** (`low_write`): one third-person statement with
kind and a category → `add_claims` with `status: approved` (the owner said it).  Categories are per persona and
the owner can add more, so the schema carries no enum (a per-turn enum would cost a wwyd
call before every turn): the model's path is matched against `list_categories` (cached 5 min
per profile) by exact path, leaf name or parent, else `communication/style`.

**Write path, automatic — `wwyd_capture`.**  After the reply has been sent (the pipeline's
`finally`), the turn is queued for the identified owner only; a cheap first-person/"tell X"
regex skips most turns, and an in-memory per-profile drain — the conversation summary
fold's pattern — asks the LLM which parts are the owner's own words: dictated messages
or statements about themselves.  Requests to the assistant are dropped.  Items go to
`ingest_raw` as wwyd episodes: a dictated message is a reply (`situation` says it was
dictated, `said` is the text, audience from the recipient's mapping); a statement is `said`
alone.  No category is sent: wwyd places episodes itself, and a path made up here could
carry a name.  wwyd derives claims that land PENDING for review in wwyd.  Unstripped code-word attempts, mixed
turns, guests and turns that already used a wwyd tool are never sent.

**Measured.**  `scripts/wwyd_capture_probe.py` (15 hand-written ru/en/de utterances,
qwen3-vl-8b on LocalAI): 14/15 with the first prompt ("Spiel meine Playlist ab" picked as a
message), 15/15 after naming "play my playlist" among the requests.  End to end against a
local wwyd: OAuth connect, approved claim, two pending claims derived from dictated words,
review, predict for the owner, a mapped member and a guest (2–7 s per predict), revoke on
disconnect.

---

## Where to look next

- [Adding a tool](adding-a-tool.md)
- [`orchestrator/CLAUDE.md`](../orchestrator/CLAUDE.md) — contributor map inside the orchestrator service.
- [`desktop-agent/README.md`](../desktop-agent/README.md) — bootstrap, per-OS capabilities, autostart, reverse-WSS.
- [`xtts-server/README.md`](../xtts-server/README.md) — TTS service bootstrap, model location, wake-word retraining.
