# Contributor patterns — orchestrator

Patterns reference for editing orchestrator code. Start with the [root CLAUDE.md](../CLAUDE.md) for navigation and [`README.md`](README.md) for what this service does. This file's job: get you to the right file in one hop.

## Patterns

### Adding a tool

Drop a module under `app/tools/`, decorate an `async def` with `@tool(...)` from `app.tools.base`, restart. Auto-discovery in [`app/tools/__init__.py`](app/tools/__init__.py) imports it. Full walk-through, decorator params, and worked example: [`docs/adding-a-tool.md`](../docs/adding-a-tool.md).

### `ToolCtx` — use `unwrap_ctx` at the top of every tool

Tools that need request-scoped data (client_id, language, profile, progress) take `ctx` as a kwarg. The agent loop detects via `inspect.signature(handler)` ([`app/tools/base.py::tool`](app/tools/base.py)). Call `cx = unwrap_ctx(ctx)` to get `cx.client_id`, `cx.user_lang`, `cx.profile_id`, `cx.is_authenticated`, and an always-safe `await cx.progress(step, detail=None)` that no-ops without a sink.

Don't reach into raw `AgentContext` — `getattr` diverges into subtly different None-checks across tools.

### User-visible strings go through `i18n.t()`

User-facing locale: `cx.user_lang` (`"en"` / `"ru"` / `"de"`). Build replies with `t(key, lang=cx.user_lang, **fmt)` where `key` is a stable English identifier and the catalog holds the translations. See [`app/i18n.py`](app/i18n.py).

Pick a dot-separated key (`"weather.no_location"`), add EN + RU + DE entries, call `t()`. EN is the fallback; missing keys log a warning and return the key itself so typos are visible.

Numbers, currencies, weather codes, intents and duration phrases also live in the locale JSON — see `app/i18n.py` for the helper API (`num_to_words`, `currency_name`, `currency_alias`, `weather_phrase`, `intent_patterns`, `format_duration_seconds`, `format_when`). Adding a new language is one `locales/<code>.json` file plus the `num2words` / Babel libraries already knowing the code.

### Pick the right risk level

Three levels: `read` / `low_write` / `high_write` — see [`docs/architecture.md` §4](../docs/architecture.md#three-risk-levels) for definitions and examples. Gate is in [`app/agent.py::_execute_one`](app/agent.py). The agent loop never escalates or demotes — classification is the tool's responsibility.

### Terminal vs non-terminal tools

`terminal=True` (default): the tool's `text` IS the final spoken reply. `terminal=False` only when you want the LLM to chain — currently only `general_answer`, signalling `data={"unknown": True}` to make the LLM follow up with `web_search`. See `agent._format_tool_message`.

### LLM calls go through `llm_utils.chat()` / `chat_stream()`

Never call `httpx.post(LLM_URL, …)` directly. [`app/llm_utils.py::chat`](app/llm_utils.py) is the single point — it handles the OpenAI-style body, the text/vision endpoint split, token-usage logging, and the empty-content fallback to `reasoning_content`.

Pass `client_id=ctx.client_id` and `tool_name="my_tool"` for `/api/stats` attribution. The agent loop uses `tool_name="<agent_loop>"` for dispatch round-trips.

Streaming text answers (early-TTS pipelines like `web_search`) use `chat_stream()` and feed sentences into `ctx.stream_sink`.

### Storage writes go through `app/storage/*.py`

Every table has its own module. Public helpers are re-exported from [`app/storage/__init__.py`](app/storage/__init__.py) — import from there, never per-table. All helpers are `async` (wrap blocking SQL in `asyncio.to_thread`).

Do NOT `c.close()` a connection borrowed from `_conn()`. It's thread-local; closing forces re-open on the next query. Legacy `with _lock:` and `try/finally: c.close()` patterns are no-ops kept for migration ([`app/storage/db.py`](app/storage/db.py)).

## Common questions

| Question | File / function |
|---|---|
| WS protocol? | [`app/ws.py`](app/ws.py) — top docstring + `Session.handle_message` |
| Audio FSM states? | [`app/ws.py`](app/ws.py) — `State` enum + FSM diagram at the top |
| WebRTC ↔ FSM? | [`app/webrtc.py`](app/webrtc.py) + `Session.on_mic_pcm` in `ws.py` |
| End-to-end turn? | [`app/pipeline.py::Pipeline.run`](app/pipeline.py) |
| Tool selection? | [`app/agent.py::run_agent`](app/agent.py), dispatch in [`app/tools/base.py::dispatch`](app/tools/base.py) |
| Tool schema cache? | [`app/agent.py::_build_cached_schemas`](app/agent.py) — built once, clock placeholder swapped per turn |
| Offline detection? | [`app/net.py::has_internet`](app/net.py) — cached probe against DuckDuckGo lite |
| Speaker ID? | [`app/speaker.py`](app/speaker.py) (resemblyzer d-vectors) + [`app/storage/speaker_profiles.py`](app/storage/speaker_profiles.py) |
| Passphrase auth window? | [`app/pipeline.py::_extract_passphrase`](app/pipeline.py) + [`app/user_files.py::verify_passphrase`](app/user_files.py) + [`app/storage/auth_sessions.py`](app/storage/auth_sessions.py) (UI cookie path) |
| Pending actions? | [`app/storage/pending_actions.py`](app/storage/pending_actions.py) + [`app/pending_executor.py`](app/pending_executor.py) |
| Semantic memory? | [`app/memory.py`](app/memory.py) — fastembed + cosine in [`app/storage/utterances.py::get_candidate_utterances`](app/storage/utterances.py) |
| Vision (one-shot)? | [`app/vision.py`](app/vision.py), used by [`app/tools/look_at_screen.py`](app/tools/look_at_screen.py) and the image-attach short-circuit in `pipeline.py` |
| Agentic vision loop? | [`app/vision_loop.py`](app/vision_loop.py) + [`app/vision_primitives.py`](app/vision_primitives.py) |
| Read-only gate on `computer_use`? | [`app/tools/computer_use.py`](app/tools/computer_use.py) + [`app/tools/desktop.py::_classify_applescript_risk`](app/tools/desktop.py) + [`app/vision_loop.py::_validate_action`](app/vision_loop.py) |
| LLM-generated AppleScript? | [`app/tools/computer_use.py`](app/tools/computer_use.py) — goal → secondary `chat()` → risk gate → execute |
| Multi-agent registry? | [`app/desktop_client.py`](app/desktop_client.py) — HTTP + reverse-WSS |
| Reverse-WSS desktop-agents? | [`app/agent_proxy.py`](app/agent_proxy.py) + `/v1/agent/connect` in `main.py` |
| Reminders fire? | [`app/scheduler.py`](app/scheduler.py) — `add_reminder` in [`app/storage/reminders.py`](app/storage/reminders.py) |
| Per-user memory.md / settings.json? | [`app/user_files.py`](app/user_files.py) — flat files under `$DATA_DIR_CONTAINER/users/<profile_id>/` |
| Web Push? | [`app/push.py`](app/push.py) + [`app/storage/push_subscriptions.py`](app/storage/push_subscriptions.py) |
| Voicemail rows + audio? | [`app/storage/voice_messages.py`](app/storage/voice_messages.py) — row + WAV under `$DATA_DIR_CONTAINER/voice_messages/` |
| Live voicemail push to recipient's tab? | [`app/ws.py::notify_voicemail_arrived`](app/ws.py) + `_SESSIONS_BY_PROFILE` registry |

## Gotchas

- **Tests use a real `/tmp` SQLite file, NOT `:memory:`.** Thread-local connections + `:memory:` break — each pool thread gets its own empty DB. See [`tests/conftest.py`](tests/conftest.py). If a fixture opens its own connection, route through `app.storage.db._conn()` or call `close_thread_conn()` in teardown.

- **`network_mode: host` means `host.docker.internal` doesn't work** — the container's `localhost` IS the host. Talk to LM Studio / mlx-whisper / xtts-server / desktop-agent via `localhost:<port>`.

- **Schema migrations are idempotent.** Adding a column = another entry in `_add_columns([...])` in [`app/storage/schema.py`](app/storage/schema.py). The helper swallows `OperationalError: duplicate column` — don't write conditional ALTERs.

- **Frontend is mounted read-only** (`./frontend:/app/static:ro`). User-generated content (custom voice WAVs, voicemail audio, VAPID private key) goes under `$DATA_DIR_CONTAINER` (default `/data`).

- **LM Studio / OpenAI-compatible providers diverge on tool-calling.** Gemma 4 E4B in LM Studio emits `tool_calls` consistently; some providers don't, or return non-standard shapes. Parsing tolerance lives in [`app/llm_utils.py::parse_tool_calls`](app/llm_utils.py) — extend it there, don't scatter try/except across tool code.

- **Gemma / DeepSeek-R1 split output into `content` + `reasoning_content`.** If the model spent its budget thinking, `content` is empty. `extract_text()` falls back to the last paragraph of reasoning — don't read `message["content"]` directly.

- **`pipeline.py` owns the per-turn state machine.** Don't add side state in `ws.py`; the Session is just a hookable shell. New turn-level concerns (auth window, image attach, voicemail short-circuit) land in pipeline branches.

- **`ws.py` and `pipeline.py` interlock to avoid a circular import.** `pipeline.py` does `from .ws import notify_voicemail_arrived` inside the voicemail-save branch (not at module load). Keep the late import.

- **Tool schemas are cached after first build.** [`app/agent.py::_TOOL_SCHEMAS_CACHED`](app/agent.py). Unit tests that register a fresh `@tool(...)` after the cache must call `agent.invalidate_schemas_cache()`.

- **Sessions can authenticate two ways.** Voice (passphrase spoken, 5-min window) and cookie (UI login, `va_session`, sliding 30-day). Both land in `Session._set_known_profile`; the by-profile registry needs the funnel so a closed cookie session doesn't keep collecting voicemail pings.

- **XTTS reference-audio path crosses the container boundary.** Custom voices live at `$DATA_DIR_CONTAINER/custom_voices/<id>.wav` inside, but xtts-server reads `$DATA_DIR_HOST/custom_voices/<id>.wav` on the host. See `app/tts.py::_to_host_path` — don't pass container paths to the TTS API.

- **`pending_actions` rows expire silently.** Read paths filter by `expires_at > now()` rather than running a sweep job. Same for `auth_sessions`. Sweep helpers (`_sweep_expired_sync`) exist but aren't wired into the scheduler — at single-household scale the tables stay tiny.

## Where the layers meet

```
Browser PWA
    │  WebRTC audio + JSON control
    ▼
ws.py::handle_ws → Session  (FSM: LISTENING_WAKE → RECORDING → PROCESSING → SPEAKING → CONTINUATION)
    │  on utterance_end / VAD / ptt_end
    ▼
pipeline.py::Pipeline.run   (ASR + speaker ID parallel; short-circuits for passphrase / voicemail / vision)
    │
    ▼
agent.py::run_agent          (LLM tool-use loop, 4-step bound)
    │  one tool call per step
    ▼
tools/base.py::dispatch      (handler lookup + ctx injection + exception → ToolResult)
    │
    ▼
tool handler   →  llm_utils.chat / chat_stream      (LLM calls)
               →  desktop_client.run_applescript    (host automation)
               →  storage/*                         (SQLite reads + writes)
               →  net.has_internet                  (offline gating)
```

Outbound TTS audio flows back: tool's `ToolResult.text` → `Pipeline` → `hooks.on_response` → `Session._synth_for_playback` → `tts.stream()` (HTTP to xtts-server) → `webrtc.push_tts_pcm()` → browser's WebRTC voice engine → speakers (with AEC3 reference, so the mic doesn't echo).

Streaming tools (`web_search`) call `ctx.stream_sink(sentence)` per sentence — runs through `hooks.on_response_chunk` → straight to TTS, so first-audio drops from ~5 s to ~1.5 s.
