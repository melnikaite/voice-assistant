# Frontend — patterns and navigation map

Vanilla JS, no framework. Read `frontend/README.md` first for file layout. This doc is the contributor's "where do I touch what?" map.

## Core patterns

### 1. i18n

Every user-facing string is keyed in `frontend/i18n.js::CATALOG` (mirror of `orchestrator/app/i18n.py`). Two attach points:

- **HTML** — `<button data-i18n="controls.start">Start</button>`. Walked at boot by `applyI18n()` (`frontend/i18n.js:260-275`). The English text between tags is the fallback served before JS runs. Also supported: `data-i18n-placeholder` for inputs and `data-i18n-title` for tooltips.
- **JS** — `import { t } from './i18n.js'` then `t('inbox.empty')` or `t('inbox.notify_body', {from_name: 'Alice'})`. Placeholders are `{name}` (mirrors Python's `str.format`).

Don't hardcode user-facing text. If you need a new string, add it to the catalog in both `frontend/i18n.js` and `orchestrator/app/i18n.py` (the latter if the orchestrator returns the same string). See `docs/adding-a-locale.md`.

### 2. State changes via `setState(value)`

All UI state transitions go through `main.js::setState` (`main.js:156-172`). Don't manipulate `stateCardEl.dataset.state` or `stateLabelEl.textContent` from event handlers — call `setState('processing')`. This guarantees the matching sound cue plays, the label/sub-text pair stay in sync, and the continuation countdown cleans up.

Valid states are the keys of `STATE_LABELS` (`main.js:82-94`). Adding a new state means adding a label/sub pair AND wiring the backend to emit it (server sends `{type: 'state', value: ...}` over WS — `main.js:413-426`).

### 3. WS messages — switch in `onWsMessage`

`main.js::onWsMessage` (`main.js:382`) is one `switch (msg.type)`:

| `msg.type`           | Effect                                           |
|----------------------|--------------------------------------------------|
| `ready`              | Initial state set; history-restore count logged. |
| `webrtc_answer`      | Apply remote SDP.                                |
| `webrtc_ice`         | Add remote ICE candidate.                        |
| `state`              | `setState(msg.value)`; manage countdown.         |
| `progress`           | Update PROCESSING card sub-text via `PROGRESS_LABELS`. |
| `wake_ack`           | Server confirmed wake; reset transcript / response panes. |
| `transcript`         | Show ASR result + speaker label.                 |
| `tool_called`        | Show tool name + args, flash agent panel.        |
| `response`           | Display final reply text (TTS audio comes over RTC). |
| `attach_image_ack`   | Server accepted an attached image.               |

Add a new type? Add the `case` in `onWsMessage`. Don't dispatch WS messages anywhere else.

### 4. Audio path

```
mic getUserMedia → AudioContext → AudioWorkletNode (worklet.js)
                                        │
                                        ├─→ wake.js (onnxruntime-web)
                                        │
                 RTCPeerConnection ──→ server (aiortc → VAD → ASR)
```

Two sinks on the same mic stream: the worklet feeds Int16 PCM into `wake.js`; WebRTC sends the same mic over an audio track to the orchestrator for ASR.

The browser's WebRTC voice engine handles AEC — that's why remote TTS arrives as a **remote audio track** attached to `<audio id="remote-tts">`, NOT decoded into the `AudioContext`. AEC subtracts the playback from the mic stream before consumers see it. Bypassing the `<audio>` element (e.g. routing TTS through `AudioContext.destination`) breaks AEC; mic→speaker feedback comes back.

Don't introduce an intermediate `<audio>` for any other audio path — voicemail playback has its own `_voicemailPlayer` element (`main.js:773`); reuse that pattern.

## Where things live

| Question                                  | File / function                                            |
|-------------------------------------------|------------------------------------------------------------|
| Wake-word trigger?                        | `wake.js::BrowserWakeDetector` — three ONNX models on Int16 PCM. Triggers `onWake` past `threshold`. |
| Mic enable / disable?                     | `main.js::start()` (`main.js:212`) / `cleanup()` (`main.js:800`). |
| LLM progress reaction?                    | `main.js::onWsMessage` → `case 'progress'` (`main.js:427-439`). Looks up `PROGRESS_LABELS[msg.step]`. |
| Push notification register?               | `main.js::_setupWebPush` (`main.js:671`) + `sw.js::push`. VAPID public key from `/api/push/vapid_public_key`. |
| Image attach?                             | `main.js::handleAttachFile` (`main.js:869`) — file input + drag-drop + paste converge here. Sends `{type: 'attach_image'}` over WS. |
| Language source of truth?                 | `localStorage.va_lang` + `i18n.js::_readLang()`. Settings panel writes via `setLang(...)`. |
| Voicemail player?                         | `main.js::_playVoicemailAudio` (`main.js:774`) — lazy `<audio>`, reused across plays. |
| Push-to-talk?                             | `main.js::pttStart` / `pttEnd` (`main.js:992`/`998`). Also bound to `Space`. |
| Speaker / voice enrollment?               | `main.js::startEnrollFlow` / `startCloneRecording` — own `getUserMedia`, POST to `/api/speakers` or `/api/custom_voices`. |
| Stats dashboard?                          | `main.js::renderStats` (`main.js:1453`) + chart helpers below. Chart.js loaded once from `vendor/`. |

## Gotchas

- **Service worker caching.** A new `sw.js` doesn't activate until the user closes every tab OR you manually unregister in DevTools. Symptom: pushes go to the old worker. Fix: DevTools → Application → Service Workers → Unregister + hard reload.
- **openWakeWord ONNX models are ~5 MB total.** Don't break the cache by fingerprinting filenames — `wake.js` builds paths via `./models/${name}.onnx` from `/api/config`.
- **AudioContext needs a user gesture in some browsers.** The wake detector starts ONLY after the Start button click, never automatically — keep it that way.
- **Frontend is mounted READ-ONLY** (`./frontend:/app/static:ro`). Uploads go through dedicated API endpoints (`/api/custom_voices`, `/api/speakers`).
- **`localStorage` keys aren't namespaced.** Prefix new keys with `va_` to avoid collisions on the same origin (`va_lang`, `va_client_id`).
- **No build step.** No bundler, no transpiler. The file the user fetches is the file you edit. Modern browsers handle ES modules natively.

## Adding a new UI feature

1. **HTML** in `index.html`. Use `data-i18n="key.path"` for text. Group controls into an `<article>` or `<details>` following existing panels.
2. **JS state + handler** in `main.js`. Pin DOM refs at the top (existing block ~`main.js:24-50`).
3. **WS protocol** if backend round-trips are needed. Pick a `type`, document where the message is sent (search `ws.send(`), add a `case` in `onWsMessage`. Backend: add a handler in `orchestrator/app/ws.py`.
4. **CSS** in `style.css`. Stick to Pico variables (`var(--pico-muted-color)` etc.) so dark mode "just works".
5. **i18n** — every `data-i18n` key needs a matching `CATALOG` entry in `frontend/i18n.js`.
