# frontend

Vanilla-JS PWA. No build step, no framework, no bundler. The files in
this directory are served verbatim by the orchestrator at
`http://localhost:8080` over a read-only Docker mount
(`./frontend:/app/static:ro` — see `docker-compose.yml:25`).

## Layout

| File              | Purpose                                                                                  |
|-------------------|------------------------------------------------------------------------------------------|
| `index.html`      | Top-level UI: auth banner, controls, state card, panels for inbox / pending / memory / settings / stats. `data-i18n` attributes drive translation. |
| `main.js`         | Everything that runs. Mic capture, WebRTC peer, WS dispatch, UI state, push subscription, panel logic. ~2400 lines, no modules beyond `i18n.js` + `wake.js`. |
| `wake.js`         | Browser-side openWakeWord port — three ONNX models (melspec → embedding → wake). Receives Int16 PCM from the worklet, fires `onWake` when the score crosses threshold. |
| `worklet.js`      | AudioWorklet that pulls mic samples out of the AudioContext as Int16 PCM frames. |
| `sw.js`           | Service worker. Web Push delivery + notificationclick routing. No static-asset cache (the orchestrator already serves at LAN speed). |
| `i18n.js`         | `t(key, lang)` + `applyI18n()`. Mirror of `orchestrator/app/i18n.py`. en/ru/de today. |
| `style.css`       | Pico.css overrides + custom state-card / chart / panel rules. BEM-ish naming. |
| `vendor/`         | Pinned third-party JS/CSS — pico.min.css, chart.umd.min.js. Refreshed via `vendor/fetch.sh`. Vendored so the assistant works fully offline. |
| `models/`         | openWakeWord ONNX models. Default ships `hey_jarvis_v0.1.onnx` + `melspectrogram.onnx` + `embedding_model.onnx`. Drop additional `.onnx` files here to enable different wake-words. |

## Dependencies

Vendored in `frontend/vendor/`. No npm, no package.json. If you need to
refresh the pins, edit `vendor/fetch.sh` (it has the upstream URLs +
versions) and re-run it.

ONNX Runtime Web is loaded directly from jsdelivr at the top of
`wake.js` — pinned to `1.20.0`. The browser caches it after first
load; if you need true offline cold-start, vendor it too.

## Running locally

The orchestrator serves `frontend/` directly. Bring it up:

```bash
docker compose up -d
open http://localhost:8080
```

The mount is `:ro`, so edits to any file in `frontend/` show up on the
next browser refresh — no container rebuild needed. The orchestrator
container itself never touches these files; it just serves them.

## Service worker

`sw.js` registers at `/` after a successful login (see
`main.js::_setupWebPush`). It handles two events:

- `push` — show a system notification. Payload schema set by the
  orchestrator's `app/push.py`: `{title, body, voicemail_id, tag}`.
- `notificationclick` — focus an existing tab if one is open;
  otherwise open a fresh one at the URL stashed in the payload.

When you change `sw.js`, browsers serve the cached old version until
the next navigation. To force a refresh during development:

1. DevTools → Application → Service Workers → "Unregister".
2. Hard reload (Cmd-Shift-R / Ctrl-Shift-F5).
3. The new SW installs on the next page load.

There is no static-asset cache, no offline shell. The service worker
is purely for push delivery — see the comment block at the top of
`sw.js` for the rationale.
