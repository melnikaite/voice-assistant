# xtts-server — contributor notes

Service overview lives in [`README.md`](README.md).  This file is for
folks (LLM or human) coming in to modify the TTS server.

## Mental model

A single Python process holding the Coqui XTTS-v2 model in memory and
streaming PCM over HTTP.  Lives on the **host** (not Docker) so
PyTorch can use Apple Silicon MPS or NVIDIA CUDA — both ~3-5× faster
than CPU-only inside a container.

Same architectural pattern as `mlx-whisper` and the LLM provider:
long-running process, HTTP-friendly, OS-native GPU access.  Swappable
— the orchestrator only knows the OpenAI-style synth contract.

## File map

```
xtts-server/
├── xtts-server.py              # Single-file FastAPI service
├── pyproject.toml              # uv-managed deps (coqui-tts + torch)
├── start.sh                    # `uv run` wrapper, sets device
├── install-autostart.sh        # launchd plist (macOS)
├── uninstall-autostart.sh
├── uninstall.sh                # Nukes .venv and uv.lock
└── README.md                   # User-facing docs (training, install, etc.)
```

## Key code in `xtts-server.py`

| Section | What lives there |
|---|---|
| `_device_for()` | Picks MPS / CUDA / CPU at boot |
| `XTTS` load | One-time on startup; model in `~/.cache/voice-assistant/xtts/` |
| `/v1/health` | Liveness + active device + sample rate |
| `/v1/speakers` | Built-in 58 speaker names |
| `/v1/synthesize` | Streams audio/pcm (chunked); takes text + lang + optional speaker + optional reference_audio for cloning |

## Adding a feature

The most common need: support a new TTS model.

1. Pick something that has a similar streaming-PCM contract.  If it
   doesn't stream, the orchestrator-side latency will suffer (TTS-to-
   first-audio is the biggest variable in time-to-spoken-reply).
2. Either replace `xtts-server.py` or add a sibling module behind an
   `XTTS_BACKEND` env switch.
3. Match the `/v1/health` and `/v1/synthesize` contracts — orchestrator
   does NOT speak XTTS-specific protocols, only the HTTP shapes here.
4. Update the README's "Configuration" table.

## Voice cloning path

`POST /v1/synthesize` with `reference_audio` set to an ABSOLUTE HOST
PATH triggers Voicebox v2 cloning.  The orchestrator (in container)
maps `/data` to `${PWD}/data` via `DATA_DIR_HOST`/`DATA_DIR_CONTAINER`
so it can send xtts-server a host-visible path.  See
`orchestrator/app/tools/speaker.py::save_custom_voice` for the recipe.

## Gotchas

- **First boot downloads ~1.8 GB** — model into `~/.cache/voice-assistant/xtts/`.
  Subsequent starts are ~3 s.  If `XTTS_MODEL_DIR` env is empty the
  default cache path is used.
- **CPML license is non-commercial**.  Don't ship a commercial product
  bundling these weights without re-licensing.  Swap to Bark or Piper
  for commercial use; both have streaming-friendly servers in the
  community.
- **MPS doesn't pass through Docker** — that's WHY this service is on
  the host.  Don't try to containerize without a re-evaluation of the
  perf cost (CPU XTTS is ~RTF 4 — i.e. 4 seconds of synth per second
  of speech — vs MPS ~RTF 0.3).
- **Streaming chunk size** is `XTTS_STREAM_CHUNK_SIZE` (default 20).
  Lower = faster first-audio but more network overhead; higher =
  smoother but later.  20 is the sweet spot on M-series.
- **Sample rate is 24 kHz mono int16**, exposed via `X-Sample-Rate`
  response header so clients don't hard-code it.  Browser-side
  AudioContext resamples to its own native rate.
