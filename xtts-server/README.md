# xtts-server

Host-side TTS service for the voice assistant.  Runs **directly on the
host** (not in Docker) so PyTorch can use Apple Silicon's MPS backend
or NVIDIA CUDA — both ~3-5× faster than CPU-only Docker.

Same architectural pattern as `mlx-whisper` (port 18000) and
`lm-studio` (port 1234): a long-running Python process that holds the
model in memory and exposes HTTP.

## Why this is isolated from your system

Everything goes through [**`uv`**](https://github.com/astral-sh/uv),
the Python project manager from astral.sh.  Key property: uv brings
its **own Python toolchain** that lives in `~/.local/share/uv/python/`
and is fully independent of the system / Homebrew Python.

What this means in practice:

* Installing xtts-server doesn't add anything to your system Python.
* `brew upgrade python` can never break xtts-server — it doesn't use
  Homebrew's Python at all.
* The project's venv lives at `./xtts-server/.venv/` — nuke that
  directory and xtts-server is gone from your system.
* The model (1.8 GB) lives in `~/.cache/voice-assistant/xtts/`.
  Survives every `docker compose` operation; survives uninstall of
  xtts-server itself.  Delete it manually when you actually want to
  reclaim the disk.

## Install + run (one terminal, foreground)

```bash
# One-time: install uv (single binary, no system Python required)
curl -LsSf https://astral.sh/uv/install.sh | sh         # or `brew install uv`

# Every time you want xtts-server up:
cd xtts-server
./start.sh
```

First run:
1. `uv sync` downloads Python 3.12 into uv's data dir (~150 MB)
2. Installs deps into `./.venv/` (~500 MB — torch, coqui-tts, etc.)
3. Writes `uv.lock` (committed; pins exact versions for reproducible reinstalls)
4. `xtts-server.py` boots → downloads XTTS-v2 model (~1.8 GB) to
   `~/.cache/voice-assistant/xtts/`
5. Loads model onto MPS (Apple Silicon) / CUDA / CPU
6. Listens on `http://127.0.0.1:9876`

Subsequent runs: ~3 seconds to load, model + venv already cached.

## Run in the background

The simplest approach is tmux:
```bash
tmux new -s xtts
./start.sh
# Ctrl-b d to detach.  Reattach later: tmux attach -t xtts
```

## Autostart on Mac (optional)

If you want xtts-server to come up automatically every time you log
into macOS:

```bash
cd xtts-server
./install-autostart.sh
```

This drops a launchd plist into `~/Library/LaunchAgents/` (only your
user — no system-wide changes), wires up auto-restart on **crash**
(KeepAlive.SuccessfulExit=false, so a clean `kill -TERM` won't be
fought by launchd), points stdout/stderr at
`~/Library/Logs/voice-assistant/xtts.{out,err}.log`, and bakes in
`/opt/homebrew/bin`, `~/.local/bin`, `/usr/local/bin` and the standard
system dirs as the PATH so launchd can find `uv`.

Logs:
```bash
tail -f ~/Library/Logs/voice-assistant/xtts.err.log
```

Disable later:
```bash
./uninstall-autostart.sh
```

## Training a custom wake-word

The default wake phrase is **"Hey Jarvis"** (model file:
`frontend/models/hey_jarvis_v0.1.onnx`).  Two ways to change it:

### 1. Use a pre-trained model from openWakeWord

`openWakeWord` ships a small zoo of pre-trained models — "alexa",
"hey mycroft", "hey rhasspy" — drop one into `frontend/models/`, then
flip `WAKE_WORD_NAME` in `docker-compose.yml` (orchestrator service) to
the file name without the `.onnx` suffix.  Restart the orchestrator —
no rebuild needed, it just hands the new name to the frontend via
`/api/config`.

### 2. Train your own model in Colab

The openWakeWord author published a Colab notebook that trains a
custom wake phrase in about 30 minutes:

* Repo: https://github.com/dscripka/openWakeWord
* Training notebook: see the README under "Custom Models" in the repo;
  link is `openwakeword/Custom_Model_Training_Notebook.ipynb` in the
  Colab badge.

The four steps:

1. **Open the notebook in Colab** (GPU runtime required — free tier
   T4 is enough).
2. **Set your wake phrase** in the first cell, e.g. `"hey rosa"`.
   The notebook will synthesise ~2000 positive samples with multiple
   voices via Piper TTS and add a few thousand negative samples from
   the Mozilla Common Voice corpus.
3. **Run all cells** — training takes ~15-30 min on a free Colab T4.
4. **Download the resulting `.tflite` and `.onnx`.**  Drop the
   `.onnx` file into `voice-assistant/frontend/models/`.

Then set `WAKE_WORD_NAME` to the filename without `.onnx` (and tweak
`WAKE_WORD_THRESHOLD` if you find it too jumpy / too dull) and
`docker compose up -d --force-recreate orchestrator`.

## Configuration (env vars)

| Var | Default | What it does |
|--|--|--|
| `XTTS_PORT` | `9876` | Bind port (9000 conflicts with OrbStack on macOS) |
| `XTTS_HOST` | `127.0.0.1` | Bind interface |
| `XTTS_DEVICE` | `auto` | `mps` / `cuda` / `cpu` / `auto` |
| `XTTS_SPEAKER` | `Claribel Dervla` | Default built-in speaker |
| `XTTS_MODEL_DIR` | `~/.cache/voice-assistant/xtts` | Where the 1.8 GB model lives |

Set in your shell before running `./start.sh`, e.g.:
```bash
XTTS_SPEAKER="Sofia Hellen" ./start.sh
```

## API

```http
GET  /v1/health        — liveness + device + sample rate
GET  /v1/speakers      — list of 58 built-in speaker names
POST /v1/synthesize    — streams audio/pcm (chunked transfer-encoding)
```

Synth request body:
```json
{
  "text": "Hello, world!",
  "lang": "en",
  "speaker": "Claribel Dervla",
  "reference_audio": null,
  "stream_chunk_size": 20
}
```

Response: `audio/pcm`, int16 mono at 24 kHz.  Sample rate is also in
the `X-Sample-Rate` response header so clients don't need to hard-code
it.  Setting `reference_audio` to an absolute host path enables voice
cloning (Voicebox v2 feature).

## Uninstall

Two layers:

```bash
./uninstall.sh              # removes .venv/ and uv.lock from this folder
./uninstall-autostart.sh    # disables launchd autostart (if installed)
```

These are intentionally separate.  The script tells you what's still
on disk afterwards (the model and uv itself) and the exact commands
to nuke them.

## License notes

XTTS-v2 model is under Coqui's **CPML** — non-commercial OSS.  Fine for
this personal home assistant.  Commercial deployment needs different
licensing.
