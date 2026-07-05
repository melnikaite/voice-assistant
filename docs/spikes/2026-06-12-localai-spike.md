# Spike: LocalAI as the default bundled backend (task #58)

**Date:** 2026-06-12 · **Host:** MacBook Pro M1 Max 64 GB, macOS · **LocalAI:** 4.4.1 (brew)

Question: can LocalAI replace the "bring your own LM Studio" default and the
two-wrapper MLX stack (mlx_vlm.server :1237 + mlx-openai-server :18000) as the
single bundled backend — and does its new `/v1/audio/diarization` give us the
mixed-speaker mechanism for #59 on macOS?

## Verdict: GO for LLM + ASR. Diarization is Linux-only today.

One brew-installable binary serves LLM (text+vision+tools) and Whisper ASR on
one port with Metal acceleration, at performance parity with the MLX stack,
reusing the exact GGUF files LM Studio already had on disk. The audio-identity
endpoints LocalAI advertises are not usable on macOS yet.

## Measurements (warm, same prompt set)

| Metric | LocalAI 4.4.1 (llama-cpp Metal, Q4_K_M) | mlx_vlm.server 0.6.2 (QAT 4-bit) |
|---|---|---|
| Full reply, ~115 tok | 2.43 s | 2.26 s |
| Decode incl. prefill | 48.2 tok/s | 50.5 tok/s |
| Streaming TTFT | ~0.18 s | 0.12 s |
| Tool calls | ok (`get_weather {"city":"Berlin"}`) | ok |
| Vision (red square) | ok | ok |
| Whisper 12.7 s WAV | 0.32 s warm (RTF 0.025, whisper-base-q5_1 Metal) | n/a (separate wrapper) |

Setup notes: `brew install localai`; backends are OCI images installed
explicitly (`local-ai backends install llama-cpp whisper`); models declared as
YAML in the models dir; our LM Studio GGUFs were reused via symlink — no
re-download.

## Caveats found

1. **Adaptive thinking.** With the lmstudio-community GGUF template the model
   sporadically enters a reasoning phase (delta.reasoning chunks); content then
   starts seconds late and can exhaust `max_tokens` entirely. Non-deterministic
   — same prompt thought for 5 s in one run and answered directly in the next.
   Must be forced off in the model YAML/template before any voice use; the
   request-level knobs (`reasoning_effort`, `chat_template_kwargs`) all
   *appeared* to work but the baseline stopped thinking too, so the effective
   knob is unverified.
2. **CWD pollution.** Both the server and the CLI write state relative to the
   working directory (`data/`, `backends/`). Run everything with
   `WorkingDirectory=~/.localai` (LaunchAgent) and `LOCALAI_BACKENDS_PATH` set.
3. **MLX-inside-LocalAI is real but immature.** `localai@mlx` has a darwin
   image, installs, and loads our QAT checkpoint from the HF cache — but chat
   templating is broken for gemma-4 E4B (degenerate echo/loop output) and cold
   load took 82 s. llama-cpp is the Mac path today; revisit the MLX backend
   later as the perf option.

## Not available on macOS (Linux-only in practice)

- **`/v1/audio/diarization`**: `sherpa-onnx` backend ships no darwin/arm64
  image; `metal-vibevoice-cpp` darwin image exists but is **mis-packaged** —
  it contains sources and a half-built CMake tree, no `vibevoice-cpp` binary,
  so the backend never starts ("grpc service not ready"). Also note weight
  class: vibevoice-asr is a 9.7 GB model — disproportionate for diarizing
  10-second utterances even when fixed.
- **`/v1/voice/embed`**: `speaker-recognition` backend has no darwin/arm64
  image either.
- **Consequence for #59:** mixed-speaker detection on macOS proceeds with the
  planned v1 mechanism (resemblyzer windowed partials, zero new deps) behind
  the `SpeakerAttribution` interface; LocalAI diarization slots in later for
  Linux deployments or once darwin images are fixed upstream.

## DiffusionGemma status

llama.cpp support is an **unmerged PR** (#24427); the unsloth GGUF only runs on
their branch. No backend we can install loads it today — skip the 16.8 GB
download, recheck after the PR merges (then it becomes a drop-in YAML model).

## Post-spike deployment state (2026-06-12, same day)

- LocalAI now runs persistently via the `com.voiceassistant.localai`
  LaunchAgent on **:1240**, serving `gemma-4-e4b-it-qat-q4_0` — the gallery's
  **QAT** GGUF that LocalAI downloaded itself (with its own mmproj for
  vision).  Smoke re-passed: warm 112-tok reply 1.93 s, tools ok, vision ok.
- **LM Studio is removed** (app + `~/.lmstudio` moved to Trash).  The
  orchestrator's documented manual fallback is now LocalAI :1240.
- Two operational footguns found the hard way:
  1. `local-ai models install` (CLI) can leave a server running on the
     **default :8080**, which collides with the orchestrator.  Always run
     the server via the LaunchAgent; don't rely on CLI-spawned instances.
  2. On macOS, container-published ports are held by **OrbStack Helper**
     processes — `kill` by port without checking the owner kills the
     helper, takes down `docker.sock`, and the OrbStack restart bounces
     ALL containers.  Verify `ps -o comm=` before killing anything.

## Distribution plan (when we cut over)

- LocalAI on **:1240** serves both `LLM_URL` and `WHISPER_URL` (one env, one
  service). XTTS stays (voice cloning is our feature; LocalAI has TTS backends
  but cloning parity is unverified). MLX stack (:1237/:18000) becomes the
  documented opt-in perf path; LM Studio no longer required.
- Linux story: identical YAML/config with CUDA llama.cpp images — this is the
  r/selfhosted distribution path, and there diarization/voice-embed actually
  work, including for #59 v2.
- Wizard (#44) shrinks to: brew/curl LocalAI → install 2 backends → 2 models.

## Upstream issues filed (2026-06-12)

1. [mudler/LocalAI#10267](https://github.com/mudler/LocalAI/issues/10267) —
   darwin metal-vibevoice-cpp OCI image ships sources instead of build
   artifacts (backend cannot start).
2. [mudler/LocalAI#10268](https://github.com/mudler/LocalAI/issues/10268) —
   no darwin/arm64 images for sherpa-onnx + speaker-recognition
   (diarization/voice-embed currently Linux-only).
3. [mudler/LocalAI#10269](https://github.com/mudler/LocalAI/issues/10269) —
   mlx backend: degenerate looping output with gemma-4 E4B (chat template
   apparently not applied) + slow cold load.

When #10267/#10268 land, diarization becomes the v2 mechanism for
mixed-speaker attribution on macOS; #10269 unblocks "MLX engines inside
LocalAI" as the single-server perf path.
