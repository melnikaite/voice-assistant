FROM python:3.12-slim

WORKDIR /app

# Runtime libs:
#   libgomp1     — OpenMP for ONNX (fastembed embedding model).
#   libopus0     — WebRTC's audio codec; pylibsrtp / PyAV may load the
#                  system opus instead of their bundled copy depending on
#                  the manylinux wheel build, so we keep this around to
#                  be safe.
#   ca-certificates — TLS roots for httpx (HF / DDG / web_search fetches).
# Note: PyAV and pylibsrtp ship manylinux wheels with FFmpeg + libsrtp2
# vendored in, so we don't need libavformat / libavcodec / libsrtp2-1 as
# system packages.  If we ever build from source they're back in scope.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopus0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'fastapi==0.115.*' \
    'uvicorn[standard]==0.32.*' \
    'httpx==0.27.*' \
    'python-multipart==0.0.*' \
    'numpy<2' \
    openwakeword \
    ddgs \
    'selectolax==0.3.*' \
    'trafilatura>=1.12' \
    'fastembed==0.4.*' \
    'apscheduler==3.10.*' \
    'aiortc>=1.9' \
    'bcrypt>=4.1' \
    'pywebpush>=2.0' \
    'num2words>=0.5.13' \
    'Babel>=2.16'
# Locale-aware helpers:
#   num2words → "fifteen" / "пятнадцать" / "fünfzehn" — 40+ languages.
#   Babel     → format_timedelta, currency names, date/time per locale.
# Both replace the previous Python-coded `_num_to_ru` / `_WMO_RU` /
# `_fmt_duration` tables, so adding a new language is just dropping a
# `orchestrator/app/locales/<code>.json` file with no Python edits.
# pywebpush pulls in `py_vapid`, `http_ece`, and `cryptography`.  We use
# it for VAPID JWT signing + aes128gcm payload encryption on the Web
# Push pipeline (orchestrator/app/push.py); see that module for why we
# wrap the sync `webpush()` call in `asyncio.to_thread` rather than
# rolling the encryption by hand.

# Note on TTS: the orchestrator does NOT synthesise speech itself.
# A separate host-side service (xtts-server, see ../xtts-server/) runs
# XTTS-v2 on Apple Silicon MPS / CUDA and streams PCM over HTTP.  The
# container only carries a thin httpx client (app/tts.py) — same pattern
# we use for ASR (mlx-whisper) and LLM (lm-studio).  Big wins: image
# stays small, model stays on the host filesystem (never gets caught up
# in Docker volume cleanup), and inference runs at GPU speed instead
# of the ~4× RTF we'd hit with CPU-only Docker.

# PyTorch CPU-only build (~300 MB) — required by resemblyzer for the
# GE2E speaker encoder.  Installing separately so Docker can cache this
# heavy layer independently.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# resemblyzer pulls in webrtcvad which only ships a source tarball — no
# prebuilt wheels for linux/arm64 + Python 3.12. Install gcc just long enough
# to compile it, then purge the toolchain so the final image stays slim.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir resemblyzer && \
    apt-get purge -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Pre-download the multilingual embedding model so the image works offline.
# MiniLM-L12 paraphrase: ~220 MB ONNX. Re-run `docker compose build` if you
# change EMBEDDING_MODEL (keep this name in sync with docker-compose.yml).
ENV FASTEMBED_CACHE_PATH=/root/.cache/fastembed
RUN python -c "\
from fastembed import TextEmbedding; \
m = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
list(m.embed(['warmup'])); \
print('embedding model pre-cached')"

# Pre-download resemblyzer's GE2E speaker encoder weights (~25 MB).
# Baked into the image so it's available offline.
RUN python -c "\
from resemblyzer import VoiceEncoder; \
VoiceEncoder(); \
print('speaker encoder pre-cached')"

# No TTS env defaults here — TTS_URL points at the host xtts-server,
# and is set in docker-compose.yml.

# Noise reduction is intentionally NOT in the orchestrator anymore.  The
# browser's getUserMedia stack (`noiseSuppression: true` + AEC3) cleans
# the mic stream before it ever reaches us, and doing a second pass in
# Python added ~120 ms of latency on the critical ASR path with very
# little quality gain.  If a future device pipes raw PCM in without
# browser-level NS, the cleanest fix is to denoise *on the device*
# (browser/embedded), not here.

COPY app ./app

# Port we bind on the host (network_mode: host).  Match HOST_PORT in
# docker-compose.yml.  Changed from 8000 to 8080 because port 8000 is
# commonly used by other dev servers on macOS / Linux and clashes when
# the container shares the host's network stack.
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload", "--reload-dir", "/app/app"]
