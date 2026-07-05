"""
Shared pytest fixtures for orchestrator unit tests.

Design goals:

* **No host services required** — tests run against an in-memory
  SQLite DB and never touch LM Studio, mlx-whisper, xtts-server, or
  desktop-agent.  Anything in the orchestrator that calls out gets
  stubbed at the fixture layer.

* **Fast** — every fixture is module-scoped where possible; the
  per-test reset is just a `close_thread_conn()` + a fresh
  `init_schema()` against `:memory:`.

* **Isolated** — each test runs with a fresh DB.  No row from one
  test should leak into another.

To run:
    cd orchestrator
    pip install -e ".[test]"
    pytest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Provide harmless defaults for env vars that app modules read at
# import time via ``os.environ[...]`` (bracket-subscript, no fallback).
# Without these, a clean ``pytest`` invocation crashes on collection
# because importing ``app.main``/``app.asr``/``app.llm_utils`` raises
# KeyError before any test runs.  Production deployments still fail
# loud and early if real values are missing — only the test path is
# softened here.
os.environ.setdefault("LLM_URL", "http://test-llm")
os.environ.setdefault("LLM_MODEL", "test-llm-model")
os.environ.setdefault("WHISPER_URL", "http://test-whisper")
os.environ.setdefault("WHISPER_MODEL", "test-whisper-model")
# main.py mounts StaticFiles at import time; the container default
# (/app/static) doesn't exist on a host checkout, so point at the real
# PWA directory next to the orchestrator.
os.environ.setdefault(
    "STATIC_DIR", str(Path(__file__).resolve().parent.parent.parent / "frontend")
)

# Point storage at an ephemeral file.  ``:memory:`` would be tempting
# but breaks thread-locals (each pool thread gets its own DB), so we
# use a real path under /tmp that's recreated per session.
_DB_PATH = Path(os.environ.get("TEST_DB_PATH", "/tmp/voice-assistant-test.db"))
os.environ["DB_PATH"] = str(_DB_PATH)
# Don't probe the public internet from offline-tests.
os.environ.setdefault("OFFLINE_PROBE_URL", "http://127.0.0.1:0")
# Keep voicemail audio (and any other DATA_DIR-anchored user files) out
# of /data — that's the production mount, and tests would either fail
# without it or pollute it.  /tmp is wiped naturally; the per-test
# fixture sweep handles cleanup of the inner ``voice_messages/`` dir.
os.environ["DATA_DIR_CONTAINER"] = "/tmp/voice-assistant-test-data"

# Make `from app...` resolve to orchestrator/app/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _fresh_db(monkeypatch):
    """Wipe the test DB before every test and reinit the schema.

    Uses a real file because thread-local connections + ``:memory:``
    don't compose — each thread would see its own empty DB.  A small
    on-disk file in /tmp gets us the same isolation without the
    surprises.
    """
    import shutil
    from app.storage.db import close_thread_conn
    from app.storage import init_schema
    from app.storage.voice_messages import VOICE_MESSAGES_DIR

    # Close any leftover connection so the file unlink is safe.
    close_thread_conn()
    if _DB_PATH.exists():
        _DB_PATH.unlink()
    if VOICE_MESSAGES_DIR.exists():
        shutil.rmtree(VOICE_MESSAGES_DIR, ignore_errors=True)
    init_schema()
    yield
    close_thread_conn()
    if _DB_PATH.exists():
        _DB_PATH.unlink()
    if VOICE_MESSAGES_DIR.exists():
        shutil.rmtree(VOICE_MESSAGES_DIR, ignore_errors=True)


@pytest.fixture
def make_agent_ctx():
    """Factory for AgentContext objects with sensible defaults."""
    from app.agent import AgentContext

    def _make(**overrides) -> AgentContext:
        defaults = dict(
            client_id="test-client",
            profile_id=None,
            is_authenticated=False,
            user_lang="en",
            stream_sink=None,
            progress_sink=None,
        )
        defaults.update(overrides)
        return AgentContext(**defaults)
    return _make
