"""Shared SQLite connection management — thread-local + WAL.

Why thread-local connections rather than the previous global-mutex
pattern:

* SQLite in WAL mode supports **multiple concurrent readers and one
  writer** at the engine level.  A global `threading.Lock()` around
  every read serialised everything onto one thread, turning a
  dashboard refresh into a queue behind every voice turn.
* `sqlite3.Connection` objects are **not** thread-safe by default, so
  we keep one connection per thread (via `threading.local`).  Each
  connection enables WAL + ``synchronous=NORMAL`` + a generous
  ``busy_timeout`` on first use — the busy-timeout makes SQLITE_BUSY
  on concurrent writers transparent without an application-level lock.
* All storage queries run on the asyncio default-thread-pool
  executor (via ``asyncio.to_thread``), so each pooled thread caches
  exactly one connection for its lifetime.  The pool is bounded
  (default 40 on CPython) — well below SQLite's connection limit.

Backwards compatibility: the module still exports ``_conn`` and
``_lock``.  ``_conn()`` now returns the per-thread connection (callers
must NOT `.close()` it — that would close the thread-local).
``_lock`` is kept as a context-manager-compatible no-op so existing
``with _lock:`` blocks continue to work without edits during the
migration.  New code can drop the ``with _lock:`` entirely.

Test escape hatch: :func:`close_thread_conn` lets tests reset the
connection between cases (mostly relevant for ``:memory:`` DBs where
the schema is per-connection).
"""
from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH: str = os.environ.get("DB_PATH", "/data/assistant.db")

# How old a session's history can be for auto-restore on reconnect (seconds).
# Default: 30 minutes.
HISTORY_RESUME_MAX_AGE_S: float = float(
    os.environ.get("HISTORY_RESUME_MAX_AGE_S", "1800")
)

# Busy-wait this many ms on SQLITE_BUSY before raising — enough to let
# any other writer commit (writers in WAL are millisecond-scale).
_BUSY_TIMEOUT_MS = int(os.environ.get("SQLITE_BUSY_TIMEOUT_MS", "5000"))

_tls = threading.local()


def _conn() -> sqlite3.Connection:
    """Return this thread's SQLite connection, opening one lazily.

    The connection is configured ONCE on first use:
      • WAL journal mode — multiple readers + one writer.
      • ``synchronous=NORMAL`` — durability between checkpoints, big
        write speedup vs FULL.  In WAL this loses at most the last
        committed transaction on hard power loss (acceptable for a
        voice-assistant log DB).
      • ``busy_timeout=5000`` — sleep + retry on SQLITE_BUSY rather
        than raising, so concurrent writers don't need app-level
        coordination.
      • ``isolation_level=None`` — autocommit; every statement is its
        own transaction, matching the existing call sites which never
        BEGIN/COMMIT explicitly.

    Callers MUST NOT call ``.close()`` on the returned connection —
    closing would terminate the thread-local cache.  Use
    :func:`close_thread_conn` from test teardown only.
    """
    proxy: "_CachedConnection | None" = getattr(_tls, "conn", None)
    if proxy is not None:
        return proxy  # type: ignore[return-value]
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False)
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA synchronous=NORMAL")
    raw.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    # foreign_keys=ON is harmless even though we don't declare FK
    # constraints today — flips on if/when we add them.
    raw.execute("PRAGMA foreign_keys=ON")
    proxy = _CachedConnection(raw)
    _tls.conn = proxy
    log.debug("db: opened thread-local connection in %s", threading.current_thread().name)
    return proxy  # type: ignore[return-value]


def close_thread_conn() -> None:
    """Close this thread's connection.  Test-teardown helper.

    Drops the proxy AND closes the underlying ``sqlite3.Connection``
    — distinct from a callsite ``.close()`` (which is a no-op via
    the proxy).
    """
    proxy: "_CachedConnection | None" = getattr(_tls, "conn", None)
    if proxy is None:
        return
    try:
        proxy._raw.close()
    finally:
        _tls.conn = None


# Backwards-compat shim.  All existing storage modules wrap their
# SQL in ``with _lock: ... c = _conn(); ...; c.close()``.  The lock
# is no longer needed (busy_timeout serialises writers; readers run
# concurrently) and `.close()` would kill the thread-local — both
# patterns should disappear from new code.  We keep this as a no-op
# context manager so the existing call sites keep working until they
# are migrated module-by-module.
class _NoopLock:
    """Context-manager shim that replaces the old global threading.Lock."""

    def __enter__(self):
        return self

    def __exit__(self, *a) -> bool:
        return False

    @contextlib.contextmanager
    def __call__(self):  # pragma: no cover - not used today
        yield


_lock = _NoopLock()


# Compat shim: legacy storage call sites still wrap each query with
# ``try: ...; finally: c.close()``.  Closing the thread-local
# connection would force a re-open on every query, undoing the whole
# point.  Python 3.12 made ``sqlite3.Connection`` immutable so we
# can't monkey-patch ``.close()`` — instead we wrap the connection
# in a tiny proxy that delegates every attribute except ``close``,
# which becomes a no-op on the cached instance.  Tests that need to
# fully close (e.g. ``:memory:`` reset) call :func:`close_thread_conn`.


class _CachedConnection:
    """Attribute-forwarding proxy around a ``sqlite3.Connection``.

    All access goes to ``self._raw`` except ``.close()``, which is a
    no-op (the real close happens in :func:`close_thread_conn`).
    This keeps every existing storage call site working without edits
    while preserving the connection-cache invariant.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: sqlite3.Connection):
        object.__setattr__(self, "_raw", raw)

    def close(self) -> None:  # no-op
        return None

    def __getattr__(self, name: str):
        return getattr(self._raw, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_raw":
            object.__setattr__(self, name, value)
        else:
            setattr(self._raw, name, value)

    def __enter__(self):
        return self._raw.__enter__()

    def __exit__(self, *a):
        return self._raw.__exit__(*a)
