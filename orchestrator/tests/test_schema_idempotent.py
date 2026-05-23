"""
``init_schema`` must be idempotent.

Two scenarios:
  • First run on an empty DB → creates everything.
  • Second run on a populated DB → no-op, no exceptions, no duplicate
    columns / indexes.

Critical because _add_columns swallows OperationalError to handle
"column already exists" gracefully; a typo could silently skip a
real migration.  We verify the columns we added actually exist.
"""
from __future__ import annotations

from app.storage import init_schema
from app.storage.db import _conn


def _columns(table: str) -> set[str]:
    c = _conn()
    rows = c.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _indexes(table: str) -> set[str]:
    c = _conn()
    rows = c.execute(
        f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table}'"
    ).fetchall()
    return {r[0] for r in rows}


async def test_init_schema_idempotent_no_raise():
    """Running init_schema twice should not raise."""
    init_schema()  # autouse fixture already ran it once
    init_schema()
    init_schema()


async def test_added_columns_exist():
    """Columns added via _add_columns should be present after init."""
    cols = _columns("utterances")
    assert "speaker_name" in cols, "speaker_name column missing"
    assert "is_shared" in cols, "is_shared column missing"

    cols = _columns("speaker_profiles")
    assert "sample_count" in cols
    assert "tts_voice" in cols

    cols = _columns("voice_messages")
    assert "reply_delivered_to_sender_at" in cols, (
        "voicemail reply-replay column missing"
    )


async def test_composite_indexes_present():
    """The composite indexes added in the perf pass."""
    assert "idx_utterances_session_ts" in _indexes("utterances")
    assert "idx_token_usage_ts_tool" in _indexes("token_usage")
    assert "idx_pending_actions_status_expires" in _indexes("pending_actions")


async def test_pragmas_active():
    """WAL + synchronous=NORMAL + busy_timeout from db.py setup."""
    c = _conn()
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    # synchronous=NORMAL is integer 1
    assert c.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
