"""
Schema initialisation + idempotent migrations.

Three-phase approach so older DBs are safely brought up to date:
  1. CREATE TABLE IF NOT EXISTS (never touches existing tables)
  2. ALTER TABLE ADD COLUMN for any missing columns (idempotent)
  3. CREATE INDEX IF NOT EXISTS (safe now that all referenced cols exist)
"""
from __future__ import annotations

from .db import _conn, _lock

import sqlite3


def init_schema() -> None:
    with _lock:
        c = _conn()
        try:
            # ── Phase 1: create tables ─────────────────────────────────────
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id   TEXT    NOT NULL,
                    fire_at     REAL    NOT NULL,
                    push_text   TEXT    NOT NULL,
                    fired       INTEGER NOT NULL DEFAULT 0,
                    delivered   INTEGER,
                    created_at  REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at  REAL    NOT NULL,
                    client      TEXT,
                    client_id   TEXT
                );

                CREATE TABLE IF NOT EXISTS utterances (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id        INTEGER NOT NULL REFERENCES sessions(id),
                    ts                REAL    NOT NULL,
                    audio_duration_ms INTEGER,
                    transcript        TEXT,
                    asr_ms            INTEGER,
                    llm_ms            INTEGER,
                    tool_name         TEXT,
                    tool_args         TEXT,
                    response_text     TEXT,
                    error             TEXT,
                    embedding         BLOB,
                    speaker_name      TEXT,
                    is_shared         INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id    TEXT    NOT NULL,
                    name         TEXT    NOT NULL,
                    embedding    BLOB    NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 1,
                    tts_voice    TEXT,
                    created_at   REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS custom_voices (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL,
                    wav_path    TEXT    NOT NULL,
                    created_at  REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS token_usage (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                 REAL    NOT NULL,
                    client_id          TEXT,
                    model              TEXT    NOT NULL,
                    prompt_tokens      INTEGER NOT NULL,
                    completion_tokens  INTEGER NOT NULL,
                    reasoning_tokens   INTEGER,
                    tool_name          TEXT,
                    elapsed_ms         INTEGER
                );

                -- Sprint 2: queue for invasive (risk=high_write) tool calls
                -- that the LLM wanted to run but the speaker hadn't supplied
                -- a fresh passphrase for.  Two ways to clear an entry:
                --   1. Speaker says the passphrase later in the day, agent
                --      replays the queue.
                --   2. User opens the logged-in UI and approves on the
                --      "Pending" tab.
                -- Entries time out via ``expires_at`` (default 24 h) so the
                -- queue doesn't grow unbounded.
                CREATE TABLE IF NOT EXISTS pending_actions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id    INTEGER,
                    client_id     TEXT,
                    tool_name     TEXT    NOT NULL,
                    tool_args     TEXT    NOT NULL,  -- JSON-encoded
                    summary       TEXT    NOT NULL,  -- human-readable for UI / voice readback
                    requested_at  REAL    NOT NULL,
                    expires_at    REAL    NOT NULL,
                    status        TEXT    NOT NULL DEFAULT 'pending',
                    approved_via  TEXT,
                    approved_at   REAL
                );

                -- Sprint 2: UI cookie-session store.  Created when the
                -- user logs in via /api/auth/login with their profile +
                -- passphrase; the random token lands in a HttpOnly
                -- cookie and is verified on every subsequent /api/me
                -- and /api/users/<...> request.  Server-side so we can
                -- revoke without rotating keys; the table stays tiny
                -- (one row per active browser-tab).
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token        TEXT    PRIMARY KEY,
                    profile_id   INTEGER NOT NULL,
                    created_at   REAL    NOT NULL,
                    expires_at   REAL    NOT NULL,
                    user_agent   TEXT
                );

                -- Voicemail: messages other speakers leave for the
                -- household owner via the "tell <X> that …" pipeline
                -- branch.  Audio bytes live as a WAV file on disk under
                -- /data/voice_messages/; the row stores only the
                -- relative filename so the on-disk layout can move
                -- without a migration.  `from_profile_id` is NULL when
                -- the sender's voice didn't match any enrolled profile.
                -- `from_name` / `to_name` are frozen at write time so a
                -- profile rename later doesn't muddy the audit trail.
                -- Web Push subscriptions per profile.  Each row is one
                -- browser/device that opted in to push notifications.
                -- ``endpoint`` is the push service URL the browser
                -- generated (Mozilla / Google / Apple — flavour-specific
                -- but always uniquely identifies one subscription); we
                -- key UNIQUE on it so a re-subscribe from the same tab
                -- doesn't create duplicates.  ``p256dh_key`` + ``auth_key``
                -- are the per-subscription public key + auth secret the
                -- push service uses to derive the aes128gcm content key.
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id    INTEGER NOT NULL,
                    endpoint      TEXT    NOT NULL UNIQUE,
                    p256dh_key    TEXT    NOT NULL,
                    auth_key      TEXT    NOT NULL,
                    user_agent    TEXT,
                    created_at    REAL    NOT NULL,
                    last_used_at  REAL
                );

                CREATE TABLE IF NOT EXISTS voice_messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_profile_id INTEGER REFERENCES speaker_profiles(id) ON DELETE SET NULL,
                    from_name       TEXT,
                    to_profile_id   INTEGER NOT NULL REFERENCES speaker_profiles(id) ON DELETE CASCADE,
                    to_name         TEXT    NOT NULL,
                    audio_path      TEXT    NOT NULL,
                    transcript      TEXT    NOT NULL,
                    summary         TEXT,
                    duration_ms     INTEGER NOT NULL,
                    created_at      REAL    NOT NULL,
                    listened_at     REAL,
                    replied_at      REAL,
                    reply_text      TEXT,
                    -- See _add_columns block below for the rationale.
                    -- Kept here so fresh DBs don't rely on the migration.
                    reply_delivered_to_sender_at REAL
                );
                """
            )
            # ── Phase 2: backfill columns on pre-existing tables ──────────
            _add_columns(c, [
                ("sessions",         "client_id",    "TEXT"),
                ("utterances",       "embedding",    "BLOB"),
                ("utterances",       "speaker_name", "TEXT"),
                ("utterances",       "is_shared",    "INTEGER NOT NULL DEFAULT 0"),
                ("speaker_profiles", "sample_count", "INTEGER NOT NULL DEFAULT 1"),
                # Per-speaker TTS voice override.  NULL means "fall back
                # to xtts-server default" — the UI surfaces this as the
                # blank/auto option in the voice dropdown.
                ("speaker_profiles", "tts_voice",    "TEXT"),
                # Reply-replay: timestamp the original sender has been
                # told the recipient replied to their voicemail.  NULL =
                # the sender hasn't heard the reply yet; the pipeline's
                # next-turn surfacing block reads this to decide whether
                # to inject the "by the way, X replied…" context line.
                # Done via _add_columns so existing DBs (no column yet)
                # pick the migration up on next init_schema().
                ("voice_messages",   "reply_delivered_to_sender_at", "REAL"),
            ])
            # ── Phase 3: indexes ──────────────────────────────────────────
            c.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_reminders_client_id
                    ON reminders(client_id);
                CREATE INDEX IF NOT EXISTS idx_reminders_fired
                    ON reminders(fired);
                CREATE INDEX IF NOT EXISTS idx_sessions_client_id
                    ON sessions(client_id);
                CREATE INDEX IF NOT EXISTS idx_speaker_profiles_client_id
                    ON speaker_profiles(client_id);
                CREATE INDEX IF NOT EXISTS idx_utterances_session
                    ON utterances(session_id);
                CREATE INDEX IF NOT EXISTS idx_utterances_ts
                    ON utterances(ts);
                CREATE INDEX IF NOT EXISTS idx_token_usage_ts
                    ON token_usage(ts);
                CREATE INDEX IF NOT EXISTS idx_token_usage_client_id
                    ON token_usage(client_id);
                CREATE INDEX IF NOT EXISTS idx_token_usage_tool
                    ON token_usage(tool_name);
                CREATE INDEX IF NOT EXISTS idx_pending_actions_profile
                    ON pending_actions(profile_id);
                CREATE INDEX IF NOT EXISTS idx_pending_actions_client
                    ON pending_actions(client_id);
                CREATE INDEX IF NOT EXISTS idx_pending_actions_status
                    ON pending_actions(status);
                CREATE INDEX IF NOT EXISTS idx_pending_actions_expires
                    ON pending_actions(expires_at);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_profile
                    ON auth_sessions(profile_id);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires
                    ON auth_sessions(expires_at);

                -- ── Composite indexes for hot multi-column paths ──────────
                -- The existing single-column indexes above cover the
                -- simplest filters but SQLite can use only ONE index per
                -- table per query.  These composites match the actual
                -- WHERE+ORDER-BY shape of the four queries that run on
                -- every voice turn or stats refresh, so the planner
                -- doesn't fall back to a table scan once a table grows
                -- past tens of thousands of rows.
                CREATE INDEX IF NOT EXISTS idx_sessions_client_started
                    ON sessions(client_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_utterances_session_ts
                    ON utterances(session_id, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_token_usage_ts_tool
                    ON token_usage(ts, tool_name);
                CREATE INDEX IF NOT EXISTS idx_token_usage_ts_client
                    ON token_usage(ts, client_id);
                CREATE INDEX IF NOT EXISTS idx_pending_actions_status_expires
                    ON pending_actions(status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_pending_actions_status_profile_client
                    ON pending_actions(status, profile_id, client_id);
                CREATE INDEX IF NOT EXISTS idx_reminders_client_fired_fire
                    ON reminders(client_id, fired, fire_at);
                CREATE INDEX IF NOT EXISTS idx_reminders_client_delivered
                    ON reminders(client_id, fired, delivered, fire_at);
                CREATE INDEX IF NOT EXISTS idx_speaker_profiles_client_name
                    ON speaker_profiles(client_id, name);

                -- Voicemail: the dominant read pattern is "unread for
                -- recipient" (inbox panel + LLM `inbox_list` tool), and
                -- the secondary one is "all for recipient ordered by
                -- recency".  One composite covers both — SQLite can
                -- use a prefix scan on (to_profile_id) and the listened
                -- / created_at suffix for the secondary sort.
                CREATE INDEX IF NOT EXISTS idx_voice_messages_to_listened
                    ON voice_messages(to_profile_id, listened_at);
                CREATE INDEX IF NOT EXISTS idx_voice_messages_to_created
                    ON voice_messages(to_profile_id, created_at DESC);

                -- Web Push: lookup is always "every subscription for one
                -- profile" (when broadcasting a voicemail) — UNIQUE on
                -- endpoint already gives us the secondary lookup path.
                CREATE INDEX IF NOT EXISTS idx_push_subs_profile
                    ON push_subscriptions(profile_id);

                -- ── Personal item store ───────────────────────────────
                -- Hierarchical folders (kind='folder') and checklists
                -- (kind='checklist').  Unbounded depth via parent_id
                -- self-reference; subtree queries use a recursive CTE.
                -- Soft-deleted categories (deleted_at IS NOT NULL) are
                -- hidden from all list views — their items keep the
                -- category_id but become unreachable until restored.
                CREATE TABLE IF NOT EXISTS categories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_profile_id INTEGER NOT NULL,
                    parent_id        INTEGER REFERENCES categories(id) ON DELETE CASCADE,
                    name             TEXT    NOT NULL,
                    slug             TEXT    NOT NULL,
                    kind             TEXT    NOT NULL DEFAULT 'folder',
                    sort_order       INTEGER NOT NULL DEFAULT 0,
                    created_at       REAL    NOT NULL,
                    deleted_at       REAL
                );

                -- Links, text snippets, videos, shorts, screenshots, and
                -- checklist items.  Media files live on disk under
                -- /data/items/<id>.<ext>; only the extension (or relative
                -- path) is stored in media_path.  Embeddings are populated
                -- async by the ingest pipeline — items are visible
                -- immediately but not semantically searchable until the
                -- BLOB arrives (~1-2 s after save).
                CREATE TABLE IF NOT EXISTS items (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_profile_id      INTEGER NOT NULL,
                    created_by_profile_id INTEGER NOT NULL,
                    category_id           INTEGER REFERENCES categories(id) ON DELETE SET NULL,
                    kind                  TEXT    NOT NULL,
                    status                TEXT    NOT NULL DEFAULT 'active',
                    title                 TEXT,
                    summary               TEXT,
                    url                   TEXT,
                    media_path            TEXT,
                    source_meta           TEXT,
                    body                  TEXT,
                    embedding             BLOB,
                    sort_order            REAL    NOT NULL DEFAULT 0.0,
                    completed_at          REAL,
                    created_at            REAL    NOT NULL,
                    deleted_at            REAL
                );

                -- Per-category access grants for household sharing.
                -- Visibility is logically recursive: granting read on a
                -- folder implicitly exposes its subtree (implemented in
                -- list_items via CTE, not enforced by a DB constraint).
                CREATE TABLE IF NOT EXISTS category_shares (
                    category_id  INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
                    profile_id   INTEGER NOT NULL,
                    permission   TEXT    NOT NULL DEFAULT 'read',
                    granted_at   REAL    NOT NULL,
                    PRIMARY KEY (category_id, profile_id)
                );

                -- FTS5 virtual table for BM25 full-text search on items.
                -- content='items' means rows are read through from the
                -- items table; the triggers below keep the index in sync.
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                    title, summary, body,
                    content='items',
                    content_rowid='id'
                );

                -- FTS5 sync triggers.  IF NOT EXISTS avoids duplicate
                -- registration on repeated init_schema() calls.
                CREATE TRIGGER IF NOT EXISTS items_fts_ai
                AFTER INSERT ON items BEGIN
                    INSERT INTO items_fts(rowid, title, summary, body)
                    VALUES (new.id, new.title, new.summary, new.body);
                END;

                CREATE TRIGGER IF NOT EXISTS items_fts_ad
                AFTER DELETE ON items BEGIN
                    INSERT INTO items_fts(items_fts, rowid, title, summary, body)
                    VALUES ('delete', old.id, old.title, old.summary, old.body);
                END;

                CREATE TRIGGER IF NOT EXISTS items_fts_au
                AFTER UPDATE ON items BEGIN
                    INSERT INTO items_fts(items_fts, rowid, title, summary, body)
                    VALUES ('delete', old.id, old.title, old.summary, old.body);
                    INSERT INTO items_fts(rowid, title, summary, body)
                    VALUES (new.id, new.title, new.summary, new.body);
                END;

                -- Indexes for the item store.
                CREATE INDEX IF NOT EXISTS idx_categories_owner
                    ON categories(owner_profile_id);
                CREATE INDEX IF NOT EXISTS idx_categories_parent
                    ON categories(parent_id);
                CREATE INDEX IF NOT EXISTS idx_categories_owner_parent
                    ON categories(owner_profile_id, parent_id);
                CREATE INDEX IF NOT EXISTS idx_items_owner
                    ON items(owner_profile_id);
                CREATE INDEX IF NOT EXISTS idx_items_category
                    ON items(category_id);
                CREATE INDEX IF NOT EXISTS idx_items_owner_category
                    ON items(owner_profile_id, category_id);
                -- Items: hot path is ``list_items`` with
                -- ``WHERE owner_profile_id=? AND deleted_at IS NULL
                -- ORDER BY created_at DESC`` (plus trash view with
                -- ``deleted_at IS NOT NULL``).  The 3-col composite
                -- covers WHERE + ORDER BY in one go — the planner
                -- reads rows in index order without a sort step.
                -- Supersedes the older 2-col idx_items_owner_deleted.
                CREATE INDEX IF NOT EXISTS idx_items_owner_deleted_created
                    ON items(owner_profile_id, deleted_at, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_items_owner_created
                    ON items(owner_profile_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_items_kind
                    ON items(owner_profile_id, kind);
                CREATE INDEX IF NOT EXISTS idx_category_shares_profile
                    ON category_shares(profile_id);

                -- Housekeeping: drop dead / superseded indexes.
                --   idx_items_owner_deleted    — replaced by the 3-col
                --     idx_items_owner_deleted_created above.
                --   idx_custom_voices_name     — no call site queries
                --     ``WHERE name=?``; voices are looked up by id only.
                DROP INDEX IF EXISTS idx_items_owner_deleted;
                DROP INDEX IF EXISTS idx_custom_voices_name;
                """
            )
        finally:
            c.close()


def _add_columns(
    conn: sqlite3.Connection,
    columns: list[tuple[str, str, str]],
) -> None:
    """ALTER TABLE ADD COLUMN for each (table, col, coldef) that doesn't exist yet."""
    for table, col, coldef in columns:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coldef}")
        except sqlite3.OperationalError:
            pass  # Column already exists — normal for existing databases.
