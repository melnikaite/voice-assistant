"""
inbox — voicemail recipient tools (list / read / summary / reply).

Every tool here filters by ``ctx.profile_id`` server-side, so a child
speaker who asks "any messages for me?" only sees messages where
``to_profile_id`` matches their own — never the parent's inbox.  Voice
identification IS the auth here: if the recognised voice is yours,
you have the right to read your own mail.  No passphrase needed
(read-tier across the board).

What each tool does:

  • ``inbox_list``     — counts + line-per-message snippets.  The LLM
                         can call this to answer "what did people send me?".
  • ``inbox_read``     — full transcript + a control signal (in the
                         tool's ``data``) that tells the WS layer to
                         play back the ORIGINAL audio next.  Side
                         effect: marks the message ``listened_at``.
  • ``inbox_summary``  — LLM-generated 1-2 sentence summary, cached
                         per message (re-asks are free).  Returns
                         ``tts_voice_override`` pointing at the
                         sender's XTTS voice so the reply is spoken
                         in their voice if they have a cloned one.
  • ``inbox_reply``    — stores a textual reply against the message.
                         Out-of-scope today: delivery (no push to the
                         sender) — the host typically calls back.

Why no "high_write" risk gate: every mutation is scoped to the
caller's own profile (verified by SQL), and replies are reversible
trivially.  We err on the side of letting the recipient act fast
on their own inbox without re-authenticating.
"""
from __future__ import annotations

import logging
import time

from .. import llm_utils
from ..i18n import t
from ..storage import (
    count_unread_voicemail,
    get_speaker_profiles,
    get_voice_message,
    list_voicemail,
    mark_voicemail_listened,
    save_voicemail_reply,
    set_voicemail_summary,
)
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


# How many messages to include in `inbox_list` by default.  The
# /api/voicemail HTTP endpoint takes its own limit; this is the LLM-
# facing cap so spoken summaries don't drag on.
DEFAULT_LIST_LIMIT = 5
# Hard cap to protect the LLM context window when the user explicitly
# asks for "all of them" — we still truncate.
MAX_LIST_LIMIT = 20
# Hard cap on the snippet length in spoken/textual list output.
SNIPPET_CHARS = 80


def _relative_when(ts: float, lang: str | None) -> str:
    """Render "N min ago" / "N h ago" / "N d ago" in the recipient's lang."""
    elapsed = max(0.0, time.time() - ts)
    if elapsed < 60:
        return t("inbox.rel_time_seconds", lang)
    if elapsed < 3600:
        return t("inbox.rel_time_minutes", lang, n=int(elapsed // 60))
    if elapsed < 86400:
        return t("inbox.rel_time_hours", lang, n=int(elapsed // 3600))
    return t("inbox.rel_time_days", lang, n=int(elapsed // 86400))


def _from_label(row: dict, lang: str | None) -> str:
    """Pick a human label for the sender.

    Prefer the frozen ``from_name`` stored at write time — survives a
    later profile rename / delete.  Fall back to a localised "guest"
    when neither name nor profile was captured (anonymous walk-up).
    """
    name = (row.get("from_name") or "").strip()
    if name:
        return name
    return t("inbox.guest_sender", lang)


def _shorten(s: str, n: int = SNIPPET_CHARS) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _auth_required(lang: str | None) -> ToolResult:
    return ToolResult(
        text=t("inbox.auth_required", lang),
        data={"error": "auth_required"},
    )


# ── inbox_list ──────────────────────────────────────────────────────────


@tool(
    name="inbox_list",
    description=(
        "List voicemail messages addressed to the speaker.  Use when the "
        "user asks 'what did people send me?', 'any messages?', 'read my "
        "inbox', or 'what's in my inbox' in any supported language.  "
        "Returns counts + per-message snippets with the sender name and "
        "how long ago each arrived.  Pass `unread_only=true` to filter "
        "to messages the speaker hasn't heard yet (the default is true "
        "— silent ones first).  The numeric id printed in each line is "
        "what `inbox_read` / `inbox_reply` / `inbox_summary` take."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "unread_only": {
                "type": "boolean",
                "description": (
                    "True: only messages with listened_at IS NULL.  "
                    "False: all recent messages (default 5)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "How many messages to fetch (1-20, default 5).",
            },
        },
        "required": [],
    },
    risk="read",
)
async def inbox_list(
    *, ctx, unread_only: bool = True, limit: int = DEFAULT_LIST_LIMIT,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    if cx.profile_id is None:
        return _auth_required(cx.user_lang)
    limit = max(1, min(int(limit), MAX_LIST_LIMIT))
    rows = await list_voicemail(
        cx.profile_id, unread_only=bool(unread_only), limit=limit
    )
    if not rows:
        return ToolResult(
            text=t("inbox.empty", cx.user_lang),
            data={"count": 0, "messages": []},
        )
    header_key = (
        "inbox.list_header_unread" if unread_only else "inbox.list_header_all"
    )
    lines = [t(header_key, cx.user_lang, n=len(rows))]
    for idx, r in enumerate(rows, 1):
        lines.append(t(
            "inbox.list_item", cx.user_lang,
            idx=idx,
            from_name=_from_label(r, cx.user_lang),
            when=_relative_when(r["created_at"], cx.user_lang),
            snippet=_shorten(r["transcript"]),
        ))
        # Tag the row dict with the 1-based idx so the LLM can refer
        # back by index in chained calls (e.g. "read the second one")
        # — we keep the real id as the canonical reference though.
        r["idx"] = idx
    return ToolResult(
        text="\n".join(lines),
        data={"count": len(rows), "messages": rows},
    )


# ── inbox_read ──────────────────────────────────────────────────────────


@tool(
    name="inbox_read",
    description=(
        "Play back one voicemail message in the sender's actual voice — "
        "the original recording.  The LLM speaks a short intro ('Message "
        "from X:') and the WS layer streams the original audio next.  "
        "Use when the user says 'play it', 'listen', 'give me the first "
        "one'; if the user just wants the gist use `inbox_summary` "
        "instead.  Marks the message as listened."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": (
                    "Numeric id of the message — get it from `inbox_list`."
                ),
            },
        },
        "required": ["message_id"],
    },
    risk="read",
)
async def inbox_read(*, ctx, message_id: int) -> ToolResult:
    cx = unwrap_ctx(ctx)
    if cx.profile_id is None:
        return _auth_required(cx.user_lang)
    row = await get_voice_message(int(message_id))
    if row is None or row["to_profile_id"] != cx.profile_id:
        # Same response on "not found" and "wrong recipient" so we
        # don't leak the existence of someone else's mail.
        return ToolResult(
            text=t("inbox.not_found", cx.user_lang),
            data={"error": "not_found"},
        )
    # Mark listened best-effort.  Idempotent on the second call.
    try:
        await mark_voicemail_listened(int(message_id), cx.profile_id)
    except Exception:
        log.warning("inbox_read: mark_listened failed", exc_info=True)
    text = t(
        "inbox.read_intro", cx.user_lang,
        from_name=_from_label(row, cx.user_lang),
    )
    # The `voicemail_play_id` key signals the WS layer to push a
    # ``voicemail_play`` event after the intro reply — the frontend
    # fetches /api/voicemail/<id>/audio and plays it.  We don't try
    # to send the audio bytes via WS (binary frames are reserved for
    # mic data); HTTP serves the wav.
    return ToolResult(
        text=text,
        data={
            "voicemail_play_id": row["id"],
            "from_name": row.get("from_name"),
            "from_profile_id": row.get("from_profile_id"),
            "duration_ms": row.get("duration_ms"),
            "transcript": row.get("transcript"),
        },
    )


# ── inbox_summary ──────────────────────────────────────────────────────


_SUMMARY_PROMPT = (
    "Summarise the following voicemail in ONE short sentence (max ~20 "
    "words), in the same language the message is in.  Keep the original "
    "speaker's intent and tone; do NOT add advice, opinion, or "
    "interpretation.  Output ONLY the summary text — no preamble.\n\n"
    "Message: \"{transcript}\""
)


async def _llm_summarise(transcript: str, client_id: str | None) -> str:
    """One-shot summary call.  Cheap (≤ ~200 prompt tokens, ~50 out)."""
    res = await llm_utils.chat(
        messages=[
            {
                "role": "user",
                "content": _SUMMARY_PROMPT.format(transcript=transcript),
            },
        ],
        temperature=0.2,
        max_tokens=120,
        client_id=client_id,
        tool_name="inbox_summary",
    )
    msg = (res.get("message") or {}).get("content") or ""
    return msg.strip().strip("«»\"").strip()


@tool(
    name="inbox_summary",
    description=(
        "Speak a 1-2 sentence summary of a voicemail, IN THE SENDER's "
        "voice.  Use when the user says 'what's the message about', "
        "'summary', 'briefly', or 'shorten it'.  The first call generates "
        "the summary via the LLM and caches it on the row; subsequent "
        "calls are free.  If the user wants the original recording, "
        "route to `inbox_read`."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "Numeric id of the message to summarise.",
            },
        },
        "required": ["message_id"],
    },
    risk="read",
)
async def inbox_summary(*, ctx, message_id: int) -> ToolResult:
    cx = unwrap_ctx(ctx)
    if cx.profile_id is None:
        return _auth_required(cx.user_lang)
    row = await get_voice_message(int(message_id))
    if row is None or row["to_profile_id"] != cx.profile_id:
        return ToolResult(
            text=t("inbox.not_found", cx.user_lang),
            data={"error": "not_found"},
        )
    summary = (row.get("summary") or "").strip()
    if not summary:
        try:
            summary = await _llm_summarise(row["transcript"], cx.client_id)
        except Exception:
            log.exception("inbox_summary: LLM call failed")
            # Graceful degradation: fall back to the raw transcript so
            # the user still gets the gist.  No cache write.
            summary = row["transcript"]
        else:
            try:
                await set_voicemail_summary(int(message_id), summary)
            except Exception:
                log.warning("inbox_summary: cache write failed", exc_info=True)
    # Resolve the sender's cloned XTTS voice if any.  When unknown
    # (guest / no clone enrolled), we leave the override unset so the
    # host hears their own default voice — better than a silent fail.
    sender_voice: str | None = None
    if row.get("from_profile_id") is not None and cx.client_id:
        try:
            profiles = await get_speaker_profiles(cx.client_id)
            sender_voice = next(
                (tv for pid, _, _, _, tv in profiles
                 if pid == row["from_profile_id"]),
                None,
            )
        except Exception:
            log.warning("inbox_summary: voice lookup failed", exc_info=True)
    # Mark as listened — hearing the summary counts.
    try:
        await mark_voicemail_listened(int(message_id), cx.profile_id)
    except Exception:
        pass
    text = t(
        "inbox.summary_intro", cx.user_lang,
        from_name=_from_label(row, cx.user_lang),
        summary=summary,
    )
    return ToolResult(
        text=text,
        data={
            "message_id": int(message_id),
            "summary": summary,
            "from_name": row.get("from_name"),
        },
        tts_voice_override=sender_voice,
    )


# ── inbox_reply ─────────────────────────────────────────────────────────


@tool(
    name="inbox_reply",
    description=(
        "Save a textual reply to a voicemail message — the original sender "
        "can see it in their inbox panel.  Use when the user says 'reply "
        "to her that I'll be there soon', 'reply to John: I'll call back', "
        "after a preceding `inbox_read` / `inbox_summary` (so message_id "
        "is in context).  Does NOT push the reply to the sender's device "
        "— today this is a stored-reply only."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "Numeric id of the message you're replying to.",
            },
            "reply": {
                "type": "string",
                "description": "The reply text — spoken word-for-word.",
            },
        },
        "required": ["message_id", "reply"],
    },
    risk="read",
)
async def inbox_reply(*, ctx, message_id: int, reply: str) -> ToolResult:
    cx = unwrap_ctx(ctx)
    if cx.profile_id is None:
        return _auth_required(cx.user_lang)
    reply = (reply or "").strip()
    if not reply:
        return ToolResult(
            text=t("inbox.not_found", cx.user_lang),
            data={"error": "empty_reply"},
        )
    row = await get_voice_message(int(message_id))
    if row is None or row["to_profile_id"] != cx.profile_id:
        return ToolResult(
            text=t("inbox.not_found", cx.user_lang),
            data={"error": "not_found"},
        )
    ok = await save_voicemail_reply(int(message_id), cx.profile_id, reply)
    if not ok:
        return ToolResult(
            text=t("inbox.not_found", cx.user_lang),
            data={"error": "not_found"},
        )
    return ToolResult(
        text=t(
            "inbox.reply_recorded", cx.user_lang,
            to_name=_from_label(row, cx.user_lang),
        ),
        data={"message_id": int(message_id), "reply": reply},
    )
