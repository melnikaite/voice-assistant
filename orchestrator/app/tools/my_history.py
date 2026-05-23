"""
my_history — semantic search over the speaker's own past Q&A.

This is the read-side counterpart of the "day's diary" idea: the user
asks "what did we talk about yesterday about X", "what did I search
for last week about Y", and we pull matching past exchanges via
embedding similarity.

Why a dedicated tool when the agent loop ALREADY injects semantic
memory as ``memory_context``?

* That injection is implicit — the LLM may or may not weave it into
  its answer.  An explicit tool gives the user a direct "search my
  history for X" knob that's predictable.
* It is scoped to a STRONGER identity check than memory_context: by
  ``profile_id`` (resemblyzer + cookie) AND by ``speaker_name`` so
  household members can't accidentally fish each other's past topics.
  memory_context is best-effort and falls back to client-wide search
  when nothing matched.
* It returns the matches verbatim rather than letting the LLM
  paraphrase, which is what the user usually wants ("just tell me what
  I said").

Risk = ``read``.  Returning your own past exchanges is fine without a
passphrase — the data isn't more sensitive than the speaker_name
identification we already do, and gating it would make the feature
useless for "what did I search for today".
"""
from __future__ import annotations

import logging
import time

from .. import memory
from ..i18n import t
from ..storage import get_candidate_utterances
from .base import ToolResult, tool

log = logging.getLogger(__name__)


# Default lookback for "what did we talk about" style queries.  30 d
# matches the semantic memory lookback default — keeps things consistent.
# The LLM can pass a smaller window for "today" / "this past week" via
# the param.
_DEFAULT_DAYS = 30
_MAX_RESULTS = 5
_SIMILARITY_THRESHOLD = 0.55  # looser than memory_context (0.72) — search
                              # is explicit, so the user expects matches
                              # even on loose semantic overlap


@tool(
    name="my_history",
    description=(
        "Search the SPEAKER's own past conversations (transcript + "
        "assistant reply) by topic.  Use when the user asks 'what did we "
        "discuss yesterday about X', 'what did I search for last week "
        "about Y', 'remind me how we discussed Z', 'what did we discuss "
        "about ...' in any supported language.\n"
        "Returns up to a handful of matching past exchanges, scoped to "
        "this speaker (other household members are excluded).  Different "
        "from `read_memory` — that returns the freeform notes file; this "
        "searches past LIVE Q&A.\n"
        "If you just want to inject context implicitly, leave it to the "
        "system; call this tool only when the user EXPLICITLY asks to "
        "retrieve past discussion."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic / keywords to search for.  Free-form.",
            },
            "days": {
                "type": "integer",
                "description": (
                    f"Lookback in days (default {_DEFAULT_DAYS}).  Use a "
                    "small value (1, 7) for 'today' / 'this past week'."
                ),
            },
        },
        "required": ["query"],
    },
    risk="read",
)
async def my_history(
    query: str,
    days: int | None = None,
    *,
    ctx,
) -> ToolResult:
    profile_id = getattr(ctx, "profile_id", None)
    client_id = getattr(ctx, "client_id", None)
    progress = getattr(ctx, "progress_sink", None)
    lang = getattr(ctx, "user_lang", None)

    if profile_id is None:
        # Speaker not identified — refuse rather than leak everyone's
        # history.  Frontends without speaker ID can fall back to
        # `read_memory` for unstructured notes.
        return ToolResult(text=t("history.no_profile", lang), data={"error": "no_profile"})

    if not memory.EMBEDDING_ENABLED:
        return ToolResult(text=t("history.empty", lang), data={"error": "no_embeddings"})

    # Resolve speaker_name from the profile so we can scope the SQL
    # query.  Cheap — one row by id.
    try:
        from ..storage.db import _conn, _lock
        with _lock:
            c = _conn()
            try:
                row = c.execute(
                    "SELECT name FROM speaker_profiles WHERE id=?",
                    (profile_id,),
                ).fetchone()
            finally:
                c.close()
        speaker_name = row[0] if row else None
    except Exception as exc:
        log.warning("my_history: could not resolve speaker name: %s", exc)
        speaker_name = None

    if not client_id:
        return ToolResult(text=t("history.empty", lang), data={"error": "no_client_id"})

    if progress is not None:
        await progress("search", None)

    lookback_days = max(1, int(days or _DEFAULT_DAYS))
    since_ts = time.time() - lookback_days * 86400
    try:
        query_vec = await memory.embed_query(query)
        candidates = await get_candidate_utterances(
            client_id, since_ts, limit=300, speaker_name=speaker_name
        )
        if not candidates:
            return ToolResult(
                text=t("history.empty", lang),
                data={"days": lookback_days, "speaker": speaker_name, "count": 0},
            )
        hits = memory.retrieve(
            query_vec,
            candidates,
            top_k=_MAX_RESULTS,
            threshold=_SIMILARITY_THRESHOLD,
        )
    except Exception as exc:
        log.exception("my_history: retrieval failed")
        return ToolResult(text=t("history.empty", lang), data={"error": str(exc)})

    if not hits:
        return ToolResult(
            text=t("history.empty", lang),
            data={"days": lookback_days, "speaker": speaker_name, "count": 0},
        )

    # Format for voice: short fragments, current speaker's question
    # paraphrased into an "I asked …" frame so it sounds natural when
    # read back.  We surface the top 3 in the spoken reply; the full
    # set (up to 5) goes into the structured `data` for the UI.
    spoken = [
        f"«{(h.get('transcript') or '').strip()[:90]}» → «{(h.get('response_text') or '').strip()[:120]}»"
        for h in hits
    ]
    if not spoken:
        return ToolResult(
            text=t("history.empty", lang),
            data={"query": query, "days": lookback_days, "speaker": speaker_name, "count": 0},
        )
    return ToolResult(
        text=t("history.items_header", lang, items="; ".join(spoken[:3])),
        data={
            "query": query,
            "days": lookback_days,
            "speaker": speaker_name,
            "count": len(hits),
            "items": hits,
        },
    )
