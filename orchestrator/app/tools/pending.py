"""
pending — manage the deferred-action queue from the LLM side.

When a ``high_write`` tool call is made without an active passphrase
auth window, ``agent._execute_one`` enqueues it in ``pending_actions``
instead of executing.  Later — when the user says the passphrase or
opens the UI — those entries are reviewed.

This file gives the LLM the *voice* path to that queue:

  • ``list_pending``    (tier-1) — read out what's queued.
  • ``approve_pending`` (tier-2, but auth is already required to be in
                        this code path, so it's effectively low_write
                        once gating let us through) — execute one
                        queued action by id.
  • ``reject_pending``  (tier-2 by the same logic) — drop one queued
                        action without running it.

We deliberately do NOT implement the actual replay-execution here:
that would couple this tool to every other tool's handler.  Instead
this tool just flips status to ``approved``/``rejected``; a background
task in the orchestrator reads approved rows and dispatches them via
the normal tool registry.  (Today the agent loop only flips status —
the executor task is wired in a follow-up commit; for the smoke test
flipping status is enough to verify the queue contract.)
"""
from __future__ import annotations

import logging
import time

from ..i18n import t
from ..storage import (
    get_pending_action,
    list_pending_actions,
    mark_approved,
    mark_rejected,
)
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


def _fmt_summary(action: dict, now: float, lang: str | None) -> str:
    """Short spoken description for one queued action."""
    ttl_s = action["expires_at"] - now
    if ttl_s < 3600:
        age = t("pending.age.expiring_soon", lang)
    elif ttl_s < 86400:
        age = t("pending.age.hours_left", lang, hours=int(ttl_s / 3600))
    else:
        age = t("pending.age.valid", lang)
    return f"{action['summary']} ({age})"


@tool(
    name="list_pending",
    description=(
        "Read out the speaker's queue of deferred (not-yet-approved) "
        "actions — things that were requested but parked because the "
        "speaker hadn't authenticated.  Use when the user asks 'what's "
        "in my queue', 'what actions are waiting for approval'."
    ),
    params_schema={"type": "object", "properties": {}, "required": []},
    risk="read",
)
async def list_pending(*, ctx) -> ToolResult:
    cx = unwrap_ctx(ctx)
    items = await list_pending_actions(profile_id=cx.profile_id, client_id=cx.client_id)
    if not items:
        return ToolResult(text=t("pending.none", cx.user_lang), data={"count": 0})
    now = time.time()
    entries = [_fmt_summary(a, now, cx.user_lang) for a in items]
    return ToolResult(
        text=t("pending.list", cx.user_lang, count=len(entries), body="; ".join(entries)),
        data={"count": len(items), "actions": items},
    )


@tool(
    name="approve_pending",
    description=(
        "Approve a deferred action by id.  Use when the user explicitly "
        "says 'approve', 'execute the deferred X', 'confirm number N' "
        "AFTER having said the passphrase in this turn.  Pass the "
        "numeric id from list_pending's output."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action_id": {
                "type": "integer",
                "description": "Numeric id of the queued action to approve.",
            },
        },
        "required": ["action_id"],
    },
    # Approval itself is a status flip but it unlocks an action whose
    # original risk was high_write — keep the gate strict.
    risk="high_write",
)
async def approve_pending(*, ctx, action_id: int) -> ToolResult:
    cx = unwrap_ctx(ctx)
    row = await get_pending_action(action_id)
    if row is None or row["status"] != "pending":
        return ToolResult(text=t("pending.not_found", cx.user_lang), data={"action_id": action_id})
    ok = await mark_approved(action_id, via="voice")
    if not ok:
        return ToolResult(text=t("pending.not_found", cx.user_lang), data={"action_id": action_id})
    log.info("pending: approved id=%d (%s)", action_id, row["tool_name"])
    return ToolResult(
        text=t("pending.approved", cx.user_lang, summary=row["summary"]),
        data={"action_id": action_id, "summary": row["summary"]},
    )


@tool(
    name="reject_pending",
    description=(
        "Drop a deferred action by id without executing it.  Use when "
        "the user says 'cancel the deferred X', 'forget about it'."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "action_id": {
                "type": "integer",
                "description": "Numeric id of the queued action to drop.",
            },
        },
        "required": ["action_id"],
    },
    risk="high_write",
)
async def reject_pending(*, ctx, action_id: int) -> ToolResult:
    cx = unwrap_ctx(ctx)
    row = await get_pending_action(action_id)
    if row is None or row["status"] != "pending":
        return ToolResult(text=t("pending.not_found", cx.user_lang), data={"action_id": action_id})
    ok = await mark_rejected(action_id, via="voice")
    if not ok:
        return ToolResult(text=t("pending.not_found", cx.user_lang), data={"action_id": action_id})
    log.info("pending: rejected id=%d (%s)", action_id, row["tool_name"])
    return ToolResult(
        text=t("pending.rejected", cx.user_lang, summary=row["summary"]),
        data={"action_id": action_id, "summary": row["summary"]},
    )
