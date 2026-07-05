"""
Background executor for the deferred-action queue.

The flow:

  1. LLM asks for a ``risk='high_write'`` tool without an active auth
     window → ``agent._execute_one`` enqueues a row in
     ``pending_actions(status='pending')``.
  2. User approves via voice tool or via the UI button → status flips
     to ``'approved'``.
  3. This module's periodic task picks up ``'approved'`` rows and
     dispatches them through the normal tool registry, with a
     **synthetic AgentContext** marked ``is_authenticated=True`` so
     the gating in ``_execute_one`` waves them through.
  4. Row advances to ``'executed'`` (or ``'execution_failed'``) and the
     tool's spoken reply is pushed via ``registry.push()`` so the user
     hears the outcome even if they're nowhere near the mic.

The task runs every ``POLL_INTERVAL_S`` seconds — short enough that
"approve in UI → see action complete" feels instant, but not so short
that it busies the event loop.  APScheduler handles the timing.
"""
from __future__ import annotations

import asyncio
import logging

from . import registry
from .agent import AgentContext
from .storage import list_approved_actions, mark_executed
from .tools import dispatch

log = logging.getLogger(__name__)

# How often we sweep the approved-action queue.  3 s feels snappy in
# the UI without burning CPU; the action queue is typically empty.
POLL_INTERVAL_S = 3.0

_task: asyncio.Task | None = None
# Created fresh in start() so it binds to the CURRENT event loop.  A
# module-level asyncio.Event() binds to the first loop that touches it
# (asyncio mixin caches the loop), which breaks across loops — e.g. each
# TestClient(app) spins a new loop, and reusing a stale Event raises
# "bound to a different event loop".  None until start() runs.
_stopping: asyncio.Event | None = None


async def _process_one(row: dict) -> None:
    """Dispatch a single approved action through the tool registry.

    On success the tool's spoken reply is pushed to the row's client via
    registry.push() — so the user hears "Created event …" even if they
    approved from the UI on a different device.
    """
    action_id = row["id"]
    tool_name = row["tool_name"]
    args = row["tool_args"] or {}
    profile_id = row["profile_id"]
    client_id = row["client_id"]

    # Synthetic context: approval IS authorisation, so we mark this
    # call authenticated regardless of any session's auth window.  No
    # stream_sink / progress_sink — there's nobody listening at the
    # other end of the WS chunk pipeline for a server-initiated push.
    ctx = AgentContext(
        client_id=client_id,
        profile_id=profile_id,
        is_authenticated=True,
        stream_sink=None,
        progress_sink=None,
    )

    log.info(
        "pending_executor: dispatching action=%d tool=%r profile=%s",
        action_id, tool_name, profile_id,
    )
    try:
        result = await dispatch(tool_name, args, ctx=ctx)
    except Exception as exc:
        log.exception("pending_executor: tool %r crashed", tool_name)
        await mark_executed(action_id, ok=False, note=f"failed: {exc.__class__.__name__}")
        return

    note = "OK" if not (result.data or {}).get("error") else f"failed: {result.data.get('error')}"
    await mark_executed(
        action_id,
        ok=note == "OK",
        note=note,
    )

    # Tell the user what happened.  registry.push() returns False if no
    # WS session is currently registered for this client_id; we ignore
    # that — the UI's Pending tab will still show the row moved to
    # executed/failed status the next time it's loaded.
    if client_id and result.text:
        try:
            await registry.push(client_id, result.text, reason="pending_executed")
        except Exception:
            log.exception("pending_executor: push failed for client %s", client_id)


async def _loop() -> None:
    """Periodic sweep.  Cooperative: respects ``_stopping`` between sweeps
    so ``stop()`` returns within one POLL_INTERVAL_S."""
    log.info("pending_executor: started (poll=%.1fs)", POLL_INTERVAL_S)
    try:
        while not _stopping.is_set():
            try:
                approved = await list_approved_actions()
            except Exception:
                log.exception("pending_executor: list_approved_actions failed")
                approved = []
            for row in approved:
                if _stopping.is_set():
                    break
                await _process_one(row)
            try:
                await asyncio.wait_for(_stopping.wait(), timeout=POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
    finally:
        log.info("pending_executor: stopped")


def start() -> None:
    """Launch the background executor.  Idempotent."""
    global _task, _stopping
    if _task and not _task.done():
        return
    _stopping = asyncio.Event()  # bind to the current running loop
    _task = asyncio.create_task(_loop())


def stop() -> None:
    """Signal the executor to exit at its next sweep boundary.  The task
    isn't awaited here — callers from ``lifespan`` shutdown can await it
    if they need to confirm a clean exit before tearing down the loop."""
    if _stopping is not None:
        _stopping.set()
