"""
APScheduler-based reminder engine.

Uses an in-memory AsyncIOScheduler.  Reminders are persisted in SQLite so
they survive server restarts: on startup, reload_pending() re-adds all
not-yet-fired jobs.

Jobs that missed their fire time while the server was offline are delivered
immediately if the delay is ≤ MISFIRE_GRACE_S, otherwise silently dropped.
"""
import asyncio
import logging
import time
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

log = logging.getLogger(__name__)

# Fire a job up to this many seconds late (covers brief server restarts).
MISFIRE_GRACE_S = 300  # 5 minutes

_scheduler: AsyncIOScheduler | None = None


def _get() -> AsyncIOScheduler:
    assert _scheduler is not None, "call start() first"
    return _scheduler


def start() -> None:
    global _scheduler
    _scheduler = AsyncIOScheduler()
    _scheduler.start()
    log.info("scheduler started")


def stop() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
    log.info("scheduler stopped")


def add_periodic(func, *, hours: float = 0, minutes: float = 0, job_id: str) -> None:
    """Register a recurring async job on the running scheduler.

    Runs immediately on first registration (``next_run_time=None`` is the
    APScheduler default for interval triggers — it fires at the first
    interval boundary, not at startup).  Idempotent: replaces any existing
    job with the same ``job_id``.
    """
    _get().add_job(
        func,
        trigger="interval",
        hours=hours,
        minutes=minutes,
        id=job_id,
        replace_existing=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    log.info("periodic job %r registered (every %gh %gm)", job_id, hours, minutes)


def schedule(reminder_id: int, client_id: str, fire_at: float, push_text: str) -> None:
    """Add (or replace) a one-shot job that fires at Unix timestamp ``fire_at``."""
    run_date = datetime.fromtimestamp(fire_at)
    _get().add_job(
        _fire,
        trigger="date",
        run_date=run_date,
        args=[reminder_id, client_id, push_text],
        id=f"reminder_{reminder_id}",
        replace_existing=True,
        misfire_grace_time=MISFIRE_GRACE_S,
    )
    log.info(
        "scheduled reminder %d for %.8s… at %s: %r",
        reminder_id,
        client_id,
        run_date.strftime("%H:%M:%S"),
        push_text[:60],
    )


def cancel(reminder_id: int) -> bool:
    """Remove a pending job.  Returns True if it existed."""
    job_id = f"reminder_{reminder_id}"
    job = _get().get_job(job_id)
    if job is None:
        return False
    job.remove()
    return True


async def reload_pending(client_id: str | None = None) -> int:
    """
    Re-add all not-yet-fired reminders from the DB after a server restart.
    If ``client_id`` is given, only that client's reminders are loaded.
    Returns the number of jobs scheduled.
    """
    from .storage import get_pending_reminders, mark_reminder_fired  # late import

    now = time.time()
    rows = await get_pending_reminders(client_id=client_id)
    scheduled = 0
    for reminder_id, cid, fire_at, push_text in rows:
        late = now - fire_at
        if late > MISFIRE_GRACE_S:
            log.info(
                "reminder %d missed by %.0fs while offline — dropping", reminder_id, late
            )
            asyncio.create_task(mark_reminder_fired(reminder_id, delivered=False))
        elif late > 0:
            # Missed but within grace — fire immediately.
            log.info("reminder %d was late by %.0fs — delivering now", reminder_id, late)
            asyncio.create_task(_fire(reminder_id, cid, push_text))
        else:
            schedule(reminder_id, cid, fire_at, push_text)
            scheduled += 1
    return scheduled


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

async def _fire(reminder_id: int, client_id: str, push_text: str) -> None:
    """Called by APScheduler when a job's trigger time arrives."""
    from . import registry
    from .storage import mark_reminder_fired  # late import — avoids circular dep

    delivered = await registry.push(client_id, push_text, reason="reminder")
    await mark_reminder_fired(reminder_id, delivered=delivered)
    log.info(
        "reminder %d fired (delivered=%s): %r",
        reminder_id,
        delivered,
        push_text[:60],
    )
