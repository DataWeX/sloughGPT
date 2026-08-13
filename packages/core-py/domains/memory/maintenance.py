"""Periodic memory maintenance: consolidate facts and bound the archive.

The memory lifecycle is write -> read -> maintain. The write (chat loop,
task producers) and read (RAG context frame) paths run per-request, but
consolidation only ever happens when someone enqueues a
``memory.consolidate`` task. This module closes that loop for the server:
a background asyncio task wakes up every
``MemoryConfig.maintenance_interval_minutes`` and enqueues one consolidate
pass (so near-duplicate facts are pruned) and prunes the task-backed
provenance archive to ``MemoryConfig.archive_retention_days`` (so the
audit trail stays bounded without operator action).

The scheduler is intentionally dumb (fixed interval, one pass per tick).
It enqueues through the same ``submit_memory_consolidate`` helper the rest
of the system uses, so it inherits the task queue's priority/retry/EventBus
guarantees and writes the same provenance archive record.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from domains.memory.memory_config import MemoryConfig
from domains.memory.task_memory import prune_archive, submit_memory_consolidate

logger = logging.getLogger("slo.memory_maintenance")

_maintenance_task: Optional[asyncio.Task] = None


async def maintenance_tick() -> Optional[str]:
    """
    Run a single maintenance pass: prune the archive, enqueue consolidation.

    Prunes the task-backed provenance archive to
    ``MemoryConfig.archive_retention_days`` (records older than the window
    are deleted), then enqueues one ``memory.consolidate`` task. No-op
    (returns ``None``) when memory is disabled or the interval is set to
    zero. Failures pruning or enqueueing are logged and swallowed so a
    transient error never kills the scheduler loop.

    Args:
        none.

    Returns:
        The enqueued task's id, or ``None`` when maintenance is disabled or
        enqueueing failed.

    Side effects:
        - rewrites ``facts.jsonl`` keeping only in-window archive records.
        - enqueues a ``memory.consolidate`` task on the global task queue.
        - logs a warning when pruning or enqueueing fails.
    """
    cfg = MemoryConfig.get()
    if not cfg.enabled or cfg.maintenance_interval_minutes <= 0:
        return None
    try:
        pruned = prune_archive()
        if pruned:
            logger.info("Memory maintenance pruned %d archive record(s)",
                        pruned, extra={"tag": "INFRA"})
    except Exception as e:
        logger.warning("Memory maintenance archive prune failed: %s", e,
                       extra={"tag": "INFRA"})
    try:
        from domains.infrastructure.task_queue import get_task_queue
        task_id = await submit_memory_consolidate(queue=get_task_queue())
        logger.info("Memory maintenance enqueued consolidate task %s", task_id,
                    extra={"tag": "INFRA"})
        return task_id
    except Exception as e:
        logger.warning("Memory maintenance enqueue failed: %s", e, extra={"tag": "INFRA"})
        return None


async def run_memory_maintenance() -> None:
    """
    Scheduler loop: sleep ``maintenance_interval_minutes`` then tick.

    Returns immediately when maintenance is disabled (memory off or interval
    zero). The loop never raises - every tick failure is contained inside
    ``maintenance_tick`` so the background task stays alive.

    Args:
        none.

    Returns:
        none.

    Side effects:
        - every interval: prunes the provenance archive to the retention
          window and enqueues a ``memory.consolidate`` task while running.
    """
    cfg = MemoryConfig.get()
    if not cfg.enabled or cfg.maintenance_interval_minutes <= 0:
        logger.info("Memory maintenance disabled", extra={"tag": "INFRA"})
        return
    interval_s = cfg.maintenance_interval_minutes * 60
    while True:
        await asyncio.sleep(interval_s)
        await maintenance_tick()


def start_memory_maintenance() -> Optional[asyncio.Task]:
    """
    Start the periodic maintenance scheduler as a background asyncio task.

    Idempotent: calling twice returns the already-running task.

    Args:
        none.

    Returns:
        The scheduler task, or ``None`` when maintenance is disabled.

    Side effects:
        - spawns a background task that prunes the archive and enqueues
          ``memory.consolidate`` every interval.
    """
    global _maintenance_task
    if _maintenance_task is not None and not _maintenance_task.done():
        return _maintenance_task
    cfg = MemoryConfig.get()
    if not cfg.enabled or cfg.maintenance_interval_minutes <= 0:
        return None
    _maintenance_task = asyncio.create_task(run_memory_maintenance())
    return _maintenance_task


async def stop_memory_maintenance() -> None:
    """
    Cancel the background maintenance scheduler task, if running.

    Args:
        none.

    Returns:
        none.

    Side effects:
        - cancels the scheduler task and waits for it to unwind.
    """
    global _maintenance_task
    task = _maintenance_task
    _maintenance_task = None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
