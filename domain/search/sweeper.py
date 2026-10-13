"""Periodic search-index sweep: keep indexed stores from drifting behind.

System-wide search has two kinds of store: ``live`` (answers from the source
on demand — never stale) and ``indexed`` (reads a cache/mtime-keyed copy —
stale the moment the source moves until something refreshes it). A query only
refreshes the files it touches, so a doc nobody queries drifts behind the
index forever and a deleted doc leaves a ghost entry. This module closes that
out-of-band: a background asyncio task wakes every
``SLO_SEARCH_SWEEP_INTERVAL_MINUTES`` and calls ``sweep()`` on every registered
store that advertises the :class:`~domain.search.protocol.Sweepable`
capability, so the index catches up to the source without a query forcing it.

The scheduler is deliberately dumb (fixed interval, one pass per tick) and
follows the same shape as ``domain.memory._internal.maintenance``: a
``*_tick`` that does the work and swallows transient failures, a
``run_*_loop`` that sleeps between ticks, and idempotent ``start_*`` /
``stop_*`` pair wired into the API startup/shutdown. It runs inside the async
host, so the (synchronous, file-scanning) sweep goes through
``asyncio.to_thread`` — the sync/async seam the codebase keeps explicit.
"""

from __future__ import annotations

import asyncio
import logging
import os

from domain.search.protocol import Sweepable
from domain.search.registry import get_registry

logger = logging.getLogger("slo.search_sweeper")

#: How often (minutes) the sweeper wakes to refresh indexed stores.
#: ``0`` disables the sweeper. Read fresh each tick so a runtime env change
#: (or a test monkeypatch) is picked up without a restart.
DEFAULT_SWEEP_INTERVAL_MINUTES = 60
INTERVAL_ENV = "SLO_SEARCH_SWEEP_INTERVAL_MINUTES"

_sweep_task: asyncio.Task | None = None


def sweep_interval_minutes() -> float:
    """Current sweep interval in minutes, from env (``0`` disables)."""
    raw = os.environ.get(INTERVAL_ENV, str(DEFAULT_SWEEP_INTERVAL_MINUTES))
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("invalid %s=%r; using %s", INTERVAL_ENV, raw, DEFAULT_SWEEP_INTERVAL_MINUTES)
        return float(DEFAULT_SWEEP_INTERVAL_MINUTES)


def _sweepables() -> list[Sweepable]:
    """Every registered store that advertises the Sweepable capability."""
    reg = get_registry()
    return [a for a in reg.adapters() if isinstance(a, Sweepable)]


def sweep_once() -> dict[str, dict[str, int]]:
    """Sweep every Sweepable store synchronously; return per-store counts.

    A store that raises is logged and recorded as ``{"error": 1}`` — one bad
    store never aborts the sweep of the others, and the failure is visible in
    the returned counts rather than swallowed silently.

    Returns:
        ``{store_name: outcome_counts}`` for every swept store.
    """
    results: dict[str, dict[str, int]] = {}
    for adapter in _sweepables():
        try:
            results[adapter.store] = adapter.sweep()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("search sweep failed for %s: %s", adapter.store, e)
            results[adapter.store] = {"error": 1}
    return results


async def sweep_tick() -> dict[str, dict[str, int]]:
    """Run one sweep pass off the event loop (blocking file I/O in a thread).

    Returns:
        The per-store outcome counts from :func:`sweep_once`.
    """
    return await asyncio.to_thread(sweep_once)


async def run_search_sweeper() -> None:
    """Scheduler loop: sleep ``interval`` then tick, forever.

    Reads the interval each cycle so the loop honors a changed env value. The
    loop never raises — every tick failure is contained in ``sweep_tick``.
    """
    while True:
        minutes = sweep_interval_minutes()
        if minutes <= 0:
            logger.info("search sweeper disabled (interval 0)", extra={"tag": "INFRA"})
            return
        await asyncio.sleep(minutes * 60)
        counts = await sweep_tick()
        swept = {s: c for s, c in counts.items() if c.get("error")}
        logger.debug("search sweep: %s", counts, extra={"tag": "INFRA"})
        if swept:
            logger.warning("search sweep errors: %s", swept, extra={"tag": "INFRA"})


def start_search_sweeper() -> asyncio.Task | None:
    """Start the periodic sweeper as a background task; idempotent.

    Returns:
        The running task, or ``None`` when the interval is ``0`` (disabled).
    """
    global _sweep_task
    if _sweep_task is not None and not _sweep_task.done():
        return _sweep_task
    if sweep_interval_minutes() <= 0:
        return None
    _sweep_task = asyncio.create_task(run_search_sweeper())
    return _sweep_task


async def stop_search_sweeper() -> None:
    """Cancel the sweeper task if running; safe to call when not running."""
    global _sweep_task
    if _sweep_task is not None and not _sweep_task.done():
        _sweep_task.cancel()
        try:
            await _sweep_task
        except asyncio.CancelledError:
            pass
    _sweep_task = None
