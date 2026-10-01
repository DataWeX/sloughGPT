"""Shared bounded fire-and-forget worker pool.

One dispatch surface for untracked background work ("fire-and-forget"):
notifications, watchdogs, collectors. See
``docs/plans/2026-09-20-execution-consolidation.md``.

Policy:
  - Fixed pool of daemon workers (``POOL_SIZE``); the pool never grows.
  - Submission never blocks the caller: tasks go to a bounded queue
    (``QUEUE_SIZE``). When the queue is full the task is DROPPED and
    counted — losing a notification is preferable to stalling a worker
    past job completion (the bug this replaces).
  - ``threading.Thread(...)`` at a call site is a review flag; route
    through :func:`submit` instead.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

logger = logging.getLogger("slo.fire_and_forget")

POOL_SIZE = 4
QUEUE_SIZE = 256
_STOP = object()


class FireAndForgetPool:
    """Bounded, non-blocking pool for fire-and-forget tasks."""

    __slots__ = (
        "_q",
        "_workers",
        "_submitted",
        "_dropped",
        "_locked_logged",
        "_lock",
        "_closed",
    )

    def __init__(self, size: int = POOL_SIZE, queue_size: int = QUEUE_SIZE):
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._workers: list[threading.Thread] = []
        self._submitted = 0
        self._dropped = 0
        self._locked_logged = False
        self._lock = threading.Lock()
        self._closed = False
        for _ in range(size):
            thread = threading.Thread(target=self._run, daemon=True, name="faf-worker")
            thread.start()
            self._workers.append(thread)

    def _run(self) -> None:
        q = self._q
        while True:
            item = q.get()
            try:
                if item is _STOP:
                    return
                fn = item
                try:
                    fn()
                except Exception as exc:
                    logger.debug("fire-and-forget task failed: %s", exc)
            finally:
                q.task_done()

    def submit(self, fn: Callable[[], None]) -> bool:
        """Queue a task. Never blocks; returns False when dropped or closed."""
        with self._lock:
            if self._closed:
                return False
            self._submitted += 1
        try:
            self._q.put_nowait(fn)
        except queue.Full:
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
                log = not self._locked_logged
                self._locked_logged = True
            if log:
                logger.warning(
                    "fire-and-forget queue full; dropping tasks (dropped=%d total)",
                    dropped,
                )
            return False
        if self._q.empty():
            with self._lock:
                self._locked_logged = False
        return True

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop workers. Best-effort; outstanding tasks keep running (daemon)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for _ in self._workers:
            try:
                self._q.put_nowait(_STOP)
            except queue.Full:
                break
        for thread in self._workers:
            thread.join(timeout=timeout)
        self._workers.clear()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "dropped": self._dropped,
                "queued": self._q.qsize(),
                "workers": len(self._workers),
            }


_pool: FireAndForgetPool | None = None
_pool_lock = threading.Lock()


def get_pool(size: int = POOL_SIZE, queue_size: int = QUEUE_SIZE) -> FireAndForgetPool:
    """Return the shared process-global fire-and-forget pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = FireAndForgetPool(size=size, queue_size=queue_size)
    return _pool


def reset_pool() -> None:
    """Shutdown the global pool and forget it (test teardown)."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown()
