"""Async priority request queue with bounded concurrency and metrics."""

from __future__ import annotations

import asyncio
import heapq
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Lock
from typing import Any

from .structured_log import StructuredLogger

logger = StructuredLogger("slo.infrastructure.request_queue")


class Priority(IntEnum):
    """Request priority — lower number = higher priority (dequeued first)."""

    HIGH = 0  # interactive chat
    MEDIUM = 1  # generate / inference
    LOW = 2  # batch / background


@dataclass
class QueueMetrics:
    """Snapshot of priority queue state (thread-safe)."""

    depth_high: int = 0
    depth_medium: int = 0
    depth_low: int = 0
    total_depth: int = 0
    served: int = 0
    timed_out: int = 0
    avg_wait_ms: float = 0.0
    max_wait_ms: float = 0.0


@dataclass(order=True)
class _QueueItem:
    """Item in the priority queue.  ``order=True`` makes ``heapq`` sort by
    ``(priority, enqueue_order)`` — lower priority first, FIFO within same
    priority."""

    priority: int  # Priority value (0=HIGH, 1=MEDIUM, 2=LOW)
    enqueue_order: int  # insertion order for FIFO within priority level
    coro: Any = field(compare=False)  # awaitable to execute
    future: asyncio.Future = field(compare=False)  # resolves when coro is done
    enqueued_at: float = field(compare=False, default_factory=time.time)
    request_id: str = field(compare=False, default="")


class PriorityRequestQueue:
    """Async request queue with 3 priority levels and bounded concurrency.

    Workers pull the highest-priority item (FIFO within priority level)
    and execute it, up to ``max_concurrent`` in-flight.
    """

    def __init__(self, max_concurrent: int = 2, max_queue: int = 128):
        self._max_concurrent = max_concurrent
        self._max_queue = max_queue
        self._heap: list[_QueueItem] = []
        self._order_counter = 0
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._wake_event = asyncio.Event()

        # Metrics
        self._served = 0
        self._total_wait = 0.0
        self._max_wait_s = 0.0
        self._metrics_lock = Lock()

        # Atomic depth counters (updated under _lock, read under _metrics_lock)
        self._depth_high = 0
        self._depth_medium = 0
        self._depth_low = 0

    # --- Public API ---

    async def acquire(
        self,
        priority: Priority = Priority.MEDIUM,
        request_id: str = "",
    ) -> Callable[[], None]:
        """Reserve a slot for long-lived work (e.g. streaming).

        Returns a ``release()`` callable that the caller *must* invoke when
        the work completes (or is aborted).  Until release, the slot counts
        against ``max_concurrent``, blocking other submissions.

        Unlike :meth:`submit`, ``acquire`` does **not** execute a coroutine
        inside the queue worker — it only grants permission to proceed.
        The caller runs their work externally and calls ``release()``.
        """
        loop = asyncio.get_running_loop()
        grant: asyncio.Future = loop.create_future()

        async with self._lock:
            if len(self._heap) >= self._max_queue:
                logger.warning(
                    "Queue full on acquire", max_queue=self._max_queue, request_id=request_id
                )
                raise RuntimeError(f"Queue full ({self._max_queue} items)")
            item = _QueueItem(
                priority=priority.value,
                enqueue_order=self._order_counter,
                coro=None,  # marker — no coroutine to execute
                future=grant,
                request_id=request_id or f"acq-{self._order_counter}",
            )
            self._order_counter += 1
            heapq.heappush(self._heap, item)
            # Update atomic depth counters
            if item.priority == 0:
                self._depth_high += 1
            elif item.priority == 1:
                self._depth_medium += 1
            elif item.priority == 2:
                self._depth_low += 1

        logger.debug(
            "Acquire enqueued",
            request_id=item.request_id,
            priority=priority.name,
            queue_depth=len(self._heap),
        )
        self._wake_event.set()

        released = False

        def _release() -> None:
            nonlocal released
            if not released:
                released = True
                self._in_flight -= 1
                self._wake_event.set()

        try:
            await grant  # blocks until worker pops this marker
        except (asyncio.CancelledError, Exception, GeneratorExit):
            if grant.done() and not grant.cancelled():
                # Worker already popped it — release the slot
                _release()
            else:
                # Marker still in heap — remove it
                async with self._lock:
                    self._heap = [x for x in self._heap if x.future is not grant]
                    heapq.heapify(self._heap)
            raise

        return _release

    async def submit(
        self,
        coro: Any,
        priority: Priority = Priority.MEDIUM,
        request_id: str = "",
    ) -> Any:
        """Submit an awaitable for execution, returning its result."""
        async with self._lock:
            if len(self._heap) >= self._max_queue:
                logger.warning("Queue full", max_queue=self._max_queue, request_id=request_id)
                if inspect.iscoroutine(coro):
                    coro.close()
                raise RuntimeError(f"Queue full ({self._max_queue} items)")
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            item = _QueueItem(
                priority=priority.value,
                enqueue_order=self._order_counter,
                coro=coro,
                future=future,
                request_id=request_id or f"req-{self._order_counter}",
            )
            self._order_counter += 1
            heapq.heappush(self._heap, item)
            # Update atomic depth counters
            if item.priority == 0:
                self._depth_high += 1
            elif item.priority == 1:
                self._depth_medium += 1
            elif item.priority == 2:
                self._depth_low += 1

        logger.debug(
            "Enqueued",
            request_id=item.request_id,
            priority=priority.name,
            queue_depth=len(self._heap),
        )
        self._wake_event.set()
        return await future

    def close(self) -> None:
        """Discard all pending (not yet started) submissions.

        Closes heap-resident coroutines and cancels their result futures so
        callers still awaiting :meth:`submit` unblock with ``CancelledError``.
        Safe once workers have stopped; a concurrent worker may still start a
        just-popped item, which is then awaited normally.

        Side effects:
            - closes every enqueued-but-unstarted coroutine
            - cancels every enqueued result future
            - clears the heap
        """
        items = list(self._heap)
        self._heap.clear()
        self._depth_high = 0
        self._depth_medium = 0
        self._depth_low = 0
        for item in items:
            if inspect.iscoroutine(item.coro):
                item.coro.close()
            if not item.future.done():
                item.future.cancel()
        self._wake_event.set()

    def _pop(self) -> _QueueItem | None:
        """Pop highest-priority item (caller must hold ``_lock``)."""
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        # Update atomic depth counters
        if item.priority == 0:
            self._depth_high -= 1
        elif item.priority == 1:
            self._depth_medium -= 1
        elif item.priority == 2:
            self._depth_low -= 1
        return item

    async def depth(self) -> list[int]:
        """Return queue depth per priority level: [high, medium, low]."""
        async with self._lock:
            return [self._depth_high, self._depth_medium, self._depth_low]

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def metrics_snapshot(self) -> QueueMetrics:
        """Return queue metrics snapshot (thread-safe, no heap iteration)."""
        with self._metrics_lock:
            avg = (self._total_wait / max(self._served, 1)) * 1000
            return QueueMetrics(
                depth_high=self._depth_high,
                depth_medium=self._depth_medium,
                depth_low=self._depth_low,
                total_depth=self._depth_high + self._depth_medium + self._depth_low,
                served=self._served,
                avg_wait_ms=avg,
                max_wait_ms=self._max_wait_s * 1000,
            )

    # --- Worker loop ---

    async def worker(self) -> None:
        """Pop and execute items (runs forever)."""
        while True:
            await self._wake_event.wait()

            item: _QueueItem | None = None
            async with self._lock:
                if self._in_flight < self._max_concurrent and self._heap:
                    item = self._pop()

            if item is None:
                async with self._lock:
                    if not self._heap:
                        self._wake_event.clear()
                await asyncio.sleep(0.01)
                continue

            wait_s = time.time() - item.enqueued_at
            with self._metrics_lock:
                self._served += 1
                self._total_wait += wait_s
                if wait_s > self._max_wait_s:
                    self._max_wait_s = wait_s

            logger.debug(
                "Dequeued",
                request_id=item.request_id,
                wait_ms=round(wait_s * 1000, 1),
                priority=Priority(item.priority).name,
            )

            # Reservation marker — grant the slot, let caller manage in_flight
            if item.coro is None:
                self._in_flight += 1
                item.future.set_result(None)
                continue

            # Normal work item
            self._in_flight += 1
            try:
                result = await item.coro
                item.future.set_result(result)
            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)
            finally:
                self._in_flight -= 1
                self._wake_event.set()
