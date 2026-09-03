"""
Producer-Consumer Queue — thread-safe work distribution for sync and async code.

Provides ``ProducerConsumerQueue``, a bounded, priority-capable queue with:
  - Sync API (``put``/``get``) for threaded producers/consumers
  - Async API (``async_put``/``async_get``) for coroutine contexts
  - Backpressure via configurable ``maxsize``
  - Priority scheduling (lower number = higher priority)
  - Graceful shutdown with drain or drop
  - Metrics (enqueued, consumed, dropped, active consumers)

Usage::

    from domains.infrastructure.producer_consumer import ProducerConsumerQueue

    q = ProducerConsumerQueue(maxsize=100, num_consumers=4)

    # Sync producer (thread)
    q.put(("load_model", {"path": "/models/llm.slo"}))

    # Async producer (event loop)
    await q.async_put(("load_model", {"path": "/models/llm.slo"}))

    # Sync consumer (thread)
    item = q.get(timeout=5.0)

    # Async consumer (event loop)
    item = await q.async_get(timeout=5.0)

Design:
  - Internally uses :class:`queue.PriorityQueue` (thread-safe) for sync API
  - Exposes ``asyncio.Queue`` wrappers for async API
  - Consumer threads are managed by the queue itself (daemon threads)
  - Priority items are ``(priority, sequence, item)`` tuples
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger("slo.infrastructure.producer_consumer")

T = TypeVar("T")

_POISON_PILL = object()  # sentinel for shutdown


class ShutdownMode(str, Enum):
    """How to handle in-flight items on shutdown."""
    DRAIN = "drain"   # finish all queued work
    DROP = "drop"     # discard remaining items


@dataclass(order=True)
class _PriorityItem(Generic[T]):
    """Wraps an item with priority and sequence number for stable ordering."""
    priority: int
    sequence: int
    item: T = field(compare=False)


class ProducerConsumerQueue(Generic[T]):
    """Thread-safe producer-consumer queue with backpressure and priority.

    Args:
        maxsize: Maximum queue capacity. 0 = unbounded.
        num_consumers: Number of consumer threads to start.
        handler: Callable invoked for each item. Must be thread-safe.
        priority: Enable priority scheduling (lower number = higher priority).
        shutdown_mode: How to handle pending items on shutdown.
        name: Human-readable name for logging.
    """

    def __init__(
        self,
        maxsize: int = 0,
        num_consumers: int = 1,
        handler: Callable[[T], Any] | None = None,
        priority: bool = False,
        shutdown_mode: ShutdownMode = ShutdownMode.DRAIN,
        name: str = "pcq",
    ):
        self.name = name
        self.num_consumers = num_consumers
        self._handler = handler
        self._shutdown_mode = shutdown_mode
        self._use_priority = priority

        # Core queue — PriorityQ for priority mode, simple Queue otherwise
        if priority:
            self._queue: queue.Queue[_PriorityItem[T]] = queue.PriorityQueue(maxsize=maxsize)
        else:
            self._queue = queue.Queue(maxsize=maxsize)

        self._seq = 0
        self._seq_lock = threading.Lock()

        # Shutdown coordination
        self._stop_event = threading.Event()
        self._drain_event = threading.Event()
        self._drain_event.set()  # starts drained

        # Consumer threads
        self._consumers: list[threading.Thread] = []
        self._consumer_active = [False] * num_consumers

        # Metrics
        self._enqueued = 0
        self._consumed = 0
        self._dropped = 0
        self._errors = 0
        self._metrics_lock = threading.Lock()

        # Async bridge (lazy)
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_bridge_task: asyncio.Task | None = None

    # ── Sync API ──────────────────────────────────────────────────────

    def put(self, item: T, timeout: float | None = None, priority: int = 0) -> bool:
        """Put an item into the queue (thread-safe).

        Returns True if enqueued, False if queue is full or shut down.
        """
        if self._stop_event.is_set():
            return False
        try:
            if self._use_priority:
                with self._seq_lock:
                    self._seq += 1
                    seq = self._seq
                self._queue.put(_PriorityItem(priority=priority, sequence=seq, item=item),
                                timeout=timeout, block=True)
            else:
                self._queue.put(item, timeout=timeout, block=True)
            with self._metrics_lock:
                self._enqueued += 1
            return True
        except queue.Full:
            with self._metrics_lock:
                self._dropped += 1
            return False
        except (OSError, ValueError):
            return False

    def put_nowait(self, item: T, priority: int = 0) -> bool:
        """Non-blocking put. Returns False if full."""
        return self.put(item, timeout=0, priority=priority)

    def get(self, timeout: float | None = None) -> tuple[bool, T | None]:
        """Get an item from the queue (thread-safe).

        Returns ``(True, item)`` on success, ``(False, None)`` on timeout/shutdown.
        """
        if self._stop_event.is_set() and self._queue.empty():
            return False, None
        try:
            if self._use_priority:
                entry: _PriorityItem[T] = self._queue.get(timeout=timeout)
                item = entry.item
            else:
                item = self._queue.get(timeout=timeout)
            with self._metrics_lock:
                self._consumed += 1
            return True, item
        except queue.Empty:
            return False, None
        except (OSError, ValueError):
            return False, None

    def task_done(self) -> None:
        """Mark a task as done (for JoinableQueue semantics)."""
        try:
            self._queue.task_done()
        except (ValueError, OSError):
            pass

    # ── Async API ─────────────────────────────────────────────────────

    async def async_put(self, item: T, timeout: float | None = None, priority: int = 0) -> bool:
        """Async put — bridges to sync queue via to_thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.put(item, timeout=timeout, priority=priority))

    async def async_get(self, timeout: float | None = None) -> tuple[bool, T | None]:
        """Async get — bridges to sync queue via to_thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.get(timeout=timeout))

    # ── Consumer Management ───────────────────────────────────────────

    def start(self) -> None:
        """Start consumer threads."""
        if self._consumers:
            return
        self._stop_event.clear()
        self._drain_event.clear()
        for i in range(self.num_consumers):
            t = threading.Thread(
                target=self._consumer_loop,
                args=(i,),
                name=f"{self.name}-consumer-{i}",
                daemon=True,
            )
            self._consumers.append(t)
            t.start()
        logger.info("ProducerConsumerQueue[%s] started (%d consumers, maxsize=%d, priority=%s)",
                     self.name, self.num_consumers,
                     self._queue.maxsize if hasattr(self._queue, 'maxsize') else 0,
                     self._use_priority,
                     extra={"tag": "INFRA"})

    def stop(self, timeout: float = 5.0) -> None:
        """Stop consumers. DRAIN or DROP remaining items per shutdown_mode."""
        if not self._consumers:
            return

        if self._shutdown_mode == ShutdownMode.DRAIN:
            # Wait for queue to drain (with timeout)
            deadline = time.monotonic() + timeout
            while not self._queue.empty() and time.monotonic() < deadline:
                time.sleep(0.05)

        self._stop_event.set()

        # Send poison pills so consumers break out of get()
        for _ in self._consumers:
            try:
                if self._use_priority:
                    with self._seq_lock:
                        self._seq += 1
                    self._queue.put_nowait(_PriorityItem(priority=999999, sequence=self._seq, item=_POISON_PILL))
                else:
                    self._queue.put_nowait(_POISON_PILL)
            except (queue.Full, OSError):
                pass

        deadline = time.monotonic() + timeout
        for t in self._consumers:
            remaining = max(0.01, deadline - time.monotonic())
            t.join(timeout=remaining)
            if t.is_alive():
                logger.warning("Consumer thread %s did not stop in time", t.name)

        self._consumers.clear()
        self._drain_event.set()

        with self._metrics_lock:
            queued = self._queue.qsize()
            if self._shutdown_mode == ShutdownMode.DROP and queued > 0:
                self._dropped += queued
                logger.info("ProducerConsumerQueue[%s] dropped %d items on shutdown", self.name, queued)
            else:
                logger.info("ProducerConsumerQueue[%s] stopped, %d items remaining", self.name, queued)

    def _consumer_loop(self, consumer_id: int) -> None:
        """Main loop for a consumer thread."""
        self._consumer_active[consumer_id] = True
        try:
            while not self._stop_event.is_set():
                ok, item = self.get(timeout=0.5)
                if not ok:
                    continue
                if item is _POISON_PILL:
                    break
                try:
                    if self._handler:
                        self._handler(item)
                except Exception as e:
                    with self._metrics_lock:
                        self._errors += 1
                    logger.warning("Consumer %d handler error: %s", consumer_id, e,
                                   extra={"tag": "INFRA"})
                finally:
                    self.task_done()
        finally:
            self._consumer_active[consumer_id] = False

    # ── Metrics ───────────────────────────────────────────────────────

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        if hasattr(self._queue, 'full'):
            return self._queue.full()
        return False

    @property
    def active_consumers(self) -> int:
        return sum(self._consumer_active)

    @property
    def metrics(self) -> dict[str, int]:
        with self._metrics_lock:
            return {
                "enqueued": self._enqueued,
                "consumed": self._consumed,
                "dropped": self._dropped,
                "errors": self._errors,
                "queued": self._queue.qsize(),
                "active_consumers": self.active_consumers,
            }

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and bool(self._consumers)

    def __repr__(self) -> str:
        return (f"ProducerConsumerQueue(name={self.name!r}, consumers={self.num_consumers}, "
                f"queued={self.qsize}, priority={self._use_priority})")


# ── Convenience factory ──────────────────────────────────────────────

_default_queue: ProducerConsumerQueue | None = None


def get_producer_consumer_queue() -> ProducerConsumerQueue:
    """Get or create the global producer-consumer queue."""
    global _default_queue
    if _default_queue is None:
        _default_queue = ProducerConsumerQueue(
            maxsize=256,
            num_consumers=4,
            name="global",
        )
    return _default_queue


def set_producer_consumer_queue(q: ProducerConsumerQueue) -> None:
    """Override the global queue (for testing)."""
    global _default_queue
    _default_queue = q
