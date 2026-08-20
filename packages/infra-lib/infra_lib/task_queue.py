"""
Task queue with worker pool, priority scheduling, pause/resume, cancels,
and dependency tracking.

Standalone library version — zero internal dependencies.
Optional integrations (EventBus, SSE) are injected at runtime if available.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """A unit of work for the queue."""
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    name: str = ""
    task_type: str = "generic"
    payload: dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    progress: float = 0.0
    progress_msg: str = ""
    result: Any = None
    error: str | None = None
    max_retries: int = 0
    retry_count: int = 0
    timeout: float | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    pause_event: asyncio.Event = field(default_factory=lambda: asyncio.Event())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.pause_event.set()

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at


class WorkerPool:
    """Pool of async workers that pull tasks from a queue."""

    def __init__(self, num_workers: int = 2):
        self.num_workers = num_workers
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handler: Callable[[Task], Awaitable[Any]] | None = None

    @property
    def queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    def set_handler(self, handler: Callable[[Task], Awaitable[Any]]):
        self._handler = handler

    async def start(self):
        """Start the worker pool on the current event loop.

        Idempotent per loop: calling ``start`` twice on the same loop is a
        no-op. If the pool was previously started on a different loop (e.g. an
        earlier ``asyncio.run`` that has since closed), the queue and workers
        are recreated so the pool keeps working.
        """
        loop = asyncio.get_running_loop()
        if self._running and self._loop is loop:
            return
        self._running = True
        self._loop = loop
        self._queue = asyncio.Queue()
        self._workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.num_workers)
        ]
        logger.info("Worker pool started with %d workers", self.num_workers)

    async def stop(self, timeout: float = 5.0):
        self._running = False
        self._loop = None
        for _ in self._workers:
            await self.queue.put(None)
        pending_workers = [w for w in self._workers if not w.done()]
        if pending_workers:
            done, pending = await asyncio.wait(pending_workers, timeout=timeout)
            for p in pending:
                p.cancel()
        self._workers.clear()
        logger.info("Worker pool stopped")

    async def _worker_loop(self, worker_id: int):
        while self._running:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            task: Task = item
            try:
                if self._handler:
                    await self._handler(task)
            except Exception:
                logger.exception("Worker %d handler failed for task %s", worker_id, task.id)
            finally:
                self.queue.task_done()

    @property
    def active_workers(self) -> int:
        return sum(1 for w in self._workers if not w.done())


class TaskQueue:
    """Async task queue with priority scheduling, pause/resume, cancel, dependencies."""

    def __init__(self, num_workers: int = 2):
        self._pool = WorkerPool(num_workers=num_workers)
        self._pool.set_handler(self._process_task)
        self._tasks: dict[str, Task] = {}
        self._pending: list[Task] = []
        self._running: dict[str, Task] = {}
        self._paused: dict[str, Task] = {}
        self._completed: dict[str, Task] = {}
        self._failed: dict[str, Task] = {}
        self._cancelled: dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._sse_callbacks: list[Callable[[str, Task], None]] = []
        self._dispatcher_task: asyncio.Task | None = None
        self._started = False
        self._started_loop: asyncio.AbstractEventLoop | None = None
        self._event_bus = None
        self._stop_event = asyncio.Event()
        self.stats = {
            "enqueued": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }

    def set_event_bus(self, event_bus: Any) -> None:
        """Inject an EventBus instance for task lifecycle events."""
        self._event_bus = event_bus

    def _emit_event(self, event_name: str, task: Task, extra: dict | None = None):
        if self._event_bus:
            data = {
                "task_id": task.id,
                "task_name": task.name,
                "task_type": task.task_type,
                "status": task.status.value,
                "progress": task.progress,
                "error": task.error,
            }
            if extra:
                data.update(extra)
            try:
                asyncio.ensure_future(
                    self._event_bus.emit(event_name, data, source="task_queue")
                )
            except Exception:
                pass

    # ── Lifecycle ──

    async def start(self):
        """Start the dispatcher and worker pool on the current event loop.

        Idempotent per loop.
        """
        loop = asyncio.get_running_loop()
        if self._started and self._started_loop is loop:
            return
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        await self._pool.start()
        self._started = True
        self._started_loop = loop
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info("TaskQueue started with %d workers", self._pool.num_workers)

    async def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        self._started = False
        self._started_loop = None
        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
        for task in list(self._running.values()):
            task.cancel_event.set()
        await self._pool.stop(timeout=timeout)
        logger.info("TaskQueue stopped")

    # ── SSE subscriptions ──

    def subscribe(self, callback: Callable[[str, Task], None]):
        self._sse_callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[str, Task], None]):
        if callback in self._sse_callbacks:
            self._sse_callbacks.remove(callback)

    def _emit(self, event: str, task: Task):
        for cb in self._sse_callbacks:
            try:
                cb(event, task)
            except Exception:
                pass
        bus_event = f"task.{event}"
        self._emit_event(bus_event, task)

    def _push_sse_terminal(self, task: Task, status: TaskStatus):
        """Push a terminal SSE event onto ``task.metadata["sse_queue"]`` if present."""
        sse_queue = task.metadata.get("sse_queue")
        if sse_queue is None:
            return
        stream_name = task.metadata.get("sse_stream", "auto-train")
        try:
            error_event = {"event": "error", "stream": stream_name, "data": {"type": status.value, "message": task.error or f"Task {status.value}"}}
            sse_queue.put_nowait(error_event)
        except Exception:
            pass

    # ── Enqueue ──

    async def enqueue(self, task: Task) -> str:
        async with self._lock:
            task.status = TaskStatus.QUEUED
            self._tasks[task.id] = task
            self._pending.append(task)
            self.stats["enqueued"] += 1
            self._reorder_pending()
        self._emit("enqueued", task)
        return task.id

    async def enqueue_front(self, task: Task) -> str:
        async with self._lock:
            task.status = TaskStatus.QUEUED
            task.priority = Priority.CRITICAL
            self._tasks[task.id] = task
            self._pending.insert(0, task)
            self.stats["enqueued"] += 1
        self._emit("enqueued", task)
        return task.id

    def _reorder_pending(self):
        self._pending.sort(key=lambda t: t.priority, reverse=True)

    # ── Status / query ──

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    def count(self, status: TaskStatus | None = None) -> int:
        if status is None:
            return len(self._tasks)
        return sum(1 for t in self._tasks.values() if t.status == status)

    # ── Control ──

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
                return False
            task.status = TaskStatus.CANCELLED
            task.cancel_event.set()
            task.completed_at = time.time()
            if task.id in self._running:
                del self._running[task.id]
            self._pending[:] = [t for t in self._pending if t.id != task_id]
            if task.id in self._paused:
                del self._paused[task.id]
            self._cancelled[task.id] = task
            self.stats["cancelled"] += 1
        self._emit("cancelled", task)
        self._push_sse_terminal(task, TaskStatus.CANCELLED)
        return True

    async def pause(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.RUNNING:
                return False
            task.pause_event.clear()
            task.status = TaskStatus.PAUSED
            if task.id in self._running:
                del self._running[task.id]
            self._paused[task.id] = task
        self._emit("paused", task)
        return True

    async def resume(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PAUSED:
                return False
            task.pause_event.set()
            task.status = TaskStatus.QUEUED
            if task.id in self._paused:
                del self._paused[task.id]
            self._pending.append(task)
            self._reorder_pending()
        self._emit("resumed", task)
        return True

    async def update_progress(self, task_id: str, progress: float, msg: str = ""):
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.progress = progress
            task.progress_msg = msg
            self._emit("progress", task)

    # ── Dispatch loop ──

    async def _dispatch_loop(self):
        """Continuously feed ready tasks to the worker pool queue."""
        while not self._stop_event.is_set():
            ready: list[Task] = []
            async with self._lock:
                still_pending: list[Task] = []
                for t in self._pending:
                    if t.status == TaskStatus.CANCELLED:
                        continue
                    deps_met = all(
                        dep in self._completed or (
                            dep in self._tasks and self._tasks[dep].status == TaskStatus.COMPLETED
                        )
                        for dep in t.dependencies
                    )
                    if deps_met:
                        ready.append(t)
                    else:
                        still_pending.append(t)
                self._pending = still_pending
                for t in ready:
                    t.status = TaskStatus.RUNNING
                    t.started_at = time.time()
                    self._running[t.id] = t
            for t in ready:
                await self._pool.queue.put(t)
            await asyncio.sleep(0.05)

    # ── Task processor ──

    async def _process_task(self, task: Task):
        """Default handler — subclasses/register override via set_handler."""
        if task.timeout:
            try:
                await asyncio.wait_for(
                    self._run_with_controls(task),
                    timeout=task.timeout,
                )
            except asyncio.TimeoutError:
                async with self._lock:
                    task.status = TaskStatus.FAILED
                    task.error = f"Timeout after {task.timeout}s"
                    task.completed_at = time.time()
                    self._running.pop(task.id, None)
                    self._failed[task.id] = task
                    self.stats["failed"] += 1
                self._emit("failed", task)
                self._push_sse_terminal(task, TaskStatus.FAILED)
        else:
            await self._run_with_controls(task)

    async def _run_with_controls(self, task: Task):
        """Check pause/cancel events and run the task's handler."""
        raise NotImplementedError("Subclasses must implement _run_with_controls or set handler")


class InProcessTaskQueue(TaskQueue):
    """TaskQueue that runs handlers via registered callbacks (no subprocess)."""

    def __init__(self, num_workers: int = 2):
        super().__init__(num_workers=num_workers)
        self._handlers: dict[str, Callable[[Task], Awaitable[Any]]] = {}

    def register_handler(self, task_type: str, handler: Callable[[Task], Awaitable[Any]]):
        self._handlers[task_type] = handler

    def unregister_handler(self, task_type: str):
        self._handlers.pop(task_type, None)

    async def _run_with_controls(self, task: Task):
        handler = self._handlers.get(task.task_type)
        if handler is None:
            async with self._lock:
                task.status = TaskStatus.FAILED
                task.error = f"No handler registered for task_type={task.task_type}"
                task.completed_at = time.time()
                self._running.pop(task.id, None)
                self._failed[task.id] = task
                self.stats["failed"] += 1
            self._emit("failed", task)
            self._push_sse_terminal(task, TaskStatus.FAILED)
            return

        for attempt in range(task.max_retries + 1):
            if task.cancel_event.is_set():
                async with self._lock:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
                    self._running.pop(task.id, None)
                    self._cancelled[task.id] = task
                    self.stats["cancelled"] += 1
                self._emit("cancelled", task)
                self._push_sse_terminal(task, TaskStatus.CANCELLED)
                return

            await task.pause_event.wait()

            if task.cancel_event.is_set():
                async with self._lock:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
                    self._running.pop(task.id, None)
                    self._cancelled[task.id] = task
                    self.stats["cancelled"] += 1
                self._emit("cancelled", task)
                self._push_sse_terminal(task, TaskStatus.CANCELLED)
                return

            try:
                result = await handler(task)
                async with self._lock:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()
                    self._running.pop(task.id, None)
                    self._completed[task.id] = task
                    self.stats["completed"] += 1
                self._emit("completed", task)
                return
            except Exception as e:
                task.retry_count = attempt + 1
                if attempt < task.max_retries:
                    logger.warning(
                        "Task %s attempt %d failed: %s — retrying",
                        task.id, attempt + 1, e,
                    )
                    await asyncio.sleep(0.2 * (attempt + 1))
                else:
                    async with self._lock:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = time.time()
                        self._running.pop(task.id, None)
                        self._failed[task.id] = task
                        self.stats["failed"] += 1
                    self._emit("failed", task)
                    self._push_sse_terminal(task, TaskStatus.FAILED)
                    return
