"""
Task queue — manages tasks with priority, routing, and persistence.

Provides:
  - Priority-based task ordering
  - Tree/instance routing
  - Persistence to disk
  - Pause/resume/cancel support
  - Event callbacks
  - Worker pool via ProducerConsumerQueue (optional threaded execution)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("slo.pugqeep")


class TaskStatus(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Task:
    """A unit of work in the queue."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    data: Any = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    tree_id: Optional[str] = None  # assigned tree/instance
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "data": self.data,
            "status": self.status.value,
            "priority": self.priority.value,
            "tree_id": self.tree_id,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            data=d.get("data"),
            status=TaskStatus(d["status"]),
            priority=TaskPriority(d["priority"]),
            tree_id=d.get("tree_id"),
            result=d.get("result"),
            error=d.get("error"),
            created_at=d.get("created_at", 0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            retries=d.get("retries", 0),
            max_retries=d.get("max_retries", 3),
            metadata=d.get("metadata", {}),
        )


class TaskQueue:
    """Priority task queue with persistence, routing, and optional worker pool."""

    def __init__(self,
                 name: str = "default",
                 storage_dir: Optional[Path] = None,
                 max_size: int = 10000):
        """Initialize task queue.

        Args:
            name: Queue name.
            storage_dir: Directory for persistence. None = in-memory only.
            max_size: Maximum tasks in queue.
        """
        self.name = name
        self._storage_dir = storage_dir
        self._max_size = max_size
        self._tasks: Dict[str, Task] = {}
        self._pending: List[str] = []  # task ids, sorted by priority
        self._running: Dict[str, Task] = {}
        self._completed: List[str] = []
        self._handlers: Dict[str, Callable] = {}
        self._paused = False
        self._callbacks: List[Callable] = []

        # Worker pool (optional — via ProducerConsumerQueue)
        self._worker_queue: Optional[Any] = None  # ProducerConsumerQueue[Task]
        self._num_workers: int = 0

    def submit(self, task: Task) -> Task:
        """Submit a task to the queue.

        If workers are running, the task is dispatched automatically.
        Otherwise, it waits in the pending list for manual ``next()`` calls.
        """
        if len(self._tasks) >= self._max_size:
            raise ValueError(f"Queue full (max {self._max_size} tasks)")

        self._tasks[task.id] = task

        if self._worker_queue is not None and not self._paused:
            # Dispatch to worker pool (priority mapped from TaskPriority)
            priority = self._task_priority_to_int(task.priority)
            dispatched = self._worker_queue.put(task, priority=priority)
            if not dispatched:
                logger.warning("TaskQueue[%s]: worker queue full, task %s queued locally",
                               self.name, task.id)
                self._pending.append(task.id)
            else:
                self._pending.append(task.id)
        else:
            self._pending.append(task.id)
            self._sort_pending()

        if self._storage_dir:
            self._persist()

        logger.debug("TaskQueue[%s]: submitted %s (%s)", self.name, task.id, task.name)
        return task

    @staticmethod
    def _task_priority_to_int(p: TaskPriority) -> int:
        """Map TaskPriority to integer (lower = higher priority) for ProducerConsumerQueue."""
        return {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
        }.get(p, 2)

    def submit_many(self, tasks: List[Task]) -> List[Task]:
        """Submit multiple tasks."""
        for task in tasks:
            self.submit(task)
        return tasks

    def next(self) -> Optional[Task]:
        """Get the next task to process (highest priority, oldest first)."""
        if self._paused:
            return None

        while self._pending:
            task_id = self._pending.pop(0)
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                self._running[task.id] = task
                return task

        return None

    def complete(self, task_id: str, result: Any = None) -> Optional[Task]:
        """Mark a task as completed."""
        task = self._running.pop(task_id, None)
        if task is None:
            return None

        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = time.time()
        self._completed.append(task_id)

        if self._storage_dir:
            self._persist()

        self._notify_callbacks(task)
        return task

    def fail(self, task_id: str, error: str) -> Optional[Task]:
        """Mark a task as failed."""
        task = self._running.pop(task_id, None)
        if task is None:
            return None

        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = time.time()

        # Retry if possible
        if task.retries < task.max_retries:
            task.retries += 1
            task.status = TaskStatus.PENDING
            task.started_at = None
            self._pending.append(task.id)
            self._sort_pending()
            logger.info("TaskQueue[%s]: retrying %s (attempt %d)",
                       self.name, task_id, task.retries,
                       extra={"tag": "INFRA"})
        else:
            self._completed.append(task_id)

        if self._storage_dir:
            self._persist()

        self._notify_callbacks(task)
        return task

    def cancel(self, task_id: str) -> Optional[Task]:
        """Cancel a task."""
        # Remove from pending
        if task_id in self._pending:
            self._pending.remove(task_id)

        # Mark as cancelled
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()

            if self._storage_dir:
                self._persist()

            self._notify_callbacks(task)

        return task

    def cancel_many(self, task_ids: List[str]) -> List[Optional[Task]]:
        """Cancel multiple tasks by ID."""
        return [self.cancel(tid) for tid in task_ids]

    def cancel_all(self) -> int:
        """Cancel all pending tasks. Returns count of cancelled tasks."""
        cancelled = 0
        for task_id in list(self._pending):
            self.cancel(task_id)
            cancelled += 1
        return cancelled

    def retry(self, task_id: str, reset_retries: bool = False) -> Optional[Task]:
        """Retry a failed or completed task.

        Moves the task back to PENDING status and re-queues it.

        Args:
            task_id: Task ID to retry.
            reset_retries: If True, reset retry counter to 0.

        Returns:
            The re-queued Task, or None if not found/invalid status.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None

        # Remove from completed list if present
        if task_id in self._completed:
            self._completed.remove(task_id)

        # Remove from running if present
        self._running.pop(task_id, None)

        task.status = TaskStatus.PENDING
        task.error = None
        task.started_at = None
        task.completed_at = None
        if reset_retries:
            task.retries = 0

        self._pending.append(task.id)
        self._sort_pending()

        if self._storage_dir:
            self._persist()

        logger.info("TaskQueue[%s]: retrying %s", self.name, task_id)
        return task

    def retry_all(self, reset_retries: bool = False) -> int:
        """Retry all failed tasks. Returns count of re-queued tasks."""
        retried = 0
        for task in list(self._tasks.values()):
            if task.status == TaskStatus.FAILED:
                self.retry(task.id, reset_retries=reset_retries)
                retried += 1
        return retried

    def submit_batch(self, items: List[Dict[str, Any]],
                     priority: TaskPriority = TaskPriority.NORMAL) -> List[Task]:
        """Create and submit multiple tasks from dicts.

        Each dict must have at least a "name" key. Optional: "data", "tree_id", "metadata".

        Args:
            items: List of dicts with task parameters.
            priority: Default priority for all tasks (overridden by per-item "priority" if present).

        Returns:
            List of submitted Task objects.
        """
        tasks = []
        for item in items:
            task = Task(
                name=item.get("name", ""),
                data=item.get("data"),
                priority=TaskPriority(item["priority"]) if "priority" in item else priority,
                tree_id=item.get("tree_id"),
                metadata=item.get("metadata", {}),
            )
            self.submit(task)
            tasks.append(task)
        return tasks

    def pause(self) -> None:
        """Pause the queue (no new tasks dispatched)."""
        self._paused = True

    def resume(self) -> None:
        """Resume the queue."""
        self._paused = False

    def register_handler(self, task_name: str, handler: Callable) -> None:
        """Register a handler for a task type."""
        self._handlers[task_name] = handler

    def on_complete(self, callback: Callable) -> None:
        """Register a completion callback."""
        self._callbacks.append(callback)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List tasks, optionally filtered by status."""
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    def wait_for(self, task_id: str, timeout: Optional[float] = None) -> Optional[Task]:
        """Wait for a specific task to complete.

        Args:
            task_id: Task ID to wait for.
            timeout: Maximum seconds to wait. None = wait forever.

        Returns:
            The completed/failed/cancelled Task, or None if timeout.
        """
        import threading as _threading
        deadline = time.time() + timeout if timeout else None

        while True:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return task
            if deadline and time.time() > deadline:
                return None
            _threading.Event().wait(0.05)

    def wait_all(self, timeout: Optional[float] = None) -> List[Task]:
        """Wait for all running/pending tasks to complete.

        Args:
            timeout: Maximum seconds to wait. None = wait forever.

        Returns:
            List of completed/failed/cancelled Tasks.
        """
        import threading as _threading
        deadline = time.time() + timeout if timeout else None

        while True:
            has_pending = len(self._pending) > 0
            has_running = len(self._running) > 0
            if not has_pending and not has_running:
                break
            if deadline and time.time() > deadline:
                break
            _threading.Event().wait(0.05)

        return [t for t in self._tasks.values()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)]

    def stats(self) -> dict:
        """Queue statistics."""
        result = {
            "name": self.name,
            "total": len(self._tasks),
            "pending": len(self._pending),
            "running": len(self._running),
            "completed": sum(1 for t in self._tasks.values()
                           if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self._tasks.values()
                         if t.status == TaskStatus.FAILED),
            "cancelled": sum(1 for t in self._tasks.values()
                           if t.status == TaskStatus.CANCELLED),
            "paused": self._paused,
            "handlers": list(self._handlers.keys()),
        }
        if self._worker_queue is not None:
            result["workers"] = {
                "num_workers": self._num_workers,
                "active": self.workers_active,
                "queue_depth": self.workers_queue_depth,
                **self.workers_metrics,
            }
        return result

    def clear_completed(self) -> int:
        """Remove completed tasks. Returns count removed."""
        count = 0
        for task_id in self._completed[:]:
            task = self._tasks.pop(task_id, None)
            if task:
                count += 1
        self._completed.clear()

        if self._storage_dir:
            self._persist()

        return count

    def save(self, path: Optional[Path] = None) -> Path:
        """Save queue to disk."""
        if path is None:
            if self._storage_dir is None:
                raise ValueError("No storage_dir set")
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            path = self._storage_dir / f"{self.name}.tasks.json"

        data = {
            "name": self.name,
            "tasks": [t.to_dict() for t in self._tasks.values()],
            "pending": self._pending,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "TaskQueue":
        """Load queue from disk."""
        data = json.loads(path.read_text())
        q = cls(name=data["name"])
        for td in data.get("tasks", []):
            task = Task.from_dict(td)
            q._tasks[task.id] = task
        q._pending = data.get("pending", [])
        return q

    def _sort_pending(self) -> None:
        """Sort pending tasks by priority (highest first), then by creation time."""
        self._pending.sort(
            key=lambda tid: (
                -self._tasks[tid].priority.value,
                self._tasks[tid].created_at,
            ),
            reverse=False,
        )

    def _persist(self) -> None:
        """Save state to disk."""
        try:
            self.save()
        except Exception as e:
            logger.warning("TaskQueue[%s]: persist failed: %s", self.name, e,
                extra={"tag": "INFRA"})

    def _notify_callbacks(self, task: Task) -> None:
        """Notify completion callbacks."""
        for cb in self._callbacks:
            try:
                cb(task)
            except Exception as e:
                logger.warning("TaskQueue[%s]: callback error: %s", self.name, e,
                    extra={"tag": "INFRA"})

    # ── Worker pool (ProducerConsumerQueue-backed) ────────────────────

    def start_workers(self, num_workers: int = 2, max_queue: int = 128) -> None:
        """Start worker threads that automatically process submitted tasks.

        Uses ProducerConsumerQueue for bounded execution with backpressure.
        Tasks submitted after ``start_workers()`` are dispatched to workers
        automatically — no need to call ``next()`` / ``complete()`` manually.

        Args:
            num_workers: Number of consumer threads.
            max_queue: Maximum pending tasks before backpressure kicks in.
        """
        if self._worker_queue is not None:
            return  # already running

        from domains.infrastructure.producer_consumer import ProducerConsumerQueue

        self._num_workers = num_workers
        self._worker_queue = ProducerConsumerQueue[Task](
            maxsize=max_queue,
            num_consumers=num_workers,
            handler=self._execute_task,
            name=f"pgq-task-{self.name}",
        )
        self._worker_queue.start()
        logger.info("TaskQueue[%s]: started %d workers (max_queue=%d)",
                     self.name, num_workers, max_queue,
                     extra={"tag": "INFRA"})

    def stop_workers(self, timeout: float = 5.0) -> None:
        """Stop worker threads gracefully (drains pending tasks)."""
        if self._worker_queue is not None:
            self._worker_queue.stop(timeout=timeout)
            self._worker_queue = None
            self._num_workers = 0
            logger.info("TaskQueue[%s]: workers stopped", self.name,
                         extra={"tag": "INFRA"})

    def _execute_task(self, task: Task) -> None:
        """Worker callback: look up handler and execute the task."""
        if self._paused:
            return

        # Move task to running state
        if task.id in self._pending:
            self._pending.remove(task.id)
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._running[task.id] = task

        handler = self._handlers.get(task.name)
        if handler is None:
            logger.warning("TaskQueue[%s]: no handler for task '%s'",
                           self.name, task.name)
            self.fail(task.id, error=f"No handler registered for '{task.name}'")
            return

        try:
            result = handler(task)
            self.complete(task.id, result=result)
        except Exception as e:
            self.fail(task.id, error=str(e))

    @property
    def workers_active(self) -> int:
        """Number of active worker threads."""
        if self._worker_queue is None:
            return 0
        return self._worker_queue.active_consumers

    @property
    def workers_queue_depth(self) -> int:
        """Number of tasks waiting in the worker queue."""
        if self._worker_queue is None:
            return 0
        return self._worker_queue.qsize

    @property
    def workers_metrics(self) -> dict:
        """Worker queue metrics."""
        if self._worker_queue is None:
            return {}
        return self._worker_queue.metrics
