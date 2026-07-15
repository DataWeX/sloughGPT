"""
Task queue — manages tasks with priority, routing, and persistence.

Provides:
  - Priority-based task ordering
  - Tree/instance routing
  - Persistence to disk
  - Pause/resume/cancel support
  - Event callbacks
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
    """Priority task queue with persistence and routing."""

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

    def submit(self, task: Task) -> Task:
        """Submit a task to the queue."""
        if len(self._tasks) >= self._max_size:
            raise ValueError(f"Queue full (max {self._max_size} tasks)")

        self._tasks[task.id] = task
        self._pending.append(task.id)
        self._sort_pending()

        if self._storage_dir:
            self._persist()

        logger.debug("TaskQueue[%s]: submitted %s (%s)", self.name, task.id, task.name)
        return task

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

    def stats(self) -> dict:
        """Queue statistics."""
        return {
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
