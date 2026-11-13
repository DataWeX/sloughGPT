"""Task — a named sequence of steps with preconditions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class TaskStatus(Enum):
    """Outcome of a task run."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """One step: run action if precondition holds.

    action: async callable taking the context dict.
    precondition: sync or async callable; step is skipped when False.
    optional: failure does not fail the task.
    """

    name: str
    action: Callable[[dict[str, Any]], Awaitable[Any]]
    precondition: Callable[[dict[str, Any]], Any] | None = None
    optional: bool = False


@dataclass
class TaskResult:
    """Outcome of Task.execute."""

    task_name: str
    status: TaskStatus = TaskStatus.PASSED
    steps_passed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "status": self.status.value,
            "steps_passed": self.steps_passed,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class Task:
    """A named journey: steps run in order against a context dict.

    Usage::

        task = Task("chat_opens", steps=[
            TaskStep("goto", lambda ctx: ctx["arken"].goto("/chat")),
        ])
        result = await task.execute({"arken": session})
    """

    name: str
    description: str = ""
    steps: list[TaskStep] = field(default_factory=list)

    def add_step(self, step: TaskStep) -> Task:
        self.steps.append(step)
        return self

    async def execute(self, context: dict[str, Any] | None = None) -> TaskResult:
        ctx = context if context is not None else {}
        result = TaskResult(task_name=self.name)
        start = time.perf_counter()
        for step in self.steps:
            if step.precondition is not None:
                try:
                    pre = step.precondition(ctx)
                    if hasattr(pre, "__await__"):
                        pre = await pre
                    if not pre:
                        result.steps_skipped += 1
                        continue
                except Exception as e:
                    result.steps_skipped += 1
                    if not step.optional:
                        result.status = TaskStatus.FAILED
                        result.error = f"precondition '{step.name}' failed: {e}"
                        break
                    continue
            try:
                await step.action(ctx)
                result.steps_passed += 1
            except Exception as e:
                result.steps_failed += 1
                if not step.optional:
                    result.status = TaskStatus.FAILED
                    result.error = f"step '{step.name}' failed: {e}"
                    break
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result
