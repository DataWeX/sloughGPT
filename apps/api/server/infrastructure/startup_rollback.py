"""Startup rollback — provides rollback capabilities when startup fails.

When a startup fails, the rollback system can:
1. Record the failure for diagnostics
2. Trigger cleanup of partial state
3. Optionally restart with a minimal configuration

Usage:
    from infrastructure.startup_rollback import get_startup_rollback

    rollback = get_startup_rollback()
    rollback.record_failure("model_load", "CUDA out of memory")
    rollback.trigger_rollback()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("slo.startup")


@dataclass
class RollbackAction:
    """A single rollback action."""

    name: str
    description: str
    action: callable
    priority: int = 0  # Higher = executed first


@dataclass
class RollbackState:
    """Current rollback state."""

    triggered: bool = False
    failure_hook: str | None = None
    failure_error: str | None = None
    failure_time: float = 0.0
    actions_executed: list[str] = field(default_factory=list)
    actions_failed: list[str] = field(default_factory=list)


class StartupRollback:
    """Manages startup rollback on failure.

    Registers cleanup actions that can be triggered when startup fails.
    Actions are executed in priority order (highest first).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._actions: list[RollbackAction] = []
        self._state = RollbackState()
        self._enabled = True

    @property
    def triggered(self) -> bool:
        return self._state.triggered

    @property
    def state(self) -> RollbackState:
        return self._state

    def register_action(
        self,
        name: str,
        description: str,
        action: callable,
        priority: int = 0,
    ) -> None:
        """Register a rollback action."""
        with self._lock:
            self._actions.append(RollbackAction(
                name=name,
                description=description,
                action=action,
                priority=priority,
            ))
            # Sort by priority (highest first)
            self._actions.sort(key=lambda a: a.priority, reverse=True)

    def record_failure(self, hook_name: str, error: str) -> None:
        """Record a startup failure."""
        with self._lock:
            self._state.failure_hook = hook_name
            self._state.failure_error = error
            self._state.failure_time = time.time()
            logger.warning(
                "Startup failure recorded: hook=%s error=%s",
                hook_name,
                error,
                extra={"tag": "START"},
            )

    def trigger_rollback(self) -> dict:
        """Execute all rollback actions.

        Returns:
            Summary of rollback execution.
        """
        with self._lock:
            if self._state.triggered:
                return {"status": "already_triggered"}

            self._state.triggered = True
            logger.info(
                "Startup rollback triggered (failure: %s)",
                self._state.failure_hook,
                extra={"tag": "START"},
            )

        results = []
        for action in self._actions:
            try:
                logger.info("Executing rollback action: %s", action.name, extra={"tag": "START"})
                action.action()
                with self._lock:
                    self._state.actions_executed.append(action.name)
                results.append({"action": action.name, "status": "ok"})
            except Exception as e:
                logger.error(
                    "Rollback action '%s' failed: %s",
                    action.name,
                    e,
                    extra={"tag": "START"},
                )
                with self._lock:
                    self._state.actions_failed.append(action.name)
                results.append({"action": action.name, "status": "error", "error": str(e)})

        return {
            "status": "completed",
            "failure_hook": self._state.failure_hook,
            "failure_error": self._state.failure_error,
            "actions_executed": self._state.actions_executed,
            "actions_failed": self._state.actions_failed,
            "results": results,
        }

    def get_status(self) -> dict:
        """Get rollback status."""
        return {
            "enabled": self._enabled,
            "triggered": self._state.triggered,
            "failure_hook": self._state.failure_hook,
            "failure_error": self._state.failure_error,
            "failure_time": self._state.failure_time,
            "actions_registered": len(self._actions),
            "actions_executed": self._state.actions_executed,
            "actions_failed": self._state.actions_failed,
        }

    def disable(self) -> None:
        """Disable rollback."""
        self._enabled = False

    def enable(self) -> None:
        """Enable rollback."""
        self._enabled = True


# ── Singleton ───────────────────────────────────────────────────────
_rollback: StartupRollback | None = None
_rollback_lock = threading.Lock()


def get_startup_rollback() -> StartupRollback:
    """Return the global startup rollback instance."""
    global _rollback
    if _rollback is None:
        with _rollback_lock:
            if _rollback is None:
                _rollback = StartupRollback()
    return _rollback
