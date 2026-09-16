"""Startup profiler — detailed timing breakdowns for startup hooks.

Usage:
    from infrastructure.startup_profiler import get_profiler, profile_hook

    profiler = get_profiler()
    with profile_hook("db_pool") as hook:
        await init_db_pool()
        hook.add_metric("connections", 5)
        hook.add_metric("warm_time_ms", 12.3)
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HookMetrics:
    """Detailed metrics for a single hook execution."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, key: str, value: Any) -> None:
        """Add a custom metric."""
        self.metrics[key] = value

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class StageProfile:
    """Profile data for a startup stage."""

    name: str
    hooks: list[HookMetrics] = field(default_factory=list)
    total_duration_ms: float = 0.0
    hook_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    slowest_hook: str = ""
    fastest_hook: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "hook_count": self.hook_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "slowest_hook": self.slowest_hook,
            "fastest_hook": self.fastest_hook,
            "hooks": [h.to_dict() for h in self.hooks],
        }


@dataclass
class StartupProfile:
    """Complete startup profile with all stages."""

    stages: list[StageProfile] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_hooks: int = 0
    total_success: int = 0
    total_failure: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_hooks": self.total_hooks,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "timestamp": self.timestamp,
            "stages": [s.to_dict() for s in self.stages],
        }


class StartupProfiler:
    """Tracks detailed timing for every hook in every stage."""

    def __init__(self) -> None:
        self._profile = StartupProfile(timestamp=time.time())
        self._current_stage: StageProfile | None = None
        self._current_hook: HookMetrics | None = None

    def start_stage(self, stage_name: str) -> None:
        """Begin profiling a new stage."""
        stage = StageProfile(name=stage_name)
        self._profile.stages.append(stage)
        self._current_stage = stage

    def finish_stage(self) -> None:
        """Finish the current stage."""
        if self._current_stage:
            hooks = self._current_stage.hooks
            if hooks:
                durations = [h.duration_ms for h in hooks]
                self._current_stage.total_duration_ms = sum(durations)
                self._current_stage.hook_count = len(hooks)
                self._current_stage.success_count = sum(1 for h in hooks if h.success)
                self._current_stage.failure_count = sum(1 for h in hooks if not h.success)
                slowest = max(hooks, key=lambda h: h.duration_ms)
                fastest = min(hooks, key=lambda h: h.duration_ms)
                self._current_stage.slowest_hook = slowest.name
                self._current_stage.fastest_hook = fastest.name
            self._current_stage = None

    @contextmanager
    def profile_hook(self, hook_name: str) -> Generator[HookMetrics, None, None]:
        """Context manager to profile a hook."""
        hook = HookMetrics(name=hook_name, start_time=time.perf_counter())
        self._current_hook = hook

        try:
            yield hook
        except Exception as e:
            hook.success = False
            hook.error = str(e)
            raise
        finally:
            hook.end_time = time.perf_counter()
            hook.duration_ms = (hook.end_time - hook.start_time) * 1000

            if self._current_stage:
                self._current_stage.hooks.append(hook)

            self._current_hook = None

            if hook.success:
                logger.debug("Hook %s completed in %.1fms", hook_name, hook.duration_ms)
            else:
                logger.warning(
                    "Hook %s failed after %.1fms: %s", hook_name, hook.duration_ms, hook.error
                )

    def finish(self) -> StartupProfile:
        """Finish profiling and return the complete profile."""
        self.finish_stage()
        total_hooks = sum(s.hook_count for s in self._profile.stages)
        total_success = sum(s.success_count for s in self._profile.stages)
        total_failure = sum(s.failure_count for s in self._profile.stages)
        total_duration = sum(s.total_duration_ms for s in self._profile.stages)

        self._profile.total_hooks = total_hooks
        self._profile.total_success = total_success
        self._profile.total_failure = total_failure
        self._profile.total_duration_ms = total_duration

        logger.info(
            "Startup profile: %d hooks, %dms total (%d success, %d failure)",
            total_hooks,
            total_duration,
            total_success,
            total_failure,
        )
        return self._profile

    def get_profile(self) -> StartupProfile:
        """Get current profile (may be in-progress)."""
        return self._profile

    def get_summary(self) -> dict:
        """Get a human-readable summary of the startup profile."""
        profile = self.finish()
        summary = {
            "total_ms": round(profile.total_duration_ms, 1),
            "total_hooks": profile.total_hooks,
            "stages": [],
        }

        for stage in profile.stages:
            stage_summary = {
                "name": stage.name,
                "ms": round(stage.total_duration_ms, 1),
                "hooks": stage.hook_count,
                "slowest": stage.slowest_hook,
            }
            summary["stages"].append(stage_summary)

        return summary


_global_profiler: StartupProfiler | None = None


def get_profiler() -> StartupProfiler:
    """Get the global startup profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = StartupProfiler()
    return _global_profiler


def reset_profiler() -> StartupProfiler:
    """Reset the global profiler."""
    global _global_profiler
    _global_profiler = StartupProfiler()
    return _global_profiler
