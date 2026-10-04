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

        try:
            yield hook
        except Exception as e:
            hook.success = False
            # Bare TimeoutError stringifies to "" — keep the class name.
            hook.error = str(e) or type(e).__name__
            raise
        finally:
            hook.end_time = time.perf_counter()
            hook.duration_ms = (hook.end_time - hook.start_time) * 1000

            if self._current_stage:
                self._current_stage.hooks.append(hook)

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
        """Get a human-readable summary of the startup profile.

        Non-mutating: safe to call while a stage is still running (the
        ``health.startup_profile`` endpoint hits it mid-boot). The previous
        implementation called ``finish()``, which closed the *open* stage —
        so any health call during boot silently dropped in-flight hook
        timings. An open stage derives its totals from the hooks recorded
        so far instead.
        """
        stages: list[dict] = []
        total_ms = 0.0
        total_hooks = 0
        for stage in self._profile.stages:
            hooks = stage.hooks
            ms = stage.total_duration_ms or sum(h.duration_ms for h in hooks)
            count = stage.hook_count or len(hooks)
            slowest = stage.slowest_hook or (
                max(hooks, key=lambda h: h.duration_ms).name if hooks else ""
            )
            total_ms += ms
            total_hooks += count
            stages.append(
                {
                    "name": stage.name,
                    "ms": round(ms, 1),
                    "hooks": count,
                    "slowest": slowest,
                }
            )

        return {"total_ms": round(total_ms, 1), "total_hooks": total_hooks, "stages": stages}


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


def profile_hook(hook_name: str) -> Generator[HookMetrics, None, None]:
    """Profile a hook on the global profiler (module-level convenience).

    Matches the usage documented at the top of this module:
    ``with profile_hook("db_pool") as hook: ...`` — previously the import
    failed because only the bound method existed.
    """
    return get_profiler().profile_hook(hook_name)
