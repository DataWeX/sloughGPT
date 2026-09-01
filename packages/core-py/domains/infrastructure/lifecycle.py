"""
Lifecycle Manager — ordered startup/shutdown, health gates, graceful drain.

Manages the application lifecycle through phases:

  INIT → STARTING → RUNNING ↔ DRAINING → STOPPING → STOPPED

Components register startup/shutdown hooks with dependency ordering.
Health gates prevent the phase from advancing before prereqs are met.
All phase transitions emit typed events on the EventBus.

Usage::

    mgr = get_lifecycle_manager()

    mgr.register_startup_hook(StartupHook("db", connect_db, depends_on=[]))
    mgr.register_startup_hook(StartupHook("model", load_model, depends_on=["db"]))

    await mgr.start()           # runs hooks in dependency order
    assert mgr.phase == LifecyclePhase.RUNNING

    await mgr.shutdown()        # reverses hook order, graceful drain
    assert mgr.phase == LifecyclePhase.STOPPED
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

logger = logging.getLogger("slo.lifecycle")


class LifecyclePhase(str, Enum):
    """Every phase the application can be in, from birth to shutdown."""

    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"


class StartupProfile(str, Enum):
    """Startup profile controlling which hooks run during startup.

    Profiles allow the server to start with different sets of
    components, trading capability for speed.

    * ``full`` — all components: AI model, W&B, multimodal, TaskQueue, Config
    * ``quick`` — everything except the AI model load (for dev/testing)
    * ``minimal`` — only core infrastructure (logging, registry, routers)
    """

    FULL = "full"
    QUICK = "quick"
    MINIMAL = "minimal"

    @classmethod
    def from_env(cls) -> Self:
        """Read profile from ``SLO_STARTUP_PROFILE`` env var (default ``full``)."""
        raw = os.environ.get("SLO_STARTUP_PROFILE", "full").strip().lower()
        try:
            return cls(raw)
        except ValueError:
            logger.warning("Unknown startup profile %r — falling back to full", raw,
                extra={"tag": "INFRA"})
            return cls.FULL


ALL_PROFILES: frozenset[StartupProfile] = frozenset(StartupProfile)


# ── Hook types ──


@dataclass
class StartupHook:
    """A single startup task.

    Args:
        name: Unique identifier for this hook.
        handler: Async callable. Receives no arguments.
        depends_on: Names of hooks that must complete before this one.
        timeout: Maximum seconds to wait for the handler.
        critical: If True, a failure transitions the phase to CRASHED.
        profiles: Set of profiles this hook belongs to. Empty means all profiles.
    """

    name: str
    handler: Callable[[], Awaitable[Any]]
    depends_on: list[str] = field(default_factory=list)
    timeout: float = 30.0
    critical: bool = True
    profiles: frozenset[StartupProfile] = field(default_factory=lambda: ALL_PROFILES)


@dataclass
class ShutdownHook:
    """A single shutdown task (reverse dependency order of startup)."""

    name: str
    handler: Callable[[], Awaitable[Any]]
    depends_on: list[str] = field(default_factory=list)
    timeout: float = 30.0
    critical: bool = False


@dataclass
class _HookResult:
    """Outcome of running a single hook."""

    name: str
    success: bool
    elapsed: float
    error: str = ""


# ── Topological sort ──


def _topological_sort(hooks: list[StartupHook]) -> list[StartupHook]:
    """Return hooks in dependency order using Kahn's algorithm."""
    by_name: dict[str, StartupHook] = {h.name: h for h in hooks}
    in_degree: dict[str, int] = {h.name: 0 for h in hooks}

    for h in hooks:
        for dep in h.depends_on:
            if dep in by_name:
                in_degree[h.name] = in_degree.get(h.name, 0) + 1

    queue = [h for h in hooks if in_degree.get(h.name, 0) == 0]
    ordered: list[StartupHook] = []

    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for h in hooks:
            if node.name in h.depends_on:
                in_degree[h.name] -= 1
                if in_degree[h.name] == 0:
                    queue.append(h)

    if len(ordered) != len(hooks):
        # Cycle detected — return best-effort order with remaining hooks appended
        remaining = [h for h in hooks if h not in ordered]
        ordered.extend(remaining)
        logger.warning(
            "Dependency cycle detected: %d hooks unordered",
            len(remaining),
            extra={"tag": "INFRA"},
        )
    return ordered


def _dependency_levels(hooks: list[StartupHook]) -> list[list[StartupHook]]:
    """Group hooks into dependency levels for parallel execution.

    Level 0 = no dependencies, level 1 = depends only on level 0, etc.
    Hooks within the same level can run concurrently.
    """
    by_name: dict[str, StartupHook] = {h.name: h for h in hooks}
    resolved: set[str] = set()
    levels: list[list[StartupHook]] = []
    remaining = list(hooks)

    while remaining:
        # Find hooks whose deps are all resolved
        level = [h for h in remaining if all(d in resolved or d not in by_name for d in h.depends_on)]
        if not level:
            # Break cycle: put remaining in one level
            level = list(remaining)
            logger.warning("Dependency cycle in parallel startup — running remaining hooks sequentially",
                           extra={"tag": "INFRA"})
        levels.append(level)
        for h in level:
            resolved.add(h.name)
            remaining.remove(h)

    return levels


# ── Event names ──

EVT_PHASE_CHANGED = "lifecycle.phase_changed"
EVT_HOOK_STARTED = "lifecycle.hook_started"
EVT_HOOK_COMPLETED = "lifecycle.hook_completed"
EVT_HOOK_FAILED = "lifecycle.hook_failed"


# ── Lifecycle Manager ──


class LifecycleManager:
    """Ordered startup/shutdown with health gates, graceful drain, and EventBus integration.

    Thread-safe: all state mutations are protected by an asyncio lock.
    """

    def __init__(self, event_bus: Any | None = None) -> None:
        self._phase: LifecyclePhase = LifecyclePhase.INIT
        self._startup_hooks: list[StartupHook] = []
        self._shutdown_hooks: list[ShutdownHook] = []
        self._health_gates: dict[str, Callable[[], bool]] = {}
        self._gate_results: dict[str, bool] = {}
        self._lock = asyncio.Lock()
        self._event_bus = event_bus
        self._started_at: float = 0.0
        self._drain_event = asyncio.Event()
        self._drain_event.set()  # not draining by default
        self._in_flight_count: int = 0
        self._in_flight_lock = asyncio.Lock()
        self._profile: StartupProfile = StartupProfile.FULL
        self._last_startup_results: list[_HookResult] = []
        self._last_shutdown_results: list[_HookResult] = []

    # ── Properties ──

    @property
    def phase(self) -> LifecyclePhase:
        return self._phase

    @property
    def started_at(self) -> float:
        return self._started_at

    @property
    def uptime_seconds(self) -> float:
        if self._started_at == 0.0:
            return 0.0
        return time.time() - self._started_at

    def is_running(self) -> bool:
        """True when the app is in RUNNING phase."""
        return self._phase == LifecyclePhase.RUNNING

    def is_draining(self) -> bool:
        """True during DRAINING or STOPPING — new work should be rejected."""
        return self._phase in (LifecyclePhase.DRAINING, LifecyclePhase.STOPPING)

    def get_profile(self) -> StartupProfile:
        """Return the active startup profile."""
        return self._profile

    def preview(self, profile: StartupProfile | None = None) -> list[dict[str, Any]]:
        """Preview which hooks would run for a given profile.

        Args:
            profile: The profile to preview. Defaults to the active profile.

        Returns:
            List of hook dicts with name, critical, timeout, and depends_on fields.
        """
        target = profile if profile is not None else self._profile
        filtered = self._filter_hooks_for_profile(self._startup_hooks, target)
        ordered = _topological_sort(filtered)
        return [
            {
                "name": h.name,
                "critical": h.critical,
                "timeout": h.timeout,
                "depends_on": h.depends_on,
            }
            for h in ordered
        ]

    # ── Phase transitions ──

    async def _set_phase(self, phase: LifecyclePhase) -> None:
        """Set the phase and emit an event."""
        old = self._phase
        self._phase = phase
        logger.info("Lifecycle: %s → %s", old.value, phase.value,
            extra={"tag": "INFRA"})
        if self._event_bus is not None:
            await self._event_bus.emit(
                EVT_PHASE_CHANGED,
                {"from": old.value, "to": phase.value},
                source="lifecycle",
            )

    # ── Hook registration ──

    def register_startup_hook(self, hook: StartupHook) -> None:
        """Register a startup hook (duplicate names are silently skipped)."""
        if any(h.name == hook.name for h in self._startup_hooks):
            logger.warning("Duplicate startup hook: %s — skipping", hook.name,
                extra={"tag": "INFRA"})
            return
        self._startup_hooks.append(hook)

    def register_shutdown_hook(self, hook: ShutdownHook) -> None:
        """Register a shutdown hook (duplicate names silently skipped)."""
        if any(h.name == hook.name for h in self._shutdown_hooks):
            logger.warning("Duplicate shutdown hook: %s — skipping", hook.name,
                extra={"tag": "INFRA"})
            return
        self._shutdown_hooks.append(hook)

    # ── Health gates ──

    def register_gate(self, name: str, check: Callable[[], bool]) -> None:
        """Register a health gate. The gate check returns True when ready."""
        self._health_gates[name] = check

    def unregister_gate(self, name: str) -> None:
        """Remove a previously registered gate."""
        self._health_gates.pop(name, None)

    def gate_ready(self, name: str) -> bool:
        """Check if a specific gate reports ready."""
        check = self._health_gates.get(name)
        if check is None:
            return True
        ready = check()
        self._gate_results[name] = ready
        return ready

    def gates_ready(self) -> bool:
        """Check all registered gates. Returns True only when all pass."""
        all_ready = True
        for name in list(self._health_gates.keys()):
            if not self.gate_ready(name):
                all_ready = False
        return all_ready

    async def wait_for_gates(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        """Poll all gates until they pass or timeout expires.

        Returns True if all gates passed, False on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.gates_ready():
                if self._event_bus is not None:
                    await self._event_bus.emit(
                        "lifecycle.gates_passed",
                        {},
                        source="lifecycle",
                    )
                return True
            await asyncio.sleep(poll_interval)
        logger.warning("Health gates not ready after %.0fs", timeout,
            extra={"tag": "INFRA"})
        return False

    # ── In-flight task tracking (for graceful drain) ──

    async def acquire_in_flight(self) -> bool:
        """Mark a new in-flight task. Returns False if draining."""
        async with self._in_flight_lock:
            if self.is_draining():
                return False
            self._in_flight_count += 1
            return True

    async def release_in_flight(self) -> None:
        """Mark an in-flight task as complete."""
        async with self._in_flight_lock:
            self._in_flight_count = max(0, self._in_flight_count - 1)
            if self._in_flight_count == 0:
                self._drain_event.set()

    @property
    def in_flight_count(self) -> int:
        return self._in_flight_count

    @property
    def startup_results(self) -> list[dict[str, Any]]:
        """Results from the most recent startup run."""
        return [
            {"name": r.name, "success": r.success, "elapsed": round(r.elapsed, 3), "error": r.error}
            for r in self._last_startup_results
        ]

    @property
    def shutdown_results(self) -> list[dict[str, Any]]:
        """Results from the most recent shutdown run."""
        return [
            {"name": r.name, "success": r.success, "elapsed": round(r.elapsed, 3), "error": r.error}
            for r in self._last_shutdown_results
        ]

    async def _wait_for_drain(self, timeout: float = 30.0) -> bool:
        """Wait for all in-flight tasks to complete.

        Returns True if drained, False on timeout.
        """
        if self._in_flight_count == 0:
            return True
        logger.info("Draining: waiting for %d in-flight tasks", self._in_flight_count,
            extra={"tag": "INFRA"})
        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Drain timeout after %.0fs with %d tasks still in-flight",
                timeout,
                self._in_flight_count,
                extra={"tag": "INFRA"},
            )
            return False

    def _filter_hooks_for_profile(
        self,
        hooks: list[StartupHook],
        profile: StartupProfile,
    ) -> list[StartupHook]:
        """Return only hooks whose profiles include the given profile."""
        result: list[StartupHook] = []
        for h in hooks:
            if profile in h.profiles:
                result.append(h)
            else:
                logger.debug(
                    "Skipping startup hook %s (not in profile %s)",
                    h.name,
                    profile.value,
                )
        return result

    # ── Startup ──

    async def start(self, timeout: float = 60.0, profile: StartupProfile | None = None) -> bool:
        """Run all startup hooks in dependency order and transition to RUNNING.

        Args:
            timeout: Overall timeout for all hooks combined.

        Returns:
            True if all hooks (including non-critical) succeeded. Even if a
            non-critical hook fails, the system transitions to RUNNING.
        """
        async with self._lock:
            if self._phase != LifecyclePhase.INIT:
                logger.warning(
                    "start() called in phase %s — ignoring", self._phase.value,
                    extra={"tag": "INFRA"},
                )
                return self._phase == LifecyclePhase.RUNNING

            self._profile = profile or StartupProfile.from_env()
            await self._set_phase(LifecyclePhase.STARTING)
            self._started_at = time.time()
            self._drain_event = asyncio.Event()

        filtered = self._filter_hooks_for_profile(self._startup_hooks, self._profile)
        levels = _dependency_levels(filtered)
        results: list[_HookResult] = []
        all_ok = True

        for level in levels:
            if len(level) == 1:
                hook = level[0]
                self._emit_sync(EVT_HOOK_STARTED, {"hook": hook.name})
                started = time.time()
                try:
                    await asyncio.wait_for(hook.handler(), timeout=hook.timeout)
                    elapsed = time.time() - started
                    results.append(_HookResult(name=hook.name, success=True, elapsed=elapsed))
                    self._emit_sync(EVT_HOOK_COMPLETED, {"hook": hook.name, "elapsed": round(elapsed, 3)})
                    logger.info("Startup hook %s completed in %.2fs", hook.name, elapsed,
                        extra={"tag": "INFRA"})
                except asyncio.TimeoutError:
                    elapsed = time.time() - started
                    error = f"timed out after {hook.timeout:.0f}s"
                    results.append(_HookResult(name=hook.name, success=False, elapsed=elapsed, error=error))
                    self._emit_sync(EVT_HOOK_FAILED, {"hook": hook.name, "error": error, "elapsed": round(elapsed, 3)})
                    logger.error("Startup hook %s %s", hook.name, error,
                        extra={"tag": "INFRA"})
                    if hook.critical:
                        all_ok = False
                        break
                except Exception as exc:
                    elapsed = time.time() - started
                    error = str(exc)
                    results.append(_HookResult(name=hook.name, success=False, elapsed=elapsed, error=error))
                    self._emit_sync(EVT_HOOK_FAILED, {"hook": hook.name, "error": error, "elapsed": round(elapsed, 3)})
                    logger.error("Startup hook %s failed: %s", hook.name, exc,
                        extra={"tag": "INFRA"})
                    if hook.critical:
                        all_ok = False
                        break
            else:
                # Run independent hooks in parallel
                names = [h.name for h in level]
                logger.info("Startup: running %d hooks in parallel: %s", len(level), ", ".join(names),
                    extra={"tag": "INFRA"})
                level_started = time.time()

                async def _run_hook(h: StartupHook) -> _HookResult:
                    self._emit_sync(EVT_HOOK_STARTED, {"hook": h.name})
                    started = time.time()
                    try:
                        await asyncio.wait_for(h.handler(), timeout=h.timeout)
                        elapsed = time.time() - started
                        self._emit_sync(EVT_HOOK_COMPLETED, {"hook": h.name, "elapsed": round(elapsed, 3)})
                        logger.info("Startup hook %s completed in %.2fs", h.name, elapsed,
                            extra={"tag": "INFRA"})
                        return _HookResult(name=h.name, success=True, elapsed=elapsed)
                    except asyncio.TimeoutError:
                        elapsed = time.time() - started
                        error = f"timed out after {h.timeout:.0f}s"
                        self._emit_sync(EVT_HOOK_FAILED, {"hook": h.name, "error": error, "elapsed": round(elapsed, 3)})
                        logger.error("Startup hook %s %s", h.name, error, extra={"tag": "INFRA"})
                        return _HookResult(name=h.name, success=False, elapsed=elapsed, error=error)
                    except Exception as exc:
                        elapsed = time.time() - started
                        error = str(exc)
                        self._emit_sync(EVT_HOOK_FAILED, {"hook": h.name, "error": error, "elapsed": round(elapsed, 3)})
                        logger.error("Startup hook %s failed: %s", h.name, exc, extra={"tag": "INFRA"})
                        return _HookResult(name=h.name, success=False, elapsed=elapsed, error=error)

                level_results = await asyncio.gather(*[_run_hook(h) for h in level])
                level_elapsed = time.time() - level_started
                logger.info("Startup: parallel level (%s) completed in %.2fs",
                    ", ".join(names), level_elapsed, extra={"tag": "INFRA"})
                results.extend(level_results)

                # Check critical failures
                for r in level_results:
                    if not r.success:
                        hook = next(h for h in level if h.name == r.name)
                        if hook.critical:
                            all_ok = False
                            break
                if not all_ok:
                    break

        self._last_startup_results = results
        async with self._lock:
            if all_ok:
                await self._set_phase(LifecyclePhase.RUNNING)
                self._emit_sync("lifecycle.started", {"elapsed": round(time.time() - self._started_at, 3)})
                return True
            await self._set_phase(LifecyclePhase.CRASHED)
            self._emit_sync(
                "lifecycle.crashed",
                {
                    "failed_hook": results[-1].name if results else "?",
                    "error": results[-1].error if results else "unknown",
                },
            )
            return False

    # ── Shutdown ──

    async def shutdown(self, timeout: float = 60.0) -> bool:
        """Graceful drain, then run shutdown hooks in reverse startup order.

        Args:
            timeout: Overall timeout for drain + all hooks combined.

        Returns:
            True if all hooks completed (non-critical failures tolerated).
        """
        async with self._lock:
            if self._phase not in (LifecyclePhase.RUNNING, LifecyclePhase.CRASHED):
                logger.warning(
                    "shutdown() called in phase %s — ignoring", self._phase.value,
                    extra={"tag": "INFRA"},
                )
                return self._phase == LifecyclePhase.STOPPED
            await self._set_phase(LifecyclePhase.DRAINING)

        # Step 1: drain in-flight tasks
        drain_ok = await self._wait_for_drain(timeout=timeout * 0.3)

        # Step 2: run shutdown hooks
        async with self._lock:
            await self._set_phase(LifecyclePhase.STOPPING)

        # Reverse of startup order: topological sort then reverse
        shutdown_list = _topological_sort(
            [StartupHook(name=h.name, handler=h.handler, depends_on=h.depends_on, timeout=h.timeout)
             for h in self._shutdown_hooks]
        )
        shutdown_list.reverse()

        results: list[_HookResult] = []
        all_ok = True
        remaining_timeout = timeout * 0.7

        for hook in shutdown_list:
            started = time.time()
            try:
                await asyncio.wait_for(hook.handler(), timeout=min(hook.timeout, remaining_timeout))
                elapsed = time.time() - started
                results.append(_HookResult(name=hook.name, success=True, elapsed=elapsed))
                logger.info("Shutdown hook %s completed in %.2fs", hook.name, elapsed,
                    extra={"tag": "INFRA"})
            except asyncio.TimeoutError:
                elapsed = time.time() - started
                results.append(
                    _HookResult(name=hook.name, success=False, elapsed=elapsed, error=f"timeout")
                )
            except Exception as exc:
                elapsed = time.time() - started
                results.append(
                    _HookResult(name=hook.name, success=False, elapsed=elapsed, error=str(exc))
                )
                logger.exception("Shutdown hook %s failed", hook.name, extra={"tag": "INFRA"})

        self._last_shutdown_results = results
        async with self._lock:
            await self._set_phase(LifecyclePhase.STOPPED)
            self._emit_sync(
                "lifecycle.stopped",
                {
                    "drain_ok": drain_ok,
                    "drain_remaining": self._in_flight_count,
                },
            )
            return all_ok

    # ── Crash recovery ──

    async def mark_crashed(self, reason: str = "") -> None:
        """Forcibly transition to CRASHED without running shutdown hooks."""
        async with self._lock:
            if self._phase in (LifecyclePhase.STOPPED, LifecyclePhase.CRASHED):
                return
            await self._set_phase(LifecyclePhase.CRASHED)
            self._emit_sync("lifecycle.crashed", {"reason": reason})

    # ── Helpers ──

    def _emit_sync(self, event: str, data: dict[str, Any]) -> None:
        """Fire-and-forget emit on the event bus (runs in current thread)."""
        if self._event_bus is not None:
            try:
                self._event_bus.emit_sync(event, data, source="lifecycle")
            except Exception:
                logger.exception("Failed to emit lifecycle event %s", event, extra={"tag": "INFRA"})

    def get_results(self) -> dict[str, Any]:
        """Return a summary snapshot of the current lifecycle state."""
        startup_results = self.startup_results
        shutdown_results = self.shutdown_results
        return {
            "phase": self._phase.value,
            "profile": self._profile.value,
            "uptime": round(self.uptime_seconds, 1),
            "started_at": self._started_at,
            "in_flight": self._in_flight_count,
            "hooks": {
                "startup": len(self._startup_hooks),
                "shutdown": len(self._shutdown_hooks),
                "preview": self.preview(),
                "startup_results": startup_results,
                "shutdown_results": shutdown_results,
            },
            "gates": {
                "total": len(self._health_gates),
                "passed": sum(
                    1 for n in self._health_gates if self._gate_results.get(n, False)
                ),
            },
        }


# ── Singleton ──

_lifecycle_manager: Optional[LifecycleManager] = None
_lifecycle_lock = Lock()


def get_lifecycle_manager(event_bus: Any | None = None) -> LifecycleManager:
    """Get (or create) the singleton LifecycleManager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        with _lifecycle_lock:
            if _lifecycle_manager is None:
                _lifecycle_manager = LifecycleManager(event_bus=event_bus)
    return _lifecycle_manager


def reset_lifecycle_manager() -> None:
    """Reset the singleton (for testing)."""
    global _lifecycle_manager
    with _lifecycle_lock:
        _lifecycle_manager = None
