"""Comprehensive tests for lifecycle.py — LifecycleManager, LifecyclePhase,
StartupProfile, StartupHook, ShutdownHook, _topological_sort, _dependency_levels.

Covers: phase transitions, hook registration, startup/shutdown flows,
health gates, in-flight tracking, profile filtering, parallel startup,
drain, get_results, singleton reset.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from domains.infrastructure.lifecycle import (
    LifecycleManager,
    LifecyclePhase,
    StartupProfile,
    StartupHook,
    ShutdownHook,
    _HookResult,
    _topological_sort,
    _dependency_levels,
    get_lifecycle_manager,
    reset_lifecycle_manager,
    EVT_PHASE_CHANGED,
    EVT_HOOK_STARTED,
    EVT_HOOK_COMPLETED,
    EVT_HOOK_FAILED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop():
    pass


async def _slow():
    await asyncio.sleep(0.1)


async def _fail():
    raise ValueError("hook failed")


async def _slow_fail():
    await asyncio.sleep(0.05)
    raise RuntimeError("slow fail")


def _hook(name, handler=_noop, depends_on=None, timeout=5.0, critical=True, profiles=None):
    return StartupHook(
        name=name,
        handler=handler,
        depends_on=depends_on or [],
        timeout=timeout,
        critical=critical,
        profiles=frozenset(profiles) if profiles else frozenset(StartupProfile),
    )


def _shutdown_hook(name, handler=_noop, depends_on=None, timeout=5.0, critical=False):
    return ShutdownHook(
        name=name,
        handler=handler,
        depends_on=depends_on or [],
        timeout=timeout,
        critical=critical,
    )


class MockEventBus:
    def __init__(self):
        self.events = []

    def emit_sync(self, event, data, source=""):
        self.events.append((event, data, source))

    async def emit(self, event, data, source=""):
        self.events.append((event, data, source))


# ---------------------------------------------------------------------------
# LifecyclePhase
# ---------------------------------------------------------------------------

class TestLifecyclePhase:
    def test_all_members(self):
        phases = {p.value for p in LifecyclePhase}
        assert "init" in phases
        assert "starting" in phases
        assert "running" in phases
        assert "draining" in phases
        assert "stopping" in phases
        assert "stopped" in phases
        assert "crashed" in phases

    def test_values(self):
        assert LifecyclePhase.INIT.value == "init"
        assert LifecyclePhase.RUNNING.value == "running"
        assert LifecyclePhase.CRASHED.value == "crashed"


# ---------------------------------------------------------------------------
# StartupProfile
# ---------------------------------------------------------------------------

class TestStartupProfile:
    def test_all_members(self):
        profiles = {p.value for p in StartupProfile}
        assert "full" in profiles
        assert "quick" in profiles
        assert "minimal" in profiles

    def test_from_env_full(self, monkeypatch):
        monkeypatch.setenv("SLO_STARTUP_PROFILE", "full")
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_unknown(self, monkeypatch):
        monkeypatch.setenv("SLO_STARTUP_PROFILE", "bogus")
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("SLO_STARTUP_PROFILE", raising=False)
        assert StartupProfile.from_env() == StartupProfile.FULL


# ---------------------------------------------------------------------------
# _topological_sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_no_deps(self):
        hooks = [_hook("a"), _hook("b"), _hook("c")]
        result = _topological_sort(hooks)
        assert len(result) == 3

    def test_linear_chain(self):
        hooks = [_hook("a"), _hook("b", depends_on=["a"]), _hook("c", depends_on=["b"])]
        result = _topological_sort(hooks)
        names = [h.name for h in result]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_diamond(self):
        hooks = [
            _hook("a"),
            _hook("b", depends_on=["a"]),
            _hook("c", depends_on=["a"]),
            _hook("d", depends_on=["b", "c"]),
        ]
        result = _topological_sort(hooks)
        names = [h.name for h in result]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_cycle_detection(self):
        hooks = [_hook("a", depends_on=["b"]), _hook("b", depends_on=["a"])]
        result = _topological_sort(hooks)
        assert len(result) == 2  # still returns all hooks


# ---------------------------------------------------------------------------
# _dependency_levels
# ---------------------------------------------------------------------------

class TestDependencyLevels:
    def test_no_deps(self):
        hooks = [_hook("a"), _hook("b")]
        levels = _dependency_levels(hooks)
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_linear_chain(self):
        hooks = [_hook("a"), _hook("b", depends_on=["a"]), _hook("c", depends_on=["b"])]
        levels = _dependency_levels(hooks)
        assert len(levels) == 3

    def test_parallel_level(self):
        hooks = [_hook("a"), _hook("b"), _hook("c", depends_on=["a", "b"])]
        levels = _dependency_levels(hooks)
        assert len(levels) == 2
        assert len(levels[0]) == 2


# ---------------------------------------------------------------------------
# LifecycleManager — construction and properties
# ---------------------------------------------------------------------------

class TestLifecycleManagerInit:
    def test_initial_phase(self):
        mgr = LifecycleManager()
        assert mgr.phase == LifecyclePhase.INIT

    def test_not_running(self):
        mgr = LifecycleManager()
        assert mgr.is_running() is False

    def test_not_draining(self):
        mgr = LifecycleManager()
        assert mgr.is_draining() is False

    def test_uptime_zero(self):
        mgr = LifecycleManager()
        assert mgr.uptime_seconds == 0.0

    def test_get_profile(self):
        mgr = LifecycleManager()
        assert mgr.get_profile() == StartupProfile.FULL

    def test_in_flight_zero(self):
        mgr = LifecycleManager()
        assert mgr.in_flight_count == 0


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

class TestHookRegistration:
    def test_register_startup_hook(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db"))
        assert len(mgr._startup_hooks) == 1

    def test_duplicate_startup_hook_skipped(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db"))
        mgr.register_startup_hook(_hook("db"))
        assert len(mgr._startup_hooks) == 1

    def test_register_shutdown_hook(self):
        mgr = LifecycleManager()
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        assert len(mgr._shutdown_hooks) == 1

    def test_duplicate_shutdown_hook_skipped(self):
        mgr = LifecycleManager()
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        assert len(mgr._shutdown_hooks) == 1


# ---------------------------------------------------------------------------
# Health gates
# ---------------------------------------------------------------------------

class TestHealthGates:
    def test_register_and_check(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        assert mgr.gate_ready("db") is True

    def test_gate_not_ready(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: False)
        assert mgr.gate_ready("db") is False

    def test_gates_ready_all_pass(self):
        mgr = LifecycleManager()
        mgr.register_gate("a", lambda: True)
        mgr.register_gate("b", lambda: True)
        assert mgr.gates_ready() is True

    def test_gates_ready_one_fails(self):
        mgr = LifecycleManager()
        mgr.register_gate("a", lambda: True)
        mgr.register_gate("b", lambda: False)
        assert mgr.gates_ready() is False

    def test_unregister_gate(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        mgr.unregister_gate("db")
        assert mgr.gate_ready("db") is True  # no gate = ready

    def test_unregister_nonexistent(self):
        mgr = LifecycleManager()
        mgr.unregister_gate("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_wait_for_gates_immediate(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        result = await mgr.wait_for_gates(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_gates_timeout(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: False)
        result = await mgr.wait_for_gates(timeout=0.1, poll_interval=0.05)
        assert result is False


# ---------------------------------------------------------------------------
# In-flight tracking
# ---------------------------------------------------------------------------

class TestInFlightTracking:
    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        mgr = LifecycleManager()
        ok = await mgr.acquire_in_flight()
        assert ok is True
        assert mgr.in_flight_count == 1
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_acquire_while_draining(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.DRAINING
        ok = await mgr.acquire_in_flight()
        assert ok is False

    @pytest.mark.asyncio
    async def test_release_doesnt_go_negative(self):
        mgr = LifecycleManager()
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0


# ---------------------------------------------------------------------------
# Startup flow
# ---------------------------------------------------------------------------

class TestStartup:
    @pytest.mark.asyncio
    async def test_startup_no_hooks(self):
        mgr = LifecycleManager()
        result = await mgr.start()
        assert result is True
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_startup_single_hook(self):
        mgr = LifecycleManager()
        ran = []
        async def hook_fn():
            ran.append(True)
        mgr.register_startup_hook(_hook("db", handler=hook_fn))
        result = await mgr.start()
        assert result is True
        assert ran == [True]
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_startup_dependency_order(self):
        order = []
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("a", handler=lambda: (order.append("a"), asyncio.sleep(0))))
        mgr.register_startup_hook(_hook("b", handler=lambda: (order.append("b"), asyncio.sleep(0)), depends_on=["a"]))
        mgr.register_startup_hook(_hook("c", handler=lambda: (order.append("c"), asyncio.sleep(0)), depends_on=["b"]))
        await mgr.start()
        assert order.index("a") < order.index("b") < order.index("c")

    @pytest.mark.asyncio
    async def test_startup_critical_failure(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db", handler=_fail, critical=True))
        result = await mgr.start()
        assert result is False
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_startup_non_critical_failure(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("optional", handler=_fail, critical=False))
        result = await mgr.start()
        assert result is True
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_startup_timeout(self):
        mgr = LifecycleManager()
        async def very_slow():
            await asyncio.sleep(100)
        mgr.register_startup_hook(_hook("slow", handler=very_slow, timeout=0.05))
        result = await mgr.start()
        assert result is False
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_startup_parallel_hooks(self):
        mgr = LifecycleManager()
        order = []
        async def task_a():
            await asyncio.sleep(0.05)
            order.append("a")
        async def task_b():
            await asyncio.sleep(0.05)
            order.append("b")
        mgr.register_startup_hook(_hook("a", handler=task_a))
        mgr.register_startup_hook(_hook("b", handler=task_b))
        await mgr.start()
        assert len(order) == 2

    @pytest.mark.asyncio
    async def test_startup_profile_filtering(self):
        mgr = LifecycleManager()
        ran = []
        async def hook_full():
            ran.append("full")
        async def hook_minimal():
            ran.append("minimal")
        mgr.register_startup_hook(_hook("full_hook", handler=hook_full, profiles=[StartupProfile.FULL]))
        mgr.register_startup_hook(_hook("min_hook", handler=hook_minimal, profiles=[StartupProfile.MINIMAL]))
        await mgr.start(profile=StartupProfile.FULL)
        assert "full" in ran
        assert "minimal" not in ran

    @pytest.mark.asyncio
    async def test_startup_results(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db"))
        await mgr.start()
        results = mgr.startup_results
        assert len(results) == 1
        assert results[0]["name"] == "db"
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_startup_ignores_non_init_phase(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.RUNNING
        result = await mgr.start()
        assert result is True  # already running

    @pytest.mark.asyncio
    async def test_startup_event_bus(self):
        bus = MockEventBus()
        mgr = LifecycleManager(event_bus=bus)
        await mgr.start()
        event_types = [e[0] for e in bus.events]
        assert EVT_PHASE_CHANGED in event_types


# ---------------------------------------------------------------------------
# Shutdown flow
# ---------------------------------------------------------------------------

class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_no_hooks(self):
        mgr = LifecycleManager()
        await mgr.start()
        result = await mgr.shutdown()
        assert result is True
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_runs_hooks(self):
        ran = []
        mgr = LifecycleManager()
        mgr.register_shutdown_hook(_shutdown_hook("cleanup", handler=lambda: ran.append(True)))
        await mgr.start()
        await mgr.shutdown()
        assert ran == [True]

    @pytest.mark.asyncio
    async def test_shutdown_hook_failure(self):
        mgr = LifecycleManager()
        mgr.register_shutdown_hook(_shutdown_hook("fail_hook", handler=_fail))
        await mgr.start()
        result = await mgr.shutdown()
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_from_crashed(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db", handler=_fail, critical=True))
        await mgr.start()
        assert mgr.phase == LifecyclePhase.CRASHED
        result = await mgr.shutdown()
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_ignores_non_running(self):
        mgr = LifecycleManager()
        result = await mgr.shutdown()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_results(self):
        mgr = LifecycleManager()
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        await mgr.start()
        await mgr.shutdown()
        results = mgr.shutdown_results
        assert len(results) == 1
        assert results[0]["name"] == "cleanup"

    @pytest.mark.asyncio
    async def test_shutdown_drain(self):
        mgr = LifecycleManager()
        await mgr.acquire_in_flight()
        await mgr.start()
        # Release in-flight after a short delay
        asyncio.get_event_loop().call_later(0.1, lambda: asyncio.ensure_future(mgr.release_in_flight()))
        result = await mgr.shutdown(timeout=2.0)
        assert mgr.phase == LifecyclePhase.STOPPED


# ---------------------------------------------------------------------------
# Mark crashed
# ---------------------------------------------------------------------------

class TestMarkCrashed:
    @pytest.mark.asyncio
    async def test_mark_crashed(self):
        mgr = LifecycleManager()
        await mgr.start()
        await mgr.mark_crashed(reason="test")
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_mark_crashed_from_stopped(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STOPPED
        await mgr.mark_crashed()
        assert mgr.phase == LifecyclePhase.STOPPED  # no-op

    @pytest.mark.asyncio
    async def test_mark_crashed_already_crashed(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.CRASHED
        await mgr.mark_crashed()
        assert mgr.phase == LifecyclePhase.CRASHED  # no-op


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------

class TestGetResults:
    def test_get_results(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db"))
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        result = mgr.get_results()
        assert result["phase"] == "init"
        assert result["profile"] == "full"
        assert result["hooks"]["startup"] == 1
        assert result["hooks"]["shutdown"] == 1
        assert "preview" in result["hooks"]
        assert result["gates"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_results_after_startup(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db"))
        await mgr.start()
        result = mgr.get_results()
        assert result["phase"] == "running"
        assert result["started_at"] > 0


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

class TestPreview:
    def test_preview_default(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("db"))
        preview = mgr.preview()
        assert len(preview) == 1
        assert preview[0]["name"] == "db"

    def test_preview_profile_filter(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(_hook("full_hook", profiles=[StartupProfile.FULL]))
        mgr.register_startup_hook(_hook("min_hook", profiles=[StartupProfile.MINIMAL]))
        preview = mgr.preview(profile=StartupProfile.MINIMAL)
        names = [p["name"] for p in preview]
        assert "min_hook" in names
        assert "full_hook" not in names


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_lifecycle_manager(self):
        reset_lifecycle_manager()
        mgr = get_lifecycle_manager()
        assert isinstance(mgr, LifecycleManager)

    def test_singleton_returns_same(self):
        reset_lifecycle_manager()
        mgr1 = get_lifecycle_manager()
        mgr2 = get_lifecycle_manager()
        assert mgr1 is mgr2

    def test_reset(self):
        reset_lifecycle_manager()
        mgr1 = get_lifecycle_manager()
        reset_lifecycle_manager()
        mgr2 = get_lifecycle_manager()
        assert mgr1 is not mgr2
