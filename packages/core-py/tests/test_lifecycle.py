"""
Tests for domains.infrastructure.lifecycle — LifecycleManager, phases, hooks, gates, drain.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from domains.infrastructure.lifecycle import (
    ALL_PROFILES,
    EVT_HOOK_COMPLETED,
    EVT_HOOK_FAILED,
    EVT_HOOK_STARTED,
    EVT_PHASE_CHANGED,
    LifecycleManager,
    LifecyclePhase,
    ShutdownHook,
    StartupHook,
    StartupProfile,
    _dependency_levels,
    _topological_sort,
)


# ── Fixtures ──


@pytest.fixture
def mgr():
    return LifecycleManager()


@pytest.fixture
def mgr_with_bus():
    bus = AsyncMock()
    return LifecycleManager(event_bus=bus), bus


def _hook(name, depends_on=None, handler=None, timeout=5.0, critical=True, profiles=None):
    """Helper to create a StartupHook."""
    if handler is None:
        async def _ok():
            pass
        handler = _ok
    return StartupHook(
        name=name,
        handler=handler,
        depends_on=depends_on or [],
        timeout=timeout,
        critical=critical,
        profiles=profiles or ALL_PROFILES,
    )


def _shutdown_hook(name, handler=None, depends_on=None, timeout=5.0, critical=False):
    """Helper to create a ShutdownHook."""
    if handler is None:
        async def _ok():
            pass
        handler = _ok
    return ShutdownHook(name=name, handler=handler, depends_on=depends_on or [], timeout=timeout, critical=critical)


# ── Phase enum tests ──


class TestLifecyclePhase:
    def test_all_phases(self):
        phases = list(LifecyclePhase)
        assert len(phases) == 7
        assert LifecyclePhase.INIT.value == "init"
        assert LifecyclePhase.RUNNING.value == "running"
        assert LifecyclePhase.STOPPED.value == "stopped"

    def test_str_enum(self):
        assert LifecyclePhase.INIT.value == "init"


# ── StartupProfile tests ──


class TestStartupProfile:
    def test_all_profiles(self):
        assert StartupProfile.FULL.value == "full"
        assert StartupProfile.QUICK.value == "quick"
        assert StartupProfile.MINIMAL.value == "minimal"

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("SLO_STARTUP_PROFILE", raising=False)
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_quick(self, monkeypatch):
        monkeypatch.setenv("SLO_STARTUP_PROFILE", "quick")
        assert StartupProfile.from_env() == StartupProfile.QUICK

    def test_from_env_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("SLO_STARTUP_PROFILE", "invalid")
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_all_profiles_frozen(self):
        assert isinstance(ALL_PROFILES, frozenset)
        assert len(ALL_PROFILES) == 3


# ── Topological sort tests ──


class TestTopologicalSort:
    def test_no_hooks(self):
        assert _topological_sort([]) == []

    def test_single_hook(self):
        h = _hook("a")
        result = _topological_sort([h])
        assert len(result) == 1
        assert result[0].name == "a"

    def test_linear_chain(self):
        a = _hook("a")
        b = _hook("b", depends_on=["a"])
        c = _hook("c", depends_on=["b"])
        result = _topological_sort([c, a, b])
        names = [h.name for h in result]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_independent_hooks(self):
        a = _hook("a")
        b = _hook("b")
        result = _topological_sort([a, b])
        assert len(result) == 2

    def test_diamond_dependency(self):
        a = _hook("a")
        b = _hook("b", depends_on=["a"])
        c = _hook("c", depends_on=["a"])
        d = _hook("d", depends_on=["b", "c"])
        result = _topological_sort([d, b, a, c])
        names = [h.name for h in result]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_cycle_detection(self):
        a = _hook("a", depends_on=["c"])
        b = _hook("b", depends_on=["a"])
        c = _hook("c", depends_on=["b"])
        result = _topological_sort([a, b, c])
        assert len(result) == 3


# ── Dependency levels tests ──


class TestDependencyLevels:
    def test_empty(self):
        assert _dependency_levels([]) == []

    def test_independent(self):
        a = _hook("a")
        b = _hook("b")
        levels = _dependency_levels([a, b])
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_linear_chain(self):
        a = _hook("a")
        b = _hook("b", depends_on=["a"])
        c = _hook("c", depends_on=["b"])
        levels = _dependency_levels([a, b, c])
        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert levels[1][0].name == "b"
        assert levels[2][0].name == "c"

    def test_parallel_level(self):
        a = _hook("a")
        b = _hook("b")
        c = _hook("c", depends_on=["a", "b"])
        levels = _dependency_levels([a, b, c])
        assert len(levels) == 2
        assert len(levels[0]) == 2
        assert levels[1][0].name == "c"


# ── LifecycleManager init tests ──


class TestLifecycleManagerInit:
    def test_initial_phase(self, mgr):
        assert mgr.phase == LifecyclePhase.INIT

    def test_not_running(self, mgr):
        assert not mgr.is_running()

    def test_not_draining(self, mgr):
        assert not mgr.is_draining()

    def test_uptime_zero(self, mgr):
        assert mgr.uptime_seconds == 0.0

    def test_in_flight_zero(self, mgr):
        assert mgr.in_flight_count == 0

    def test_profile_default(self, mgr):
        assert mgr.get_profile() == StartupProfile.FULL

    def test_empty_startup_results(self, mgr):
        assert mgr.startup_results == []

    def test_empty_shutdown_results(self, mgr):
        assert mgr.shutdown_results == []


# ── Hook registration tests ──


class TestHookRegistration:
    def test_register_startup_hook(self, mgr):
        h = _hook("db")
        mgr.register_startup_hook(h)
        assert len(mgr._startup_hooks) == 1
        assert mgr._startup_hooks[0].name == "db"

    def test_register_shutdown_hook(self, mgr):
        h = _shutdown_hook("cleanup")
        mgr.register_shutdown_hook(h)
        assert len(mgr._shutdown_hooks) == 1

    def test_duplicate_startup_hook_skipped(self, mgr):
        mgr.register_startup_hook(_hook("db"))
        mgr.register_startup_hook(_hook("db"))
        assert len(mgr._startup_hooks) == 1

    def test_duplicate_shutdown_hook_skipped(self, mgr):
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        mgr.register_shutdown_hook(_shutdown_hook("cleanup"))
        assert len(mgr._shutdown_hooks) == 1

    def test_multiple_different_hooks(self, mgr):
        mgr.register_startup_hook(_hook("db"))
        mgr.register_startup_hook(_hook("model"))
        mgr.register_startup_hook(_hook("cache"))
        assert len(mgr._startup_hooks) == 3


# ── Health gate tests ──


class TestHealthGates:
    def test_register_gate(self, mgr):
        mgr.register_gate("db", lambda: True)
        assert mgr.gate_ready("db")

    def test_gate_not_ready(self, mgr):
        mgr.register_gate("db", lambda: False)
        assert not mgr.gate_ready("db")

    def test_all_gates_ready(self, mgr):
        mgr.register_gate("db", lambda: True)
        mgr.register_gate("cache", lambda: True)
        assert mgr.gates_ready()

    def test_one_gate_not_ready(self, mgr):
        mgr.register_gate("db", lambda: True)
        mgr.register_gate("cache", lambda: False)
        assert not mgr.gates_ready()

    def test_unregister_gate(self, mgr):
        mgr.register_gate("db", lambda: True)
        mgr.unregister_gate("db")
        assert mgr.gate_ready("db")

    def test_unregister_nonexistent_gate(self, mgr):
        mgr.unregister_gate("nonexistent")

    def test_unknown_gate_returns_ready(self, mgr):
        assert mgr.gate_ready("unknown")


# ── In-flight task tracking tests ──


class TestInFlightTracking:
    @pytest.mark.asyncio
    async def test_acquire_in_flight(self, mgr):
        result = await mgr.acquire_in_flight()
        assert result is True
        assert mgr.in_flight_count == 1

    @pytest.mark.asyncio
    async def test_release_in_flight(self, mgr):
        await mgr.acquire_in_flight()
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_multiple_in_flight(self, mgr):
        await mgr.acquire_in_flight()
        await mgr.acquire_in_flight()
        assert mgr.in_flight_count == 2
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 1

    @pytest.mark.asyncio
    async def test_release_below_zero(self, mgr):
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_acquire_fails_when_draining(self, mgr):
        await mgr.start()
        await mgr.acquire_in_flight()
        # Shutdown starts with DRAINING phase, so acquire should fail
        async def slow_drain():
            await asyncio.sleep(10)
        mgr.register_shutdown_hook(_shutdown_hook("slow", handler=slow_drain))
        # Start shutdown in background
        asyncio.create_task(mgr.shutdown(timeout=30))
        await asyncio.sleep(0.05)  # Let it enter DRAINING
        result = await mgr.acquire_in_flight()
        assert result is False


# ── Start tests ──


class TestStart:
    @pytest.mark.asyncio
    async def test_start_with_no_hooks(self, mgr):
        result = await mgr.start()
        assert result is True
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_start_transitions_to_running(self, mgr):
        await mgr.start()
        assert mgr.phase == LifecyclePhase.RUNNING
        assert mgr.started_at > 0

    @pytest.mark.asyncio
    async def test_start_sets_uptime(self, mgr):
        await mgr.start()
        assert mgr.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_start_with_successful_hook(self, mgr):
        called = []

        async def hook():
            called.append(True)

        mgr.register_startup_hook(_hook("test", handler=hook))
        result = await mgr.start()
        assert result is True
        assert called == [True]

    @pytest.mark.asyncio
    async def test_start_with_failing_critical_hook(self, mgr):
        async def hook():
            raise RuntimeError("boom")

        mgr.register_startup_hook(_hook("test", handler=hook, critical=True))
        result = await mgr.start()
        assert result is False
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_start_with_failing_non_critical_hook(self, mgr):
        async def hook():
            raise RuntimeError("boom")

        mgr.register_startup_hook(_hook("test", handler=hook, critical=False))
        result = await mgr.start()
        assert result is True
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_start_ignores_non_init_phase(self, mgr):
        await mgr.start()
        result = await mgr.start()
        assert result is True

    @pytest.mark.asyncio
    async def test_startup_results_recorded(self, mgr):
        await mgr.start()
        assert len(mgr.startup_results) == 0

    @pytest.mark.asyncio
    async def test_startup_results_with_hooks(self, mgr):
        async def hook():
            pass

        mgr.register_startup_hook(_hook("db", handler=hook))
        await mgr.start()
        assert len(mgr.startup_results) == 1
        assert mgr.startup_results[0]["name"] == "db"
        assert mgr.startup_results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_hook_timeout(self, mgr):
        async def hook():
            await asyncio.sleep(10)

        mgr.register_startup_hook(_hook("slow", handler=hook, timeout=0.1))
        result = await mgr.start()
        assert result is False
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_dependency_order(self, mgr):
        order = []

        async def a_hook():
            order.append("a")

        async def b_hook():
            order.append("b")

        mgr.register_startup_hook(_hook("a", handler=a_hook))
        mgr.register_startup_hook(_hook("b", handler=b_hook, depends_on=["a"]))
        await mgr.start()
        assert order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_parallel_hooks(self, mgr):
        order = []

        async def a_hook():
            order.append("a")

        async def b_hook():
            order.append("b")

        mgr.register_startup_hook(_hook("a", handler=a_hook))
        mgr.register_startup_hook(_hook("b", handler=b_hook))
        await mgr.start()
        assert set(order) == {"a", "b"}


# ── Shutdown tests ──


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_from_running(self, mgr):
        await mgr.start()
        result = await mgr.shutdown()
        assert result is True
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_ignores_non_running(self, mgr):
        result = await mgr.shutdown()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_runs_hooks(self, mgr):
        called = []

        async def hook():
            called.append(True)

        mgr.register_shutdown_hook(_shutdown_hook("cleanup", handler=hook))
        await mgr.start()
        await mgr.shutdown()
        assert called == [True]

    @pytest.mark.asyncio
    async def test_shutdown_results_recorded(self, mgr):
        async def hook():
            pass

        mgr.register_shutdown_hook(_shutdown_hook("cleanup", handler=hook))
        await mgr.start()
        await mgr.shutdown()
        assert len(mgr.shutdown_results) == 1
        assert mgr.shutdown_results[0]["name"] == "cleanup"


# ── Preview tests ──


class TestPreview:
    def test_preview_empty(self, mgr):
        assert mgr.preview() == []

    def test_preview_with_hooks(self, mgr):
        mgr.register_startup_hook(_hook("db"))
        mgr.register_startup_hook(_hook("model", depends_on=["db"]))
        result = mgr.preview()
        assert len(result) == 2
        assert result[0]["name"] == "db"
        assert result[1]["name"] == "model"


# ── EventBus integration tests ──


class TestEventBus:
    @pytest.mark.asyncio
    async def test_phase_change_emits_event(self, mgr_with_bus):
        mgr, bus = mgr_with_bus
        await mgr.start()
        bus.emit.assert_called()

    @pytest.mark.asyncio
    async def test_event_names(self, mgr_with_bus):
        mgr, bus = mgr_with_bus
        await mgr.start()
        emitted_events = [call.args[0] for call in bus.emit.call_args_list]
        assert EVT_PHASE_CHANGED in emitted_events
