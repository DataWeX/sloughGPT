"""Tests for domains.infrastructure.lifecycle — LifecyclePhase, StartupProfile, hooks, topological sort, dependency levels."""

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest

from domains.infrastructure.lifecycle import (
    LifecyclePhase, StartupProfile, StartupHook, ShutdownHook, _HookResult,
    _topological_sort, _dependency_levels, LifecycleManager,
    EVT_PHASE_CHANGED, EVT_HOOK_STARTED, EVT_HOOK_COMPLETED, EVT_HOOK_FAILED,
    ALL_PROFILES, get_lifecycle_manager, reset_lifecycle_manager,
)


class TestLifecyclePhase:
    def test_all_members(self):
        assert len(LifecyclePhase) == 7

    def test_values(self):
        assert LifecyclePhase.INIT.value == "init"
        assert LifecyclePhase.RUNNING.value == "running"
        assert LifecyclePhase.CRASHED.value == "crashed"

    def test_starting_value(self):
        assert LifecyclePhase.STARTING.value == "starting"

    def test_draining_value(self):
        assert LifecyclePhase.DRAINING.value == "draining"

    def test_stopping_value(self):
        assert LifecyclePhase.STOPPING.value == "stopping"

    def test_stopped_value(self):
        assert LifecyclePhase.STOPPED.value == "stopped"

    def test_is_str(self):
        assert isinstance(LifecyclePhase.INIT, str)

    def test_comparison(self):
        assert LifecyclePhase.INIT == LifecyclePhase.INIT
        assert LifecyclePhase.INIT != LifecyclePhase.RUNNING

    def test_in_collection(self):
        assert LifecyclePhase.RUNNING in [LifecyclePhase.INIT, LifecyclePhase.RUNNING]

    def test_string_representation(self):
        assert str(LifecyclePhase.RUNNING) == "LifecyclePhase.RUNNING"

    def test_members_are_strings(self):
        for phase in LifecyclePhase:
            assert isinstance(phase.value, str)

    def test_init_value(self):
        assert LifecyclePhase.INIT.value == "init"

    def test_stopped_value(self):
        assert LifecyclePhase.STOPPED.value == "stopped"

    def test_crashed_value(self):
        assert LifecyclePhase.CRASHED.value == "crashed"

    def test_member_names(self):
        names = {p.name for p in LifecyclePhase}
        expected = {"INIT", "STARTING", "RUNNING", "DRAINING", "STOPPING", "STOPPED", "CRASHED"}
        assert names == expected

    def test_iteration(self):
        phases = list(LifecyclePhase)
        assert len(phases) == 7

    def test_membership(self):
        assert LifecyclePhase.INIT in LifecyclePhase
        assert "init" in [p.value for p in LifecyclePhase]

    def test_from_value(self):
        assert LifecyclePhase("init") == LifecyclePhase.INIT
        assert LifecyclePhase("running") == LifecyclePhase.RUNNING

    def test_from_value_invalid(self):
        with pytest.raises(ValueError):
            LifecyclePhase("invalid_phase")

    def test_hash(self):
        s = {LifecyclePhase.INIT, LifecyclePhase.RUNNING, LifecyclePhase.INIT}
        assert len(s) == 2


class TestStartupProfile:
    def test_all_members(self):
        assert len(StartupProfile) == 3

    def test_from_env_full(self):
        os.environ["SLO_STARTUP_PROFILE"] = "full"
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_unknown(self):
        os.environ["SLO_STARTUP_PROFILE"] = "bogus"
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_quick(self):
        os.environ["SLO_STARTUP_PROFILE"] = "quick"
        assert StartupProfile.from_env() == StartupProfile.QUICK

    def test_from_env_minimal(self):
        os.environ["SLO_STARTUP_PROFILE"] = "minimal"
        assert StartupProfile.from_env() == StartupProfile.MINIMAL

    def test_from_env_empty_string(self):
        os.environ["SLO_STARTUP_PROFILE"] = ""
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_with_whitespace(self):
        os.environ["SLO_STARTUP_PROFILE"] = "  full  "
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_case_insensitive(self):
        os.environ["SLO_STARTUP_PROFILE"] = "FULL"
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_values(self):
        assert StartupProfile.FULL.value == "full"
        assert StartupProfile.QUICK.value == "quick"
        assert StartupProfile.MINIMAL.value == "minimal"

    def test_all_profiles_frozenset(self):
        assert isinstance(ALL_PROFILES, frozenset)
        assert len(ALL_PROFILES) == 3

    def test_from_env_not_set(self):
        os.environ.pop("SLO_STARTUP_PROFILE", None)
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_partial_match_not_found(self):
        os.environ["SLO_STARTUP_PROFILE"] = "fully"
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_member_names(self):
        names = {p.name for p in StartupProfile}
        assert names == {"FULL", "QUICK", "MINIMAL"}

    def test_profiles_are_subsets(self):
        assert {StartupProfile.FULL} < ALL_PROFILES
        assert {StartupProfile.QUICK, StartupProfile.MINIMAL} < ALL_PROFILES

    def test_from_env_uppercase_quick(self):
        os.environ["SLO_STARTUP_PROFILE"] = "QUICK"
        assert StartupProfile.from_env() == StartupProfile.QUICK

    def test_from_env_uppercase_minimal(self):
        os.environ["SLO_STARTUP_PROFILE"] = "MINIMAL"
        assert StartupProfile.from_env() == StartupProfile.MINIMAL


class TestStartupHook:
    def test_defaults(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop)
        assert h.depends_on == []
        assert h.critical is True
        assert h.timeout == 30.0

    def test_custom(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, depends_on=["b"], critical=False)
        assert h.depends_on == ["b"]
        assert h.critical is False

    def test_custom_timeout(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, timeout=60.0)
        assert h.timeout == 60.0

    def test_profiles_default(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop)
        assert h.profiles == ALL_PROFILES

    def test_profiles_custom(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, profiles=frozenset({StartupProfile.FULL}))
        assert StartupProfile.FULL in h.profiles
        assert StartupProfile.QUICK not in h.profiles

    def test_multiple_depends_on(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, depends_on=["b", "c", "d"])
        assert len(h.depends_on) == 3

    def test_handler_is_callable(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop)
        assert callable(h.handler)

    def test_profiles_multiple(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop,
                        profiles=frozenset({StartupProfile.FULL, StartupProfile.QUICK}))
        assert StartupProfile.FULL in h.profiles
        assert StartupProfile.QUICK in h.profiles
        assert StartupProfile.MINIMAL not in h.profiles

    def test_timeout_zero(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, timeout=0.0)
        assert h.timeout == 0.0

    def test_timeout_very_large(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, timeout=3600.0)
        assert h.timeout == 3600.0

    def test_depends_on_empty(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, depends_on=[])
        assert h.depends_on == []

    def test_name_preserved(self):
        async def noop(): pass
        h = StartupHook(name="my_hook_123", handler=noop)
        assert h.name == "my_hook_123"

    def test_profiles_frozen_set(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop)
        assert isinstance(h.profiles, frozenset)

    def test_depends_on_returns_list(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, depends_on=["x"])
        assert isinstance(h.depends_on, list)


class TestShutdownHook:
    def test_defaults(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop)
        assert h.critical is False

    def test_custom_critical(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop, critical=True)
        assert h.critical is True

    def test_depends_on(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop, depends_on=["b"])
        assert h.depends_on == ["b"]

    def test_timeout(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop, timeout=10.0)
        assert h.timeout == 10.0

    def test_name_preserved(self):
        async def noop(): pass
        h = ShutdownHook(name="shutdown_db", handler=noop)
        assert h.name == "shutdown_db"

    def test_handler_callable(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop)
        assert callable(h.handler)

    def test_depends_on_empty(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop, depends_on=[])
        assert h.depends_on == []

    def test_timeout_zero(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop, timeout=0.0)
        assert h.timeout == 0.0

    def test_critical_default_false(self):
        async def noop(): pass
        h1 = ShutdownHook(name="a", handler=noop)
        h2 = ShutdownHook(name="b", handler=noop)
        assert h1.critical is False
        assert h2.critical is False

    def test_multiple_depends_on(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop, depends_on=["b", "c"])
        assert len(h.depends_on) == 2


class TestHookResult:
    def test_fields(self):
        r = _HookResult(name="x", success=True, elapsed=1.5)
        assert r.error == ""

    def test_failure(self):
        r = _HookResult(name="x", success=False, elapsed=0.1, error="boom")
        assert r.success is False
        assert r.error == "boom"

    def test_zero_elapsed(self):
        r = _HookResult(name="x", success=True, elapsed=0.0)
        assert r.elapsed == 0.0

    def test_name(self):
        r = _HookResult(name="my_hook", success=True, elapsed=0.1)
        assert r.name == "my_hook"

    def test_negative_elapsed(self):
        r = _HookResult(name="x", success=True, elapsed=-0.5)
        assert r.elapsed == -0.5

    def test_large_elapsed(self):
        r = _HookResult(name="x", success=True, elapsed=9999.9)
        assert r.elapsed == 9999.9

    def test_success_true(self):
        r = _HookResult(name="x", success=True, elapsed=1.0)
        assert r.success is True

    def test_error_empty_string(self):
        r = _HookResult(name="x", success=True, elapsed=1.0, error="")
        assert r.error == ""

    def test_error_long_message(self):
        long_error = "x" * 5000
        r = _HookResult(name="x", success=False, elapsed=0.1, error=long_error)
        assert len(r.error) == 5000

    def test_dataclass_fields(self):
        fields = {f.name for f in __import__("dataclasses").fields(_HookResult)}
        assert fields == {"name", "success", "elapsed", "error"}


class TestTopologicalSort:
    def test_no_deps(self):
        async def noop(): pass
        hooks = [StartupHook(name="b", handler=noop), StartupHook(name="a", handler=noop)]
        ordered = _topological_sort(hooks)
        assert [h.name for h in ordered] == ["b", "a"]

    def test_with_deps(self):
        async def noop(): pass
        h1 = StartupHook(name="db", handler=noop, depends_on=[])
        h2 = StartupHook(name="api", handler=noop, depends_on=["db"])
        ordered = _topological_sort([h2, h1])
        assert [h.name for h in ordered] == ["db", "api"]

    def test_cycle_detected(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop, depends_on=["b"])
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        ordered = _topological_sort([h1, h2])
        assert len(ordered) == 2

    def test_empty_list(self):
        ordered = _topological_sort([])
        assert ordered == []

    def test_single_hook(self):
        async def noop(): pass
        h = StartupHook(name="only", handler=noop)
        ordered = _topological_sort([h])
        assert len(ordered) == 1
        assert ordered[0].name == "only"

    def test_diamond_dependency(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["a"])
        h4 = StartupHook(name="d", handler=noop, depends_on=["b", "c"])
        ordered = _topological_sort([h4, h3, h2, h1])
        names = [h.name for h in ordered]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_three_level_chain(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["b"])
        ordered = _topological_sort([h3, h2, h1])
        assert [h.name for h in ordered] == ["a", "b", "c"]

    def test_no_deps_many_hooks(self):
        async def noop(): pass
        hooks = [StartupHook(name=f"h{i}", handler=noop) for i in range(10)]
        ordered = _topological_sort(hooks)
        assert len(ordered) == 10

    def test_preserves_handler(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop)
        ordered = _topological_sort([h])
        assert ordered[0].handler is noop

    def test_depends_on_nonexistent(self):
        async def noop(): pass
        h = StartupHook(name="a", handler=noop, depends_on=["nonexistent"])
        ordered = _topological_sort([h])
        assert len(ordered) == 1

    def test_complex_graph(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["a"])
        h4 = StartupHook(name="d", handler=noop, depends_on=["b", "c"])
        h5 = StartupHook(name="e", handler=noop, depends_on=["d"])
        ordered = _topological_sort([h5, h4, h3, h2, h1])
        names = [h.name for h in ordered]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")
        assert names.index("d") < names.index("e")

    def test_three_way_cycle(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop, depends_on=["c"])
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["b"])
        ordered = _topological_sort([h1, h2, h3])
        assert len(ordered) == 3


class TestDependencyLevels:
    def test_no_deps(self):
        async def noop(): pass
        hooks = [StartupHook(name="a", handler=noop), StartupHook(name="b", handler=noop)]
        levels = _dependency_levels(hooks)
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_linear_chain(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["b"])
        levels = _dependency_levels([h3, h2, h1])
        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert levels[1][0].name == "b"
        assert levels[2][0].name == "c"

    def test_empty_list(self):
        levels = _dependency_levels([])
        assert levels == []

    def test_single_hook(self):
        async def noop(): pass
        levels = _dependency_levels([StartupHook(name="a", handler=noop)])
        assert len(levels) == 1

    def test_parallel_hooks(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop)
        h3 = StartupHook(name="c", handler=noop)
        levels = _dependency_levels([h1, h2, h3])
        assert len(levels) == 1
        assert len(levels[0]) == 3

    def test_diamond(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["a"])
        h4 = StartupHook(name="d", handler=noop, depends_on=["b", "c"])
        levels = _dependency_levels([h4, h3, h2, h1])
        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert len(levels[1]) == 2
        assert levels[2][0].name == "d"

    def test_level_0_has_no_deps(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        levels = _dependency_levels([h2, h1])
        for h in levels[0]:
            assert h.depends_on == []

    def test_levels_are_sequential(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["b"])
        h4 = StartupHook(name="d", handler=noop, depends_on=["c"])
        levels = _dependency_levels([h4, h3, h2, h1])
        assert len(levels) == 4

    def test_all_hooks_in_some_level(self):
        async def noop(): pass
        h1 = StartupHook(name="a", handler=noop)
        h2 = StartupHook(name="b", handler=noop, depends_on=["a"])
        h3 = StartupHook(name="c", handler=noop, depends_on=["b"])
        levels = _dependency_levels([h3, h2, h1])
        all_names = [h.name for level in levels for h in level]
        assert set(all_names) == {"a", "b", "c"}


class TestEventConstants:
    def test_phase_changed(self):
        assert EVT_PHASE_CHANGED == "lifecycle.phase_changed"

    def test_hook_started(self):
        assert EVT_HOOK_STARTED == "lifecycle.hook_started"

    def test_hook_completed(self):
        assert EVT_HOOK_COMPLETED == "lifecycle.hook_completed"

    def test_hook_failed(self):
        assert EVT_HOOK_FAILED == "lifecycle.hook_failed"

    def test_all_strings(self):
        assert isinstance(EVT_PHASE_CHANGED, str)
        assert isinstance(EVT_HOOK_STARTED, str)
        assert isinstance(EVT_HOOK_COMPLETED, str)
        assert isinstance(EVT_HOOK_FAILED, str)

    def test_all_start_with_lifecycle(self):
        for evt in (EVT_PHASE_CHANGED, EVT_HOOK_STARTED, EVT_HOOK_COMPLETED, EVT_HOOK_FAILED):
            assert evt.startswith("lifecycle.")


class TestLifecycleManagerProperties:
    def test_initial_phase(self):
        mgr = LifecycleManager()
        assert mgr.phase == LifecyclePhase.INIT

    def test_is_running_false_initially(self):
        mgr = LifecycleManager()
        assert mgr.is_running() is False

    def test_is_draining_false_initially(self):
        mgr = LifecycleManager()
        assert mgr.is_draining() is False

    def test_uptime_zero_initially(self):
        mgr = LifecycleManager()
        assert mgr.uptime_seconds == 0.0

    def test_started_at_zero_initially(self):
        mgr = LifecycleManager()
        assert mgr.started_at == 0.0

    def test_get_profile_default(self):
        mgr = LifecycleManager()
        assert mgr.get_profile() == StartupProfile.FULL

    def test_in_flight_count_zero(self):
        mgr = LifecycleManager()
        assert mgr.in_flight_count == 0

    def test_startup_results_empty(self):
        mgr = LifecycleManager()
        assert mgr.startup_results == []

    def test_shutdown_results_empty(self):
        mgr = LifecycleManager()
        assert mgr.shutdown_results == []


class TestLifecycleManagerHookRegistration:
    def test_register_startup_hook(self):
        mgr = LifecycleManager()
        async def noop(): pass
        hook = StartupHook(name="test", handler=noop)
        mgr.register_startup_hook(hook)
        assert len(mgr._startup_hooks) == 1

    def test_register_shutdown_hook(self):
        mgr = LifecycleManager()
        async def noop(): pass
        hook = ShutdownHook(name="test", handler=noop)
        mgr.register_shutdown_hook(hook)
        assert len(mgr._shutdown_hooks) == 1

    def test_duplicate_startup_hook_skipped(self):
        mgr = LifecycleManager()
        async def noop(): pass
        hook1 = StartupHook(name="test", handler=noop)
        hook2 = StartupHook(name="test", handler=noop)
        mgr.register_startup_hook(hook1)
        mgr.register_startup_hook(hook2)
        assert len(mgr._startup_hooks) == 1

    def test_duplicate_shutdown_hook_skipped(self):
        mgr = LifecycleManager()
        async def noop(): pass
        hook1 = ShutdownHook(name="test", handler=noop)
        hook2 = ShutdownHook(name="test", handler=noop)
        mgr.register_shutdown_hook(hook1)
        mgr.register_shutdown_hook(hook2)
        assert len(mgr._shutdown_hooks) == 1

    def test_multiple_startup_hooks(self):
        mgr = LifecycleManager()
        async def noop(): pass
        for i in range(5):
            mgr.register_startup_hook(StartupHook(name=f"h{i}", handler=noop))
        assert len(mgr._startup_hooks) == 5

    def test_multiple_shutdown_hooks(self):
        mgr = LifecycleManager()
        async def noop(): pass
        for i in range(3):
            mgr.register_shutdown_hook(ShutdownHook(name=f"h{i}", handler=noop))
        assert len(mgr._shutdown_hooks) == 3


class TestLifecycleManagerGates:
    def test_register_gate(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        assert "db" in mgr._health_gates

    def test_unregister_gate(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        mgr.unregister_gate("db")
        assert "db" not in mgr._health_gates

    def test_gate_ready(self):
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

    def test_gates_not_ready_one_fails(self):
        mgr = LifecycleManager()
        mgr.register_gate("a", lambda: True)
        mgr.register_gate("b", lambda: False)
        assert mgr.gates_ready() is False

    def test_gate_ready_nonexistent(self):
        mgr = LifecycleManager()
        assert mgr.gate_ready("nonexistent") is True

    def test_unregister_nonexistent_gate(self):
        mgr = LifecycleManager()
        mgr.unregister_gate("nonexistent")

    def test_multiple_gates(self):
        mgr = LifecycleManager()
        for i in range(10):
            mgr.register_gate(f"gate{i}", lambda: True)
        assert mgr.gates_ready() is True


class TestLifecycleManagerPreview:
    def test_preview_empty(self):
        mgr = LifecycleManager()
        result = mgr.preview()
        assert result == []

    def test_preview_with_hooks(self):
        mgr = LifecycleManager()
        async def noop(): pass
        mgr.register_startup_hook(StartupHook(name="a", handler=noop))
        mgr.register_startup_hook(StartupHook(name="b", handler=noop, depends_on=["a"]))
        result = mgr.preview()
        assert len(result) == 2
        names = [h["name"] for h in result]
        assert names.index("a") < names.index("b")

    def test_preview_profile_filter(self):
        mgr = LifecycleManager()
        async def noop(): pass
        mgr.register_startup_hook(StartupHook(
            name="full_only", handler=noop,
            profiles=frozenset({StartupProfile.FULL})))
        mgr.register_startup_hook(StartupHook(
            name="quick_only", handler=noop,
            profiles=frozenset({StartupProfile.QUICK})))
        result = mgr.preview(profile=StartupProfile.FULL)
        names = [h["name"] for h in result]
        assert "full_only" in names
        assert "quick_only" not in names


class TestLifecycleManagerGetResults:
    def test_get_results_initial(self):
        mgr = LifecycleManager()
        r = mgr.get_results()
        assert r["phase"] == "init"
        assert r["in_flight"] == 0
        assert r["hooks"]["startup"] == 0
        assert r["hooks"]["shutdown"] == 0
        assert r["gates"]["total"] == 0

    def test_get_results_with_hooks(self):
        mgr = LifecycleManager()
        async def noop(): pass
        mgr.register_startup_hook(StartupHook(name="a", handler=noop))
        mgr.register_shutdown_hook(ShutdownHook(name="b", handler=noop))
        r = mgr.get_results()
        assert r["hooks"]["startup"] == 1
        assert r["hooks"]["shutdown"] == 1


class TestLifecycleManagerReset:
    def test_reset_creates_new(self):
        reset_lifecycle_manager()
        a = get_lifecycle_manager()
        reset_lifecycle_manager()
        b = get_lifecycle_manager()
        assert a is not b

    def test_singleton(self):
        reset_lifecycle_manager()
        a = get_lifecycle_manager()
        b = get_lifecycle_manager()
        assert a is b
        reset_lifecycle_manager()

    def test_mark_crashed(self):
        mgr = LifecycleManager()
        asyncio.get_event_loop().run_until_complete(mgr.mark_crashed("test"))
        assert mgr.phase == LifecyclePhase.CRASHED

    def test_mark_crashed_when_stopped(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STOPPED
        asyncio.get_event_loop().run_until_complete(mgr.mark_crashed("test"))
        assert mgr.phase == LifecyclePhase.STOPPED
