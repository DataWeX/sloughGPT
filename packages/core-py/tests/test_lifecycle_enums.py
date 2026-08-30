"""Tests for domains.infrastructure.lifecycle — LifecyclePhase, StartupProfile, StartupHook, ShutdownHook, _HookResult, _topological_sort, _dependency_levels, LifecycleManager, ALL_PROFILES."""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from domains.infrastructure.lifecycle import (
    LifecyclePhase,
    StartupProfile,
    StartupHook,
    ShutdownHook,
    _HookResult,
    _topological_sort,
    _dependency_levels,
    LifecycleManager,
    ALL_PROFILES,
    EVT_PHASE_CHANGED,
    EVT_HOOK_STARTED,
    EVT_HOOK_COMPLETED,
    EVT_HOOK_FAILED,
    get_lifecycle_manager,
    reset_lifecycle_manager,
    _lifecycle_manager,
)


# ---------------------------------------------------------------------------
# LifecyclePhase
# ---------------------------------------------------------------------------

class TestLifecyclePhase:
    def test_all_members(self):
        assert len(LifecyclePhase) == 7

    def test_values(self):
        assert LifecyclePhase.INIT.value == "init"
        assert LifecyclePhase.STARTING.value == "starting"
        assert LifecyclePhase.RUNNING.value == "running"
        assert LifecyclePhase.DRAINING.value == "draining"
        assert LifecyclePhase.STOPPING.value == "stopping"
        assert LifecyclePhase.STOPPED.value == "stopped"
        assert LifecyclePhase.CRASHED.value == "crashed"

    def test_str_enum_comparison(self):
        assert LifecyclePhase.RUNNING == "running"
        assert LifecyclePhase.INIT == "init"
        assert LifecyclePhase.CRASHED == "crashed"

    def test_str_enum_hash(self):
        s = {LifecyclePhase.RUNNING, LifecyclePhase.RUNNING}
        assert len(s) == 1

    def test_iteration(self):
        names = [p.name for p in LifecyclePhase]
        assert "INIT" in names
        assert "STOPPED" in names

    def test_member_by_name(self):
        assert LifecyclePhase["RUNNING"] is LifecyclePhase.RUNNING

    def test_member_by_value(self):
        assert LifecyclePhase("init") is LifecyclePhase.INIT

    def test_not_equal_to_other_type(self):
        assert LifecyclePhase.RUNNING != 42

    def test_ordering_not_defined(self):
        assert LifecyclePhase.RUNNING > LifecyclePhase.INIT

    def test_str_returns_repr_in_312(self):
        r = str(LifecyclePhase.INIT)
        assert "init" in r.lower()

    def test_all_values_are_unique(self):
        values = [p.value for p in LifecyclePhase]
        assert len(values) == len(set(values))

    def test_all_names_are_unique(self):
        names = [p.name for p in LifecyclePhase]
        assert len(names) == len(set(names))

    def test_is_str_subclass(self):
        assert issubclass(LifecyclePhase, str)

    def test_repr_contains_class_name(self):
        r = repr(LifecyclePhase.RUNNING)
        assert "LifecyclePhase" in r or "running" in r

    def test_membership_test(self):
        assert LifecyclePhase.RUNNING in list(LifecyclePhase)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            LifecyclePhase("nonexistent")


# ---------------------------------------------------------------------------
# StartupProfile
# ---------------------------------------------------------------------------

class TestStartupProfile:
    def test_all_members(self):
        assert len(StartupProfile) == 3

    def test_values(self):
        assert StartupProfile.FULL.value == "full"
        assert StartupProfile.QUICK.value == "quick"
        assert StartupProfile.MINIMAL.value == "minimal"

    def test_from_env_full(self):
        os.environ["SLO_STARTUP_PROFILE"] = "full"
        try:
            assert StartupProfile.from_env() == StartupProfile.FULL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_quick(self):
        os.environ["SLO_STARTUP_PROFILE"] = "quick"
        try:
            assert StartupProfile.from_env() == StartupProfile.QUICK
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_minimal(self):
        os.environ["SLO_STARTUP_PROFILE"] = "minimal"
        try:
            assert StartupProfile.from_env() == StartupProfile.MINIMAL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_unknown_fallback(self):
        os.environ["SLO_STARTUP_PROFILE"] = "garbage"
        try:
            assert StartupProfile.from_env() == StartupProfile.FULL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_default_when_unset(self):
        os.environ.pop("SLO_STARTUP_PROFILE", None)
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_case_insensitive(self):
        os.environ["SLO_STARTUP_PROFILE"] = "MINIMAL"
        try:
            assert StartupProfile.from_env() == StartupProfile.MINIMAL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_strips_whitespace(self):
        os.environ["SLO_STARTUP_PROFILE"] = "  quick  "
        try:
            assert StartupProfile.from_env() == StartupProfile.QUICK
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_empty_string_fallback(self):
        os.environ["SLO_STARTUP_PROFILE"] = ""
        try:
            assert StartupProfile.from_env() == StartupProfile.FULL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_from_env_numeric_string_fallback(self):
        os.environ["SLO_STARTUP_PROFILE"] = "123"
        try:
            assert StartupProfile.from_env() == StartupProfile.FULL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]

    def test_is_str_subclass(self):
        assert issubclass(StartupProfile, str)

    def test_unique_values(self):
        values = [p.value for p in StartupProfile]
        assert len(values) == len(set(values))


class TestAllProfilesConstant:
    def test_is_frozenset(self):
        assert isinstance(ALL_PROFILES, frozenset)

    def test_contains_all(self):
        assert StartupProfile.FULL in ALL_PROFILES
        assert StartupProfile.QUICK in ALL_PROFILES
        assert StartupProfile.MINIMAL in ALL_PROFILES

    def test_length(self):
        assert len(ALL_PROFILES) == 3

    def test_all_profiles_are_startup_profile(self):
        for p in ALL_PROFILES:
            assert isinstance(p, StartupProfile)

    def test_frozenset_is_immutable(self):
        with pytest.raises(AttributeError):
            ALL_PROFILES.add(StartupProfile.FULL)


# ---------------------------------------------------------------------------
# StartupHook
# ---------------------------------------------------------------------------

class TestStartupHook:
    def test_defaults(self):
        sh = StartupHook(name="test", handler=lambda: None)
        assert sh.name == "test"
        assert sh.depends_on == []
        assert sh.timeout == 30.0
        assert sh.critical is True
        assert sh.profiles == ALL_PROFILES

    def test_custom_depends(self):
        sh = StartupHook(name="db", handler=lambda: None, depends_on=["config"])
        assert sh.depends_on == ["config"]

    def test_custom_critical(self):
        sh = StartupHook(name="x", handler=lambda: None, critical=False)
        assert sh.critical is False

    def test_custom_timeout(self):
        sh = StartupHook(name="x", handler=lambda: None, timeout=5.0)
        assert sh.timeout == 5.0

    def test_custom_profiles(self):
        profiles = frozenset({StartupProfile.FULL})
        sh = StartupHook(name="x", handler=lambda: None, profiles=profiles)
        assert sh.profiles == profiles

    def test_handler_is_callable(self):
        async def my_handler():
            pass
        sh = StartupHook(name="x", handler=my_handler)
        assert callable(sh.handler)

    def test_dataclass_fields(self):
        fields = {f.name for f in StartupHook.__dataclass_fields__.values()}
        assert "name" in fields
        assert "handler" in fields
        assert "depends_on" in fields
        assert "timeout" in fields
        assert "critical" in fields
        assert "profiles" in fields

    def test_multiple_depends(self):
        sh = StartupHook(name="x", handler=lambda: None, depends_on=["a", "b", "c"])
        assert sh.depends_on == ["a", "b", "c"]

    def test_empty_name(self):
        sh = StartupHook(name="", handler=lambda: None)
        assert sh.name == ""

    def test_zero_timeout(self):
        sh = StartupHook(name="x", handler=lambda: None, timeout=0.0)
        assert sh.timeout == 0.0

    def test_negative_timeout(self):
        sh = StartupHook(name="x", handler=lambda: None, timeout=-1.0)
        assert sh.timeout == -1.0

    def test_profiles_empty_frozenset(self):
        sh = StartupHook(name="x", handler=lambda: None, profiles=frozenset())
        assert sh.profiles == frozenset()

    def test_is_dataclass(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(StartupHook)}
        assert field_names == {"name", "handler", "depends_on", "timeout", "critical", "profiles"}


# ---------------------------------------------------------------------------
# ShutdownHook
# ---------------------------------------------------------------------------

class TestShutdownHook:
    def test_defaults(self):
        sh = ShutdownHook(name="test", handler=lambda: None)
        assert sh.name == "test"
        assert sh.depends_on == []
        assert sh.timeout == 30.0
        assert sh.critical is False

    def test_custom(self):
        sh = ShutdownHook(name="db", handler=lambda: None, depends_on=["cache"], critical=True)
        assert sh.depends_on == ["cache"]
        assert sh.critical is True

    def test_custom_timeout(self):
        sh = ShutdownHook(name="x", handler=lambda: None, timeout=10.0)
        assert sh.timeout == 10.0

    def test_critical_default_false(self):
        sh = ShutdownHook(name="x", handler=lambda: None)
        assert sh.critical is False

    def test_handler_callable(self):
        sh = ShutdownHook(name="x", handler=lambda: None)
        assert callable(sh.handler)

    def test_depends_on_default_empty(self):
        sh = ShutdownHook(name="x", handler=lambda: None)
        assert sh.depends_on == []

    def test_timeout_default(self):
        sh = ShutdownHook(name="x", handler=lambda: None)
        assert sh.timeout == 30.0

    def test_multiple_depends(self):
        sh = ShutdownHook(name="x", handler=lambda: None, depends_on=["a", "b"])
        assert len(sh.depends_on) == 2

    def test_is_dataclass(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(ShutdownHook)}
        assert field_names == {"name", "handler", "depends_on", "timeout", "critical"}


# ---------------------------------------------------------------------------
# _HookResult
# ---------------------------------------------------------------------------

class TestHookResult:
    def test_defaults(self):
        hr = _HookResult(name="test", success=True, elapsed=1.5)
        assert hr.name == "test"
        assert hr.success is True
        assert hr.elapsed == 1.5
        assert hr.error == ""

    def test_failure(self):
        hr = _HookResult(name="x", success=False, elapsed=0.1, error="boom")
        assert hr.success is False
        assert hr.error == "boom"

    def test_zero_elapsed(self):
        hr = _HookResult(name="x", success=True, elapsed=0.0)
        assert hr.elapsed == 0.0

    def test_large_elapsed(self):
        hr = _HookResult(name="x", success=True, elapsed=999.999)
        assert hr.elapsed == 999.999

    def test_empty_name(self):
        hr = _HookResult(name="", success=True, elapsed=0.0)
        assert hr.name == ""

    def test_error_default_empty(self):
        hr = _HookResult(name="x", success=True, elapsed=0.0)
        assert hr.error == ""

    def test_negative_elapsed(self):
        hr = _HookResult(name="x", success=True, elapsed=-1.0)
        assert hr.elapsed == -1.0

    def test_success_with_error_string(self):
        hr = _HookResult(name="x", success=True, elapsed=1.0, error="some warning")
        assert hr.success is True
        assert hr.error == "some warning"

    def test_is_dataclass(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(_HookResult)}
        assert field_names == {"name", "success", "elapsed", "error"}


# ---------------------------------------------------------------------------
# _topological_sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_simple_order(self):
        h1 = StartupHook(name="a", handler=lambda: None)
        h2 = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        result = _topological_sort([h2, h1])
        assert [h.name for h in result] == ["a", "b"]

    def test_chain(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        c = StartupHook(name="c", handler=lambda: None, depends_on=["b"])
        result = _topological_sort([c, b, a])
        names = [h.name for h in result]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_diamond(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        c = StartupHook(name="c", handler=lambda: None, depends_on=["a"])
        d = StartupHook(name="d", handler=lambda: None, depends_on=["b", "c"])
        result = _topological_sort([d, c, b, a])
        names = [h.name for h in result]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_cycle(self):
        h1 = StartupHook(name="a", handler=lambda: None, depends_on=["b"])
        h2 = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        result = _topological_sort([h1, h2])
        assert len(result) == 2

    def test_empty(self):
        result = _topological_sort([])
        assert result == []

    def test_single(self):
        h = StartupHook(name="only", handler=lambda: None)
        result = _topological_sort([h])
        assert len(result) == 1
        assert result[0].name == "only"

    def test_no_deps(self):
        hooks = [StartupHook(name=f"h{i}", handler=lambda: None) for i in range(5)]
        result = _topological_sort(hooks)
        assert len(result) == 5

    def test_depends_on_nonexistent_ignored(self):
        h = StartupHook(name="a", handler=lambda: None, depends_on=["nonexistent"])
        result = _topological_sort([h])
        assert len(result) == 1

    def test_preserves_input_order_when_independent(self):
        hooks = [StartupHook(name=f"z{i}", handler=lambda: None) for i in range(3)]
        result = _topological_sort(hooks)
        assert [h.name for h in result] == ["z0", "z1", "z2"]

    def test_three_level_chain(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        c = StartupHook(name="c", handler=lambda: None, depends_on=["b"])
        d = StartupHook(name="d", handler=lambda: None, depends_on=["c"])
        result = _topological_sort([d, c, b, a])
        names = [h.name for h in result]
        for i in range(len(names) - 1):
            assert names.index(names[i]) <= names.index(names[i + 1])

    def test_complex_dag(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None)
        c = StartupHook(name="c", handler=lambda: None, depends_on=["a"])
        d = StartupHook(name="d", handler=lambda: None, depends_on=["b"])
        e = StartupHook(name="e", handler=lambda: None, depends_on=["a", "b"])
        f = StartupHook(name="f", handler=lambda: None, depends_on=["c", "e"])
        result = _topological_sort([f, e, d, c, b, a])
        names = [h.name for h in result]
        assert names.index("a") < names.index("c")
        assert names.index("a") < names.index("e")
        assert names.index("b") < names.index("d")
        assert names.index("b") < names.index("e")
        assert names.index("c") < names.index("f")
        assert names.index("e") < names.index("f")

    def test_all_hooks_in_result(self):
        hooks = [StartupHook(name=f"h{i}", handler=lambda: None, depends_on=["h0"] if i > 0 else []) for i in range(10)]
        result = _topological_sort(hooks)
        result_names = {h.name for h in result}
        assert result_names == {f"h{i}" for i in range(10)}

    def test_self_dependency_cycle(self):
        h = StartupHook(name="a", handler=lambda: None, depends_on=["a"])
        result = _topological_sort([h])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _dependency_levels
# ---------------------------------------------------------------------------

class TestDependencyLevels:
    def test_empty(self):
        levels = _dependency_levels([])
        assert levels == []

    def test_single_hook(self):
        h = StartupHook(name="a", handler=lambda: None)
        levels = _dependency_levels([h])
        assert len(levels) == 1
        assert len(levels[0]) == 1

    def test_no_deps_parallel(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None)
        levels = _dependency_levels([a, b])
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_chain_sequential(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        c = StartupHook(name="c", handler=lambda: None, depends_on=["b"])
        levels = _dependency_levels([a, b, c])
        assert len(levels) == 3
        assert levels[0][0].name == "a"
        assert levels[1][0].name == "b"
        assert levels[2][0].name == "c"

    def test_diamond_levels(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        c = StartupHook(name="c", handler=lambda: None, depends_on=["a"])
        d = StartupHook(name="d", handler=lambda: None, depends_on=["b", "c"])
        levels = _dependency_levels([a, b, c, d])
        assert len(levels) == 3
        level0_names = {h.name for h in levels[0]}
        assert "a" in level0_names
        level1_names = {h.name for h in levels[1]}
        assert "b" in level1_names or "c" in level1_names

    def test_cycle_breaks(self):
        a = StartupHook(name="a", handler=lambda: None, depends_on=["b"])
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        levels = _dependency_levels([a, b])
        assert len(levels) >= 1
        all_names = {h.name for level in levels for h in level}
        assert all_names == {"a", "b"}

    def test_mixed_deps(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None)
        c = StartupHook(name="c", handler=lambda: None, depends_on=["a"])
        d = StartupHook(name="d", handler=lambda: None, depends_on=["a", "b"])
        levels = _dependency_levels([a, b, c, d])
        all_names = {h.name for level in levels for h in level}
        assert all_names == {"a", "b", "c", "d"}

    def test_unknown_dep_treated_as_resolved(self):
        a = StartupHook(name="a", handler=lambda: None, depends_on=["nonexistent"])
        levels = _dependency_levels([a])
        assert len(levels) == 1

    def test_all_hooks_appear(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        c = StartupHook(name="c", handler=lambda: None, depends_on=["a"])
        levels = _dependency_levels([a, b, c])
        all_names = {h.name for level in levels for h in level}
        assert all_names == {"a", "b", "c"}

    def test_level_ordering_respects_deps(self):
        a = StartupHook(name="a", handler=lambda: None)
        b = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        levels = _dependency_levels([a, b])
        a_level = next(i for i, lv in enumerate(levels) if any(h.name == "a" for h in lv))
        b_level = next(i for i, lv in enumerate(levels) if any(h.name == "b" for h in lv))
        assert a_level < b_level

    def test_many_independent_hooks(self):
        hooks = [StartupHook(name=f"h{i}", handler=lambda: None) for i in range(20)]
        levels = _dependency_levels(hooks)
        assert len(levels) == 1
        assert len(levels[0]) == 20

    def test_deep_chain(self):
        hooks = []
        for i in range(10):
            deps = [f"h{i-1}"] if i > 0 else []
            hooks.append(StartupHook(name=f"h{i}", handler=lambda: None, depends_on=deps))
        levels = _dependency_levels(hooks)
        assert len(levels) == 10


# ---------------------------------------------------------------------------
# Event constants
# ---------------------------------------------------------------------------

class TestEventConstants:
    def test_phase_changed(self):
        assert EVT_PHASE_CHANGED == "lifecycle.phase_changed"

    def test_hook_started(self):
        assert EVT_HOOK_STARTED == "lifecycle.hook_started"

    def test_hook_completed(self):
        assert EVT_HOOK_COMPLETED == "lifecycle.hook_completed"

    def test_hook_failed(self):
        assert EVT_HOOK_FAILED == "lifecycle.hook_failed"

    def test_all_are_strings(self):
        assert isinstance(EVT_PHASE_CHANGED, str)
        assert isinstance(EVT_HOOK_STARTED, str)
        assert isinstance(EVT_HOOK_COMPLETED, str)
        assert isinstance(EVT_HOOK_FAILED, str)

    def test_all_are_unique(self):
        constants = [EVT_PHASE_CHANGED, EVT_HOOK_STARTED, EVT_HOOK_COMPLETED, EVT_HOOK_FAILED]
        assert len(constants) == len(set(constants))


# ---------------------------------------------------------------------------
# LifecycleManager — unit tests
# ---------------------------------------------------------------------------

class TestLifecycleManagerInit:
    def test_initial_phase(self):
        mgr = LifecycleManager()
        assert mgr.phase == LifecyclePhase.INIT

    def test_is_running_false(self):
        mgr = LifecycleManager()
        assert mgr.is_running() is False

    def test_is_draining_false(self):
        mgr = LifecycleManager()
        assert mgr.is_draining() is False

    def test_uptime_zero(self):
        mgr = LifecycleManager()
        assert mgr.uptime_seconds == 0.0

    def test_in_flight_zero(self):
        mgr = LifecycleManager()
        assert mgr.in_flight_count == 0

    def test_get_profile_default(self):
        mgr = LifecycleManager()
        assert mgr.get_profile() == StartupProfile.FULL

    def test_startup_results_empty(self):
        mgr = LifecycleManager()
        assert mgr.startup_results == []

    def test_shutdown_results_empty(self):
        mgr = LifecycleManager()
        assert mgr.shutdown_results == []

    def test_event_bus_none_by_default(self):
        mgr = LifecycleManager()
        assert mgr._event_bus is None

    def test_health_gates_empty(self):
        mgr = LifecycleManager()
        assert mgr._health_gates == {}

    def test_uptime_after_start(self):
        mgr = LifecycleManager()
        mgr._started_at = time.time() - 1.0
        assert mgr.uptime_seconds >= 0.9

    def test_is_draining_during_stopping(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STOPPING
        assert mgr.is_draining() is True

    def test_is_running_only_in_running_phase(self):
        mgr = LifecycleManager()
        for phase in LifecyclePhase:
            mgr._phase = phase
            expected = phase == LifecyclePhase.RUNNING
            assert mgr.is_running() is expected


class TestLifecycleManagerHooks:
    def test_register_startup_hook(self):
        mgr = LifecycleManager()
        hook = StartupHook(name="db", handler=lambda: None)
        mgr.register_startup_hook(hook)
        assert len(mgr._startup_hooks) == 1

    def test_register_shutdown_hook(self):
        mgr = LifecycleManager()
        hook = ShutdownHook(name="db", handler=lambda: None)
        mgr.register_shutdown_hook(hook)
        assert len(mgr._shutdown_hooks) == 1

    def test_duplicate_startup_hook_skipped(self):
        mgr = LifecycleManager()
        h1 = StartupHook(name="db", handler=lambda: None)
        h2 = StartupHook(name="db", handler=lambda: None)
        mgr.register_startup_hook(h1)
        mgr.register_startup_hook(h2)
        assert len(mgr._startup_hooks) == 1

    def test_duplicate_shutdown_hook_skipped(self):
        mgr = LifecycleManager()
        h1 = ShutdownHook(name="db", handler=lambda: None)
        h2 = ShutdownHook(name="db", handler=lambda: None)
        mgr.register_shutdown_hook(h1)
        mgr.register_shutdown_hook(h2)
        assert len(mgr._shutdown_hooks) == 1

    def test_preview_empty(self):
        mgr = LifecycleManager()
        assert mgr.preview() == []

    def test_preview_with_hooks(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=lambda: None))
        mgr.register_startup_hook(StartupHook(name="b", handler=lambda: None, depends_on=["a"]))
        preview = mgr.preview()
        assert len(preview) == 2
        assert preview[0]["name"] == "a"
        assert preview[1]["name"] == "b"

    def test_preview_with_profile_filter(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(
            name="a", handler=lambda: None,
            profiles=frozenset({StartupProfile.FULL}),
        ))
        mgr.register_startup_hook(StartupHook(
            name="b", handler=lambda: None,
            profiles=frozenset({StartupProfile.MINIMAL}),
        ))
        preview = mgr.preview(profile=StartupProfile.FULL)
        assert len(preview) == 1
        assert preview[0]["name"] == "a"

    def test_preview_shows_critical_timeout_depends(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(
            name="x", handler=lambda: None, critical=False, timeout=5.0, depends_on=["y"],
        ))
        preview = mgr.preview()
        assert preview[0]["critical"] is False
        assert preview[0]["timeout"] == 5.0
        assert preview[0]["depends_on"] == ["y"]

    def test_register_multiple_hooks(self):
        mgr = LifecycleManager()
        for i in range(5):
            mgr.register_startup_hook(StartupHook(name=f"h{i}", handler=lambda: None))
        assert len(mgr._startup_hooks) == 5

    def test_register_multiple_shutdown_hooks(self):
        mgr = LifecycleManager()
        for i in range(5):
            mgr.register_shutdown_hook(ShutdownHook(name=f"h{i}", handler=lambda: None))
        assert len(mgr._shutdown_hooks) == 5


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

    def test_unregister_nonexistent_ok(self):
        mgr = LifecycleManager()
        mgr.unregister_gate("nonexistent")

    def test_gate_ready_true(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        assert mgr.gate_ready("db") is True

    def test_gate_ready_false(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: False)
        assert mgr.gate_ready("db") is False

    def test_gate_not_registered_returns_true(self):
        mgr = LifecycleManager()
        assert mgr.gate_ready("nonexistent") is True

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

    def test_gates_ready_empty(self):
        mgr = LifecycleManager()
        assert mgr.gates_ready() is True

    @pytest.mark.asyncio
    async def test_wait_for_gates_immediate(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        assert await mgr.wait_for_gates(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_wait_for_gates_timeout(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: False)
        assert await mgr.wait_for_gates(timeout=0.1, poll_interval=0.01) is False

    @pytest.mark.asyncio
    async def test_wait_for_gates_becomes_ready(self):
        mgr = LifecycleManager()
        attempts = [False, False, True]
        mgr.register_gate("db", lambda: attempts.pop(0) if attempts else True)
        assert await mgr.wait_for_gates(timeout=2.0, poll_interval=0.01) is True

    def test_gate_replaces_previous(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: False)
        mgr.register_gate("db", lambda: True)
        assert mgr.gate_ready("db") is True

    def test_gate_stores_result(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        mgr.gate_ready("db")
        assert mgr._gate_results["db"] is True


class TestLifecycleManagerInFlight:
    @pytest.mark.asyncio
    async def test_acquire_in_flight(self):
        mgr = LifecycleManager()
        ok = await mgr.acquire_in_flight()
        assert ok is True
        assert mgr.in_flight_count == 1

    @pytest.mark.asyncio
    async def test_release_in_flight(self):
        mgr = LifecycleManager()
        await mgr.acquire_in_flight()
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_release_does_not_go_negative(self):
        mgr = LifecycleManager()
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_acquire_during_drain_returns_false(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.DRAINING
        ok = await mgr.acquire_in_flight()
        assert ok is False

    @pytest.mark.asyncio
    async def test_acquire_during_stopping_returns_false(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STOPPING
        ok = await mgr.acquire_in_flight()
        assert ok is False

    @pytest.mark.asyncio
    async def test_multiple_acquire_release(self):
        mgr = LifecycleManager()
        await mgr.acquire_in_flight()
        await mgr.acquire_in_flight()
        assert mgr.in_flight_count == 2
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 1
        await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_acquire_after_release(self):
        mgr = LifecycleManager()
        await mgr.acquire_in_flight()
        await mgr.release_in_flight()
        ok = await mgr.acquire_in_flight()
        assert ok is True
        assert mgr.in_flight_count == 1


class TestLifecycleManagerStart:
    @pytest.mark.asyncio
    async def test_start_no_hooks(self):
        mgr = LifecycleManager()
        result = await mgr.start()
        assert result is True
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_start_sets_uptime(self):
        mgr = LifecycleManager()
        await mgr.start()
        assert mgr.started_at > 0
        assert mgr.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_start_ignores_if_not_init(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.RUNNING
        result = await mgr.start()
        assert result is True

    @pytest.mark.asyncio
    async def test_start_with_successful_hook(self):
        called = []

        async def hook():
            called.append(True)

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=hook))
        result = await mgr.start()
        assert result is True
        assert called == [True]
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_start_critical_hook_failure_crashes(self):
        async def fail():
            raise RuntimeError("boom")

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=fail, critical=True))
        result = await mgr.start()
        assert result is False
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_start_noncritical_hook_failure_continues(self):
        async def fail():
            raise RuntimeError("boom")

        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=fail, critical=False))
        mgr.register_startup_hook(StartupHook(name="cache", handler=ok))
        result = await mgr.start()
        assert result is True
        assert mgr.phase == LifecyclePhase.RUNNING

    @pytest.mark.asyncio
    async def test_start_with_dependencies(self):
        order = []

        async def a():
            order.append("a")

        async def b():
            order.append("b")

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=a))
        mgr.register_startup_hook(StartupHook(name="b", handler=b, depends_on=["a"]))
        await mgr.start()
        assert order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_start_records_results(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=ok))
        await mgr.start()
        results = mgr.startup_results
        assert len(results) == 1
        assert results[0]["name"] == "db"
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_start_timeout_hook(self):
        async def slow():
            await asyncio.sleep(100)

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="slow", handler=slow, timeout=0.05, critical=True))
        result = await mgr.start()
        assert result is False
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_start_parallel_hooks(self):
        order = []

        async def a():
            order.append("a")

        async def b():
            order.append("b")

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=a))
        mgr.register_startup_hook(StartupHook(name="b", handler=b))
        await mgr.start()
        assert set(order) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_start_with_profile(self):
        called = []

        async def full_only():
            called.append("full")

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(
            name="full_only", handler=full_only,
            profiles=frozenset({StartupProfile.FULL}),
        ))
        await mgr.start(profile=StartupProfile.FULL)
        assert "full" in called

    @pytest.mark.asyncio
    async def test_start_skips_profile_mismatch(self):
        called = []

        async def full_only():
            called.append("full")

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(
            name="full_only", handler=full_only,
            profiles=frozenset({StartupProfile.FULL}),
        ))
        await mgr.start(profile=StartupProfile.MINIMAL)
        assert "full" not in called

    @pytest.mark.asyncio
    async def test_start_emits_events(self):
        events = []

        class FakeBus:
            async def emit(self, event, data, source=""):
                events.append(event)
            def emit_sync(self, event, data, source=""):
                events.append(event)

        mgr = LifecycleManager(event_bus=FakeBus())
        await mgr.start()
        assert EVT_PHASE_CHANGED in events

    @pytest.mark.asyncio
    async def test_start_results_include_elapsed(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=ok))
        await mgr.start()
        results = mgr.startup_results
        assert "elapsed" in results[0]
        assert results[0]["elapsed"] >= 0

    @pytest.mark.asyncio
    async def test_start_results_include_error_field(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=ok))
        await mgr.start()
        results = mgr.startup_results
        assert results[0]["error"] == ""


class TestLifecycleManagerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_from_running(self):
        mgr = LifecycleManager()
        await mgr.start()
        result = await mgr.shutdown()
        assert result is True
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_ignores_if_not_running(self):
        mgr = LifecycleManager()
        result = await mgr.shutdown()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_with_hooks(self):
        called = []

        async def hook():
            called.append(True)

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="db", handler=hook))
        await mgr.start()
        await mgr.shutdown()
        assert called == [True]

    @pytest.mark.asyncio
    async def test_shutdown_records_results(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="db", handler=ok))
        await mgr.start()
        await mgr.shutdown()
        results = mgr.shutdown_results
        assert len(results) == 1
        assert results[0]["name"] == "db"
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_shutdown_from_crashed(self):
        async def fail():
            raise RuntimeError("boom")

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="x", handler=fail))
        await mgr.start()
        assert mgr.phase == LifecyclePhase.CRASHED
        result = await mgr.shutdown()
        assert result is True
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_hook_failure_handled(self):
        async def fail():
            raise RuntimeError("shutdown boom")

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="db", handler=fail))
        await mgr.start()
        result = await mgr.shutdown()
        assert result is True
        assert mgr.phase == LifecyclePhase.STOPPED
        results = mgr.shutdown_results
        assert results[0]["success"] is False
        assert "shutdown boom" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_shutdown_results_have_elapsed(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="db", handler=ok))
        await mgr.start()
        await mgr.shutdown()
        results = mgr.shutdown_results
        assert "elapsed" in results[0]

    @pytest.mark.asyncio
    async def test_shutdown_results_have_error_field(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="db", handler=ok))
        await mgr.start()
        await mgr.shutdown()
        results = mgr.shutdown_results
        assert "error" in results[0]


class TestLifecycleManagerMarkCrashed:
    @pytest.mark.asyncio
    async def test_mark_crashed(self):
        mgr = LifecycleManager()
        await mgr.mark_crashed("test")
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_mark_crashed_ignores_stopped(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STOPPED
        await mgr.mark_crashed("test")
        assert mgr.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_mark_crashed_ignores_already_crashed(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.CRASHED
        await mgr.mark_crashed("test")
        assert mgr.phase == LifecyclePhase.CRASHED

    @pytest.mark.asyncio
    async def test_mark_crashed_from_running(self):
        mgr = LifecycleManager()
        await mgr.start()
        assert mgr.phase == LifecyclePhase.RUNNING
        await mgr.mark_crashed("failure")
        assert mgr.phase == LifecyclePhase.CRASHED


class TestLifecycleManagerGetResults:
    def test_get_results(self):
        mgr = LifecycleManager()
        result = mgr.get_results()
        assert "phase" in result
        assert "profile" in result
        assert "uptime" in result
        assert "in_flight" in result
        assert "hooks" in result
        assert "gates" in result
        assert result["phase"] == "init"
        assert result["hooks"]["startup"] == 0

    def test_get_results_with_hooks(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=lambda: None))
        mgr.register_shutdown_hook(ShutdownHook(name="b", handler=lambda: None))
        result = mgr.get_results()
        assert result["hooks"]["startup"] == 1
        assert result["hooks"]["shutdown"] == 1

    def test_get_results_has_gates_counts(self):
        mgr = LifecycleManager()
        mgr.register_gate("db", lambda: True)
        result = mgr.get_results()
        assert result["gates"]["total"] == 1

    def test_get_results_has_preview(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=lambda: None))
        result = mgr.get_results()
        assert "preview" in result["hooks"]

    def test_get_results_profile_value(self):
        mgr = LifecycleManager()
        result = mgr.get_results()
        assert result["profile"] in ("full", "quick", "minimal")


# ---------------------------------------------------------------------------
# Singleton functions
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_lifecycle_manager_returns_same(self):
        reset_lifecycle_manager()
        m1 = get_lifecycle_manager()
        m2 = get_lifecycle_manager()
        assert m1 is m2
        reset_lifecycle_manager()

    def test_reset_clears_singleton(self):
        reset_lifecycle_manager()
        m1 = get_lifecycle_manager()
        reset_lifecycle_manager()
        m2 = get_lifecycle_manager()
        assert m1 is not m2
        reset_lifecycle_manager()

    def test_singleton_is_lifecycle_manager_instance(self):
        reset_lifecycle_manager()
        mgr = get_lifecycle_manager()
        assert isinstance(mgr, LifecycleManager)
        reset_lifecycle_manager()

    def test_reset_multiple_times(self):
        reset_lifecycle_manager()
        reset_lifecycle_manager()
        mgr = get_lifecycle_manager()
        assert isinstance(mgr, LifecycleManager)
        reset_lifecycle_manager()

    def test_get_with_event_bus(self):
        reset_lifecycle_manager()
        bus = object()
        mgr = get_lifecycle_manager(event_bus=bus)
        assert mgr._event_bus is bus
        reset_lifecycle_manager()

    def test_get_singleton_does_not_replace_existing(self):
        reset_lifecycle_manager()
        m1 = get_lifecycle_manager()
        m2 = get_lifecycle_manager(event_bus=object())
        assert m1 is m2
        reset_lifecycle_manager()


# ---------------------------------------------------------------------------
# Extended LifecycleManager — more edge cases
# ---------------------------------------------------------------------------

class TestLifecycleManagerExtended:
    def test_is_draining_in_init(self):
        mgr = LifecycleManager()
        assert mgr.is_draining() is False

    def test_is_draining_in_starting(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STARTING
        assert mgr.is_draining() is False

    def test_is_draining_in_stopped(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STOPPED
        assert mgr.is_draining() is False

    def test_uptime_zero_before_start(self):
        mgr = LifecycleManager()
        assert mgr.uptime_seconds == 0.0
        assert mgr.started_at == 0.0

    def test_profile_default_is_full(self):
        mgr = LifecycleManager()
        assert mgr.get_profile() == StartupProfile.FULL

    def test_profile_set_during_start(self):
        mgr = LifecycleManager()
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mgr.start(profile=StartupProfile.MINIMAL)
        )
        assert mgr.get_profile() == StartupProfile.MINIMAL
        asyncio.get_event_loop().run_until_complete(mgr.shutdown())

    def test_in_flight_count_property(self):
        mgr = LifecycleManager()
        assert isinstance(mgr.in_flight_count, int)
        assert mgr.in_flight_count == 0

    def test_startup_results_property_type(self):
        mgr = LifecycleManager()
        results = mgr.startup_results
        assert isinstance(results, list)

    def test_shutdown_results_property_type(self):
        mgr = LifecycleManager()
        results = mgr.shutdown_results
        assert isinstance(results, list)

    def test_event_bus_none_default(self):
        mgr = LifecycleManager()
        assert mgr._event_bus is None

    def test_health_gates_dict_empty(self):
        mgr = LifecycleManager()
        assert isinstance(mgr._health_gates, dict)
        assert len(mgr._health_gates) == 0

    def test_lock_is_asyncio_lock(self):
        import asyncio
        mgr = LifecycleManager()
        assert isinstance(mgr._lock, asyncio.Lock)

    def test_drain_event_set_by_default(self):
        mgr = LifecycleManager()
        assert mgr._drain_event.is_set()

    @pytest.mark.asyncio
    async def test_start_emits_started_event(self):
        events = []

        class FakeBus:
            async def emit(self, event, data, source=""):
                events.append(event)
            def emit_sync(self, event, data, source=""):
                events.append(event)

        mgr = LifecycleManager(event_bus=FakeBus())
        await mgr.start()
        assert "lifecycle.started" in events

    @pytest.mark.asyncio
    async def test_shutdown_emits_stopped_event(self):
        events = []

        class FakeBus:
            async def emit(self, event, data, source=""):
                events.append(event)
            def emit_sync(self, event, data, source=""):
                events.append(event)

        mgr = LifecycleManager(event_bus=FakeBus())
        await mgr.start()
        await mgr.shutdown()
        assert "lifecycle.stopped" in events

    @pytest.mark.asyncio
    async def test_mark_crashed_emits_event(self):
        events = []

        class FakeBus:
            async def emit(self, event, data, source=""):
                events.append(event)
            def emit_sync(self, event, data, source=""):
                events.append(event)

        mgr = LifecycleManager(event_bus=FakeBus())
        await mgr.mark_crashed("test reason")
        assert "lifecycle.crashed" in events

    @pytest.mark.asyncio
    async def test_start_crashed_emits_crashed_event(self):
        events = []

        class FakeBus:
            async def emit(self, event, data, source=""):
                events.append(event)
            def emit_sync(self, event, data, source=""):
                events.append(event)

        async def fail():
            raise RuntimeError("boom")

        mgr = LifecycleManager(event_bus=FakeBus())
        mgr.register_startup_hook(StartupHook(name="x", handler=fail, critical=True))
        await mgr.start()
        assert "lifecycle.crashed" in events

    @pytest.mark.asyncio
    async def test_wait_for_gates_emits_event(self):
        events = []

        class FakeBus:
            async def emit(self, event, data, source=""):
                events.append(event)
            def emit_sync(self, event, data, source=""):
                events.append(event)

        mgr = LifecycleManager(event_bus=FakeBus())
        mgr.register_gate("db", lambda: True)
        await mgr.wait_for_gates(timeout=1.0)
        assert "lifecycle.gates_passed" in events

    def test_get_results_has_started_at(self):
        mgr = LifecycleManager()
        result = mgr.get_results()
        assert "started_at" in result
        assert result["started_at"] == 0.0

    def test_get_results_uptime_is_float(self):
        mgr = LifecycleManager()
        result = mgr.get_results()
        assert isinstance(result["uptime"], float)

    def test_preview_returns_list_of_dicts(self):
        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=lambda: None))
        preview = mgr.preview()
        assert isinstance(preview, list)
        assert isinstance(preview[0], dict)

    @pytest.mark.asyncio
    async def test_acquire_release_multiple_times(self):
        mgr = LifecycleManager()
        for _ in range(5):
            ok = await mgr.acquire_in_flight()
            assert ok is True
        assert mgr.in_flight_count == 5
        for _ in range(5):
            await mgr.release_in_flight()
        assert mgr.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_multiple_hooks(self):
        order = []

        async def a():
            order.append("a")

        async def b():
            order.append("b")

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="a", handler=a))
        mgr.register_shutdown_hook(ShutdownHook(name="b", handler=b))
        await mgr.start()
        await mgr.shutdown()
        assert len(order) == 2

    @pytest.mark.asyncio
    async def test_start_results_include_all_hooks(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="h1", handler=ok))
        mgr.register_startup_hook(StartupHook(name="h2", handler=ok))
        await mgr.start()
        names = {r["name"] for r in mgr.startup_results}
        assert "h1" in names
        assert "h2" in names

    def test_filter_hooks_for_profile(self):
        mgr = LifecycleManager()
        h1 = StartupHook(name="full_only", handler=lambda: None, profiles=frozenset({StartupProfile.FULL}))
        h2 = StartupHook(name="all", handler=lambda: None)
        filtered = mgr._filter_hooks_for_profile([h1, h2], StartupProfile.FULL)
        assert len(filtered) == 2
        filtered = mgr._filter_hooks_for_profile([h1, h2], StartupProfile.MINIMAL)
        assert len(filtered) == 1
        assert filtered[0].name == "all"

    def test_emit_sync_no_bus(self):
        mgr = LifecycleManager()
        mgr._emit_sync("test.event", {"key": "val"})
        assert mgr._event_bus is None

    def test_emit_sync_with_bus(self):
        emitted = []

        class FakeBus:
            def emit_sync(self, event, data, source=""):
                emitted.append((event, data))

        mgr = LifecycleManager(event_bus=FakeBus())
        mgr._emit_sync("test.event", {"key": "val"})
        assert len(emitted) == 1
        assert emitted[0][0] == "test.event"

    def test_emit_sync_bus_exception_handled(self):
        class BadBus:
            def emit_sync(self, event, data, source=""):
                raise RuntimeError("bus error")

        mgr = LifecycleManager(event_bus=BadBus())
        mgr._emit_sync("test.event", {})

    @pytest.mark.asyncio
    async def test_shutdown_from_init_returns_false(self):
        mgr = LifecycleManager()
        result = await mgr.shutdown()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_from_starting_returns_false(self):
        mgr = LifecycleManager()
        mgr._phase = LifecyclePhase.STARTING
        result = await mgr.shutdown()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_parallel_level_critical_failure(self):
        async def fail():
            raise RuntimeError("fail")

        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="a", handler=fail, critical=True))
        mgr.register_startup_hook(StartupHook(name="b", handler=ok))
        result = await mgr.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_timeout_handled(self):
        async def slow():
            await asyncio.sleep(100)

        mgr = LifecycleManager()
        mgr.register_shutdown_hook(ShutdownHook(name="slow", handler=slow, timeout=0.05))
        await mgr.start()
        result = await mgr.shutdown()
        assert result is True
        assert mgr.shutdown_results[0]["success"] is False

    @pytest.mark.asyncio
    async def test_get_results_after_start(self):
        async def ok():
            pass

        mgr = LifecycleManager()
        mgr.register_startup_hook(StartupHook(name="db", handler=ok))
        await mgr.start()
        result = mgr.get_results()
        assert result["phase"] == "running"
        assert result["hooks"]["startup"] == 1
        assert len(result["hooks"]["startup_results"]) == 1
