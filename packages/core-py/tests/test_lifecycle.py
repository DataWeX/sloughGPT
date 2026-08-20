"""Tests for domains.infrastructure.lifecycle — LifecyclePhase, StartupProfile, hooks, topological sort, dependency levels."""

import asyncio
from domains.infrastructure.lifecycle import (
    LifecyclePhase, StartupProfile, StartupHook, ShutdownHook, _HookResult,
    _topological_sort, _dependency_levels,
)


class TestLifecyclePhase:
    def test_all_members(self):
        assert len(LifecyclePhase) == 7
    def test_values(self):
        assert LifecyclePhase.INIT.value == "init"
        assert LifecyclePhase.RUNNING.value == "running"
        assert LifecyclePhase.CRASHED.value == "crashed"


class TestStartupProfile:
    def test_all_members(self):
        assert len(StartupProfile) == 3
    def test_from_env_full(self):
        import os
        os.environ["SLO_STARTUP_PROFILE"] = "full"
        assert StartupProfile.from_env() == StartupProfile.FULL
    def test_from_env_unknown(self):
        import os
        os.environ["SLO_STARTUP_PROFILE"] = "bogus"
        assert StartupProfile.from_env() == StartupProfile.FULL


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


class TestShutdownHook:
    def test_defaults(self):
        async def noop(): pass
        h = ShutdownHook(name="a", handler=noop)
        assert h.critical is False


class TestHookResult:
    def test_fields(self):
        r = _HookResult(name="x", success=True, elapsed=1.5)
        assert r.error == ""


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
