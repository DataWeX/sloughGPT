"""Tests for domains.infrastructure.lifecycle — LifecyclePhase, StartupProfile, StartupHook, ShutdownHook, _HookResult, _topological_sort."""

from domains.infrastructure.lifecycle import (
    LifecyclePhase, StartupProfile, StartupHook, ShutdownHook, _HookResult, _topological_sort,
)


class TestLifecyclePhase:
    def test_all_members(self):
        assert len(LifecyclePhase) == 7
    def test_values(self):
        assert LifecyclePhase.INIT.value == "init"
        assert LifecyclePhase.CRASHED.value == "crashed"
    def test_str_enum(self):
        assert LifecyclePhase.RUNNING == "running"


class TestStartupProfile:
    def test_all_members(self):
        assert len(StartupProfile) == 3
    def test_values(self):
        assert StartupProfile.FULL.value == "full"
        assert StartupProfile.MINIMAL.value == "minimal"
    def test_from_env(self):
        import os
        os.environ["SLO_STARTUP_PROFILE"] = "minimal"
        try:
            assert StartupProfile.from_env() == StartupProfile.MINIMAL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]
    def test_from_env_unknown(self):
        import os
        os.environ["SLO_STARTUP_PROFILE"] = "garbage"
        try:
            assert StartupProfile.from_env() == StartupProfile.FULL
        finally:
            del os.environ["SLO_STARTUP_PROFILE"]


class TestStartupHook:
    def test_defaults(self):
        sh = StartupHook(name="test", handler=lambda: None)
        assert sh.name == "test"
        assert sh.depends_on == []
        assert sh.timeout == 30.0
        assert sh.critical is True

    def test_custom(self):
        sh = StartupHook(name="db", handler=lambda: None, depends_on=["config"], critical=False)
        assert sh.depends_on == ["config"]
        assert sh.critical is False


class TestShutdownHook:
    def test_defaults(self):
        sh = ShutdownHook(name="test", handler=lambda: None)
        assert sh.name == "test"
        assert sh.critical is False


class TestHookResult:
    def test_defaults(self):
        hr = _HookResult(name="test", success=True, elapsed=1.5)
        assert hr.name == "test"
        assert hr.success is True
        assert hr.elapsed == 1.5
        assert hr.error == ""


class TestTopologicalSort:
    def test_simple_order(self):
        h1 = StartupHook(name="a", handler=lambda: None)
        h2 = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        result = _topological_sort([h2, h1])
        assert [h.name for h in result] == ["a", "b"]

    def test_cycle(self):
        h1 = StartupHook(name="a", handler=lambda: None, depends_on=["b"])
        h2 = StartupHook(name="b", handler=lambda: None, depends_on=["a"])
        result = _topological_sort([h1, h2])
        assert len(result) == 2
