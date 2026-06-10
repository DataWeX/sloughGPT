"""
Tests for Shell Init System — ServiceDef, ServiceManager, InitSystem.
"""

import time
import pytest
from domains.shell.init import (
    ServiceDef, ServiceInstance, ServiceManager, InitSystem,
    get_init_system, reset_init_system,
)


# ── ServiceDef ───────────────────────────────────────────────────────────


class TestServiceDef:
    def test_default_values(self):
        s = ServiceDef(name="test")
        assert s.name == "test"
        assert s.command == ""
        assert s.deps == []
        assert s.respawn is True
        assert s.max_respawns == 3
        assert s.respawn_delay == 2.0
        assert s.runlevel == 2
        assert s.timeout == 30.0
        assert s.description == ""
        assert s.builtin is False

    def test_override_values(self):
        s = ServiceDef(name="web", command="python server.py", deps=["db"],
                       respawn=True, max_respawns=5, runlevel=3)
        assert s.name == "web"
        assert s.deps == ["db"]
        assert s.max_respawns == 5


# ── ServiceInstance ──────────────────────────────────────────────────────


class TestServiceInstance:
    def test_default_state(self):
        d = ServiceDef(name="test")
        i = ServiceInstance(definition=d)
        assert i.state == "stopped"
        assert i.pid == 0
        assert i.started_at == 0.0
        assert i.respawn_count == 0
        assert i.uptime == 0.0

    def test_uptime_running(self):
        d = ServiceDef(name="test")
        i = ServiceInstance(definition=d, state="running", started_at=time.time())
        assert i.uptime > 0.0

    def test_uptime_stopped(self):
        d = ServiceDef(name="test")
        i = ServiceInstance(definition=d)
        assert i.uptime == 0.0


# ── ServiceManager ───────────────────────────────────────────────────────


class TestServiceManager:
    def test_create_manager(self):
        d = ServiceDef(name="test", builtin=True)
        m = ServiceManager(d)
        assert m.defn.name == "test"
        assert m.instance.state == "stopped"

    def test_start_builtin(self):
        d = ServiceDef(name="test", builtin=True)
        m = ServiceManager(d)
        result = m.start()
        assert result is True
        assert m.instance.state == "running"

    def test_start_builtin_idempotent(self):
        d = ServiceDef(name="test", builtin=True)
        m = ServiceManager(d)
        assert m.start() is True
        assert m.start() is True

    def test_stop_builtin(self):
        d = ServiceDef(name="test", builtin=True)
        m = ServiceManager(d)
        m.start()
        result = m.stop()
        assert result is True
        assert m.instance.state == "stopped"

    def test_start_no_command_fails(self):
        d = ServiceDef(name="test", builtin=False, command="")
        m = ServiceManager(d)
        result = m.start()
        assert result is False
        assert m.instance.state == "failed"

    def test_is_alive_builtin(self):
        d = ServiceDef(name="test", builtin=True)
        m = ServiceManager(d)
        m.start()
        assert m.is_alive is True

    def test_is_alive_not_started(self):
        d = ServiceDef(name="test", builtin=False)
        m = ServiceManager(d)
        assert m.is_alive is False

    def test_restart_builtin(self):
        d = ServiceDef(name="test", builtin=True)
        m = ServiceManager(d)
        assert m.start() is True
        assert m.restart() is True
        assert m.instance.state == "running"

    def test_status_line(self):
        d = ServiceDef(name="test-svc", builtin=True)
        m = ServiceManager(d)
        m.start()
        line = m.status_line()
        assert "test-svc" in line
        assert "running" in line


# ── InitSystem ───────────────────────────────────────────────────────────


class TestInitSystem:
    def test_load_builtins(self):
        init = InitSystem()
        mgr = init.get_manager("kernel")
        assert mgr is not None
        assert mgr.defn.name == "kernel"

    def test_get_manager_nonexistent(self):
        init = InitSystem()
        assert init.get_manager("nonexistent") is None

    def test_services_property(self):
        init = InitSystem()
        assert len(init.services) >= 4

    def test_services_all_have_runlevels(self):
        init = InitSystem()
        for m in init.services:
            assert 1 <= m.defn.runlevel <= 3

    def test_boot_runlevel_1(self):
        init = InitSystem()
        output = init.boot(target_runlevel=1)
        assert "Booting" in output
        assert "kernel" in output
        mgr = init.get_manager("kernel")
        assert mgr.instance.state == "running"

    def test_boot_runlevel_2(self):
        init = InitSystem()
        output = init.boot(target_runlevel=2)
        assert "model-server" in output

    def test_boot_complete_flag(self):
        init = InitSystem()
        init.boot(target_runlevel=1)
        assert init._boot_complete is True

    def test_boot_status_summary(self):
        init = InitSystem()
        init.boot(target_runlevel=1)
        summary = init.status_summary
        assert "Runlevel: 1" in summary
        assert "kernel" in summary or "Uptime" in summary

    def test_shutdown(self):
        init = InitSystem()
        init.boot(target_runlevel=1)
        output = init.shutdown()
        assert "shutdown" in output or "halted" in output
        assert init._boot_complete is False
        assert init._current_runlevel == 0

    def test_service_table(self):
        init = InitSystem()
        table = init.service_table()
        assert "kernel" in table or "No services" in table

    def test_uptime_before_boot(self):
        init = InitSystem()
        assert init.uptime == 0.0

    def test_uptime_after_boot(self):
        init = InitSystem()
        init.boot(target_runlevel=1)
        assert init.uptime > 0.0


class TestInitSystemDeps:
    def test_resolve_deps_preserves_all(self):
        from domains.shell.init import ServiceManager, ServiceDef
        svc1 = ServiceManager(ServiceDef(name="a", runlevel=1, deps=[], builtin=True))
        svc2 = ServiceManager(ServiceDef(name="b", runlevel=1, deps=["a"], builtin=True))
        init = InitSystem()
        init._managers = {"a": svc1, "b": svc2}
        ordered = init._resolve_deps([svc1, svc2])
        names = [m.defn.name for m in ordered]
        assert "a" in names
        assert "b" in names

    def test_dependency_before_dependent(self):
        from domains.shell.init import ServiceManager, ServiceDef
        svc1 = ServiceManager(ServiceDef(name="a", runlevel=1, deps=[], builtin=True))
        svc2 = ServiceManager(ServiceDef(name="b", runlevel=1, deps=["a"], builtin=True))
        init = InitSystem()
        init._managers = {"a": svc1, "b": svc2}
        ordered = init._resolve_deps([svc2, svc1])
        names = [m.defn.name for m in ordered]
        assert names.index("a") < names.index("b")


class TestInitSystemSingleton:
    def test_get_init_system(self):
        reset_init_system()
        s1 = get_init_system()
        s2 = get_init_system()
        assert s1 is s2

    def test_reset_init_system(self):
        reset_init_system()
        s1 = get_init_system()
        reset_init_system()
        s2 = get_init_system()
        assert s1 is not s2
