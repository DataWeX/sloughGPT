"""Coverage tests for domains/shell/init.py."""

import json
import shlex
import sys

import pytest

from domains.shell import init as init_mod
from domains.shell.init import (
    InitSystem,
    SERVICE_STATES,
    ServiceDef,
    ServiceInstance,
    ServiceManager,
    get_init_system,
    reset_init_system,
)


class FakeProc:
    def __init__(self, rc=0):
        self.rc = rc
        self.called_wait = False

    def wait(self):
        self.called_wait = True
        return self.rc

    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass


class RaisingWaitProc(FakeProc):
    def wait(self):
        raise TimeoutError("timed out")


class RaisingKillProc(FakeProc):
    def kill(self):
        raise OSError("kill failed")


def test_service_states_constant():
    assert SERVICE_STATES == ["stopped", "starting", "running", "stopping", "failed", "crashed"]


def test_builtin_services_present():
    assert "kernel" in init_mod.BUILTIN_SERVICES
    assert "agent-orchestrator" in init_mod.BUILTIN_SERVICES
    assert "knowledge-worker" in init_mod.BUILTIN_SERVICES
    assert init_mod.BUILTIN_SERVICES["kernel"]["runlevel"] == 1


def test_service_def_defaults():
    d = ServiceDef(name="svc")
    assert d.respawn is True
    assert d.max_respawns == 3
    assert d.runlevel == 2
    assert d.builtin is False


def test_service_instance_uptime():
    inst = ServiceInstance(definition=ServiceDef(name="svc"))
    assert inst.uptime == 0.0
    inst.state = "running"
    inst.started_at = 1.0
    assert inst.uptime > 0.0


def test_manager_start_builtin():
    m = ServiceManager(ServiceDef(name="k", builtin=True))
    assert m.start() is True
    assert m.instance.state == "running"
    assert "built-in service registered" in m.instance.log[-1]


def test_manager_start_already_running():
    m = ServiceManager(ServiceDef(name="k", builtin=True))
    m.start()
    assert m.start() is True


def test_manager_start_no_command():
    m = ServiceManager(ServiceDef(name="svc", builtin=False, command=""))
    assert m.start() is False
    assert m.instance.state == "failed"
    assert "no command defined" in m.instance.log[-1]


def test_manager_start_real_process():
    m = ServiceManager(ServiceDef(
        name="worker",
        builtin=False,
        command=f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'",
    ))
    assert m.start() is True
    assert m.instance.state == "running"
    assert m.instance.pid > 0
    assert m.is_alive is True
    assert m.stop() is True
    assert m.instance.state == "stopped"
    assert m.instance.pid == 0
    assert m.is_alive is False


def test_manager_start_raises():
    m = ServiceManager(ServiceDef(
        name="worker",
        builtin=False,
        command="nonexistent-command-xyz --flag",
    ))
    assert m.start() is False
    assert m.instance.state == "failed"
    assert "start failed" in m.instance.log[-1]


def test_wait_until_healthy_no_check():
    m = ServiceManager(ServiceDef(name="svc", health_check=""))
    assert m.wait_until_healthy() is True


def test_wait_until_healthy_success():
    m = ServiceManager(ServiceDef(
        name="svc",
        health_check=f"{shlex.quote(sys.executable)} -c 'pass'",
    ))
    m.instance.state = "starting"
    assert m.wait_until_healthy() is True
    assert m.instance.state == "running"


def test_wait_until_healthy_failure():
    m = ServiceManager(ServiceDef(
        name="svc",
        health_check="nonexistent-health-cmd",
    ))
    m.instance.state = "starting"
    assert m.wait_until_healthy() is False


def test_stop_when_not_active_returns_true():
    m = ServiceManager(ServiceDef(name="svc", builtin=True))
    assert m.instance.state == "stopped"
    assert m.stop() is True


def test_stop_running_builtin():
    m = ServiceManager(ServiceDef(name="svc", builtin=True))
    m.start()
    m.instance.state = "running"
    assert m.stop() is True
    assert m.instance.state == "stopped"


def test_stop_kill_fallback(monkeypatch):
    m = ServiceManager(ServiceDef(name="svc", builtin=False, command=""))
    fake = RaisingKillProc()
    fake.pid = 123
    m.instance.process = fake
    m.instance.state = "running"
    monkeypatch.setattr(init_mod.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(init_mod.os, "getpgid", lambda pid: 456)
    assert m.stop() is True
    assert m.instance.state == "stopped"


def test_stop_no_killpg_uses_terminate(monkeypatch):
    m = ServiceManager(ServiceDef(name="svc", builtin=False, command=""))
    fake = FakeProc()
    fake.pid = 123
    m.instance.process = fake
    m.instance.state = "running"
    monkeypatch.delattr(init_mod.os, "killpg")
    monkeypatch.delattr(init_mod.os, "setsid")
    assert m.stop() is True
    assert fake.rc == 0 or fake.rc is not None
    assert m.instance.state == "stopped"


def test_wait_proc_wait_raising():
    m = ServiceManager(ServiceDef(name="svc", builtin=False, command=""))
    m.instance.process = RaisingWaitProc()
    m._stop_requested = True
    m._wait()
    assert m.instance.process is None


def test_restart():
    m = ServiceManager(ServiceDef(name="svc", builtin=True))
    m.start()
    assert m.restart() is True
    assert m.instance.state == "running"


def test_is_alive_builtin_running():
    m = ServiceManager(ServiceDef(name="svc", builtin=True))
    m.start()
    assert m.is_alive is True
    m.stop()
    assert m.is_alive is False


def test_wait_with_no_process():
    m = ServiceManager(ServiceDef(name="svc"))
    m.instance.process = None
    m._wait()


def test_wait_requested_stop():
    m = ServiceManager(ServiceDef(name="svc", builtin=False, command=""))
    fake = FakeProc()
    m.instance.process = fake
    m._stop_requested = True
    m._wait()
    assert fake.called_wait is True
    assert m.instance.process is None


def test_wait_respawn_path():
    m = ServiceManager(ServiceDef(
        name="svc",
        builtin=False,
        command="",
        respawn=True,
        max_respawns=2,
        respawn_delay=0.0,
    ))
    m.instance.process = FakeProc()
    m._wait()
    assert m.instance.respawn_count == 1
    assert any("respawning" in line for line in m.instance.log)


def test_wait_crashed_max_respawns():
    m = ServiceManager(ServiceDef(
        name="svc",
        builtin=False,
        command="",
        respawn=False,
        max_respawns=1,
    ))
    m.instance.process = FakeProc()
    m.instance.respawn_count = 1
    m._wait()
    assert m.instance.state == "crashed"
    assert "max respawns reached" in m.instance.log[-1]


def test_wait_crashed_process_exited():
    m = ServiceManager(ServiceDef(
        name="svc",
        builtin=False,
        command="",
        respawn=False,
        max_respawns=3,
    ))
    m.instance.process = FakeProc()
    m.instance.respawn_count = 0
    m._wait()
    assert m.instance.state == "crashed"
    assert "process exited" in m.instance.log[-1]


def test_status_line():
    m = ServiceManager(ServiceDef(name="svc", builtin=True))
    m.start()
    line = m.status_line(max_name=10)
    assert "svc" in line
    assert "running" in line


def test_load_definitions_from_dir(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text(json.dumps({"command": "echo hi", "runlevel": 4}))
    (tmp_path / "b.json").write_text("not valid json {")
    (tmp_path / "kernel.json").write_text(json.dumps({"command": "ignored"}))
    monkeypatch.setattr(init_mod, "SERVICES_DIR", tmp_path)
    init = InitSystem()
    assert init.get_manager("a") is not None
    assert init.get_manager("b") is None
    assert init.get_manager("kernel").defn.builtin is True


def _builtin_only_init(monkeypatch, tmp_path):
    monkeypatch.setattr(init_mod, "SERVICES_DIR", tmp_path)
    return InitSystem()


def test_boot_basic(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    output = init.boot(target_runlevel=3)
    assert "kernel" in output
    assert "agent-orchestrator" in output
    assert "knowledge-worker" in output
    assert "Boot complete" in output
    assert init.runlevel == 3
    assert init.uptime > 0.0
    assert init._boot_complete is True


def test_boot_dependency_started(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    init._managers["helper"] = ServiceManager(ServiceDef(name="helper", builtin=True, runlevel=2))
    init._managers["worker"] = ServiceManager(ServiceDef(
        name="worker", builtin=True, runlevel=2, deps=["helper"]))
    output = init.boot(target_runlevel=2)
    assert init.get_manager("worker").instance.state == "running"
    assert "worker" in output


def test_boot_dependency_failure(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    init._managers["bad-dep"] = ServiceManager(ServiceDef(name="bad-dep", builtin=False, command=""))
    init._managers["custom"] = ServiceManager(ServiceDef(
        name="custom", builtin=True, runlevel=2, deps=["bad-dep"]))
    output = init.boot(target_runlevel=2)
    assert "dependency bad-dep failed" in output
    assert "✗ dependency failed" in output
    assert init.get_manager("custom").instance.state != "running"


def test_boot_health_timeout(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    init._managers["flaky"] = ServiceManager(ServiceDef(
        name="flaky",
        builtin=True,
        runlevel=2,
        timeout=0.01,
        health_check="nonexistent-health-cmd",
    ))
    output = init.boot(target_runlevel=2)
    assert "flaky" in output
    assert init.get_manager("flaky").instance.state in ("failed", "crashed")


def test_boot_health_break_on_failed(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    mgr = ServiceManager(ServiceDef(
        name="flaky",
        builtin=True,
        runlevel=2,
        timeout=10,
        health_check="unused",
    ))
    init._managers["flaky"] = mgr

    def _fail_health():
        mgr.instance.state = "failed"
        return False

    mgr.wait_until_healthy = _fail_health
    output = init.boot(target_runlevel=2)
    assert "flaky" in output
    assert mgr.instance.state == "failed"


def test_boot_health_success(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    init._managers["healthy"] = ServiceManager(ServiceDef(
        name="healthy",
        builtin=True,
        runlevel=2,
        timeout=5,
        health_check=f"{shlex.quote(sys.executable)} -c 'pass'",
    ))
    output = init.boot(target_runlevel=2)
    assert "healthy" in output
    assert init.get_manager("healthy").instance.state == "running"


def test_resolve_deps_ordering(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    a = ServiceManager(ServiceDef(name="a", builtin=True, runlevel=2, deps=["b"]))
    b = ServiceManager(ServiceDef(name="b", builtin=True, runlevel=2))
    ordered = init._resolve_deps([a, b])
    assert [m.defn.name for m in ordered] == ["b", "a"]


def test_resolve_deps_skips_unknown_dep(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    a = ServiceManager(ServiceDef(name="a", builtin=True, runlevel=2, deps=["ghost"]))
    ordered = init._resolve_deps([a])
    assert [m.defn.name for m in ordered] == ["a"]


def test_shutdown(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    init.boot(target_runlevel=3)
    output = init.shutdown()
    assert "System halted." in output
    assert "stopped" in output
    assert init.runlevel == 0
    assert init._boot_complete is False
    assert init.get_manager("kernel").instance.state == "stopped"


def test_status_summary(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    init.boot(target_runlevel=1)
    summary = init.status_summary
    assert "Runlevel: 1" in summary
    assert "Services:" in summary
    assert "running" in summary


def test_service_table(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    table = init.service_table()
    assert "kernel" in table
    assert "pid=" in table
    init._managers = {}
    assert init.service_table() == "  No services defined"


def test_services_property(tmp_path, monkeypatch):
    init = _builtin_only_init(monkeypatch, tmp_path)
    assert len(init.services) >= 3


def test_singleton_reset():
    reset_init_system()
    a = get_init_system()
    b = get_init_system()
    assert a is b
    reset_init_system()
    c = get_init_system()
    assert c is not a
    reset_init_system()
