"""
Supplementary tests for Shell Init System — subprocess lifecycle, respawn, user config, boot branches.
"""

import json
import logging
import pytest
from domains.shell import init as init_mod
from domains.shell.init import (
    ServiceDef, ServiceManager, InitSystem,
)


class FakeProc:
    def __init__(self, pid=4242, poll_result=None, wait_raises=None):
        self.pid = pid
        self._poll = poll_result
        self._wait_raises = wait_raises
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._poll

    def wait(self, timeout=None):
        if self._wait_raises:
            raise self._wait_raises
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class ThreadStub:
    created: list = []

    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        ThreadStub.created.append(self)

    def start(self):
        pass


def _patch_proc(monkeypatch, proc=None):
    ThreadStub.created = []
    monkeypatch.setattr(init_mod.subprocess, "Popen", lambda *a, **k: proc or FakeProc())
    monkeypatch.setattr(init_mod.threading, "Thread", ThreadStub)


# ── ServiceManager.start / stop / health ─────────────────────────────────


class TestServiceManagerStart:
    def test_start_launches_process(self, monkeypatch):
        fake = FakeProc(pid=777)
        _patch_proc(monkeypatch, fake)
        m = ServiceManager(ServiceDef(name="worker", command="sleep 100", builtin=False))
        assert m.start() is True
        assert m.instance.state == "running"
        assert m.instance.pid == 777
        assert len(ThreadStub.created) == 1
        assert ThreadStub.created[0].daemon is True

    def test_start_process_exception(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("spawn failed")
        monkeypatch.setattr(init_mod.subprocess, "Popen", boom)
        m = ServiceManager(ServiceDef(name="worker", command="sleep 100", builtin=False))
        assert m.start() is False
        assert m.instance.state == "failed"
        assert "start failed" in m.instance.log[-1]


class TestWaitUntilHealthy:
    def test_no_health_check(self):
        m = ServiceManager(ServiceDef(name="x", builtin=False))
        assert m.wait_until_healthy() is True

    def test_health_check_success(self, monkeypatch):
        calls = []
        monkeypatch.setattr(init_mod.subprocess, "run",
                           lambda *a, **k: calls.append(True) or None)
        m = ServiceManager(ServiceDef(name="x", builtin=False, health_check="curl health"))
        assert m.wait_until_healthy() is True
        assert m.instance.state == "running"
        assert calls

    def test_health_check_failure(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("no curl")
        monkeypatch.setattr(init_mod.subprocess, "run", boom)
        m = ServiceManager(ServiceDef(name="x", builtin=False, health_check="curl health"))
        assert m.wait_until_healthy() is False


class TestStop:
    def test_stop_not_running(self):
        m = ServiceManager(ServiceDef(name="x", builtin=False))
        assert m.stop() is True
        assert m.instance.state == "stopped"

    def test_stop_already_exited(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=0)
        _patch_proc(monkeypatch, fake)
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.stop() is True
        assert m.instance.state == "stopped"
        assert fake.terminated is False

    def test_stop_killpg(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=None)
        _patch_proc(monkeypatch, fake)
        monkeypatch.setattr(init_mod.os, "getpgid", lambda pid: 1234)
        killed = []
        monkeypatch.setattr(init_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.stop() is True
        assert killed == [(1234, 15)]

    def test_stop_kill_on_wait_error(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=None, wait_raises=TimeoutError("wait"))
        _patch_proc(monkeypatch, fake)
        monkeypatch.setattr(init_mod.os, "getpgid", lambda pid: 1234)
        monkeypatch.setattr(init_mod.os, "killpg", lambda pgid, sig: None)
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.stop() is True
        assert fake.killed is True

    def test_stop_kill_raises_again(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=None, wait_raises=TimeoutError("wait"))
        def kill_boom():
            raise OSError("kill failed")
        fake.kill = kill_boom
        _patch_proc(monkeypatch, fake)
        monkeypatch.setattr(init_mod.os, "getpgid", lambda pid: 1234)
        monkeypatch.setattr(init_mod.os, "killpg", lambda pgid, sig: None)
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.stop() is True

    def test_stop_terminate_fallback(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=None)
        _patch_proc(monkeypatch, fake)
        monkeypatch.delattr(init_mod.os, "killpg")
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.stop() is True
        assert fake.terminated is True


class TestIsAlive:
    def test_is_alive_running_proc(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=None)
        _patch_proc(monkeypatch, fake)
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.is_alive is True

    def test_is_alive_exited_proc(self, monkeypatch):
        fake = FakeProc(pid=5, poll_result=0)
        _patch_proc(monkeypatch, fake)
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False))
        m.start()
        assert m.is_alive is False


# ── ServiceManager._wait / respawn ───────────────────────────────────────


class TestWait:
    def test_wait_no_proc(self):
        m = ServiceManager(ServiceDef(name="x", builtin=False))
        m._wait()

    def test_wait_stop_requested(self):
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False, respawn=True))
        m.instance.process = FakeProc(pid=5)
        m._stop_requested = True
        m._wait()
        assert m.instance.process is None
        assert m.instance.pid == 0

    def test_wait_respawn(self, monkeypatch):
        monkeypatch.setattr(init_mod.time, "sleep", lambda s: None)
        _patch_proc(monkeypatch)
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False,
                                      respawn=True, max_respawns=3))
        m.instance.process = FakeProc(pid=5)
        m._wait()
        assert m.instance.respawn_count == 1
        assert any("respawn" in line for line in m.instance.log)

    def test_wait_max_respawns(self, monkeypatch):
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False,
                                      respawn=True, max_respawns=1))
        m.instance.process = FakeProc(pid=5)
        m.instance.respawn_count = 1
        m._wait()
        assert m.instance.state == "crashed"
        assert any("max respawns reached" in line for line in m.instance.log)

    def test_wait_normal_exit(self):
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False, respawn=False))
        m.instance.process = FakeProc(pid=5)
        m._wait()
        assert m.instance.state == "crashed"
        assert any("process exited" in line for line in m.instance.log)

    def test_wait_proc_wait_raises(self):
        fake = FakeProc(pid=5)
        fake.wait = lambda timeout=None: (_ for _ in ()).throw(OSError("wait"))
        m = ServiceManager(ServiceDef(name="x", command="sleep 1", builtin=False, respawn=False))
        m.instance.process = fake
        m._wait()
        assert m.instance.state == "crashed"


# ── InitSystem._load_definitions user config ─────────────────────────────


class TestLoadDefinitions:
    def test_load_user_service(self, tmp_path, monkeypatch):
        svc = {"command": "echo hi", "runlevel": 3, "builtin": False, "deps": []}
        (tmp_path / "my-svc.json").write_text(json.dumps(svc))
        monkeypatch.setattr(init_mod, "SERVICES_DIR", tmp_path)
        system = InitSystem()
        mgr = system.get_manager("my-svc")
        assert mgr is not None
        assert mgr.defn.command == "echo hi"
        assert mgr.defn.runlevel == 3

    def test_load_user_does_not_override_builtin(self, tmp_path, monkeypatch):
        svc = {"command": "evil", "builtin": False}
        (tmp_path / "kernel.json").write_text(json.dumps(svc))
        monkeypatch.setattr(init_mod, "SERVICES_DIR", tmp_path)
        system = InitSystem()
        assert system.get_manager("kernel").defn.command == ""

    def test_load_bad_json_warns(self, tmp_path, monkeypatch, caplog):
        (tmp_path / "bad.json").write_text("{not json")
        monkeypatch.setattr(init_mod, "SERVICES_DIR", tmp_path)
        with caplog.at_level(logging.WARNING):
            InitSystem()
        assert any("Failed to load service bad.json" in r.message for r in caplog.records)


# ── InitSystem.boot branches ─────────────────────────────────────────────


def _bare_init():
    system = InitSystem()
    system._managers = {}
    return system


class TestBootBranches:
    def test_boot_dep_failure(self, monkeypatch):
        db = ServiceManager(ServiceDef(name="db", runlevel=1, command="", builtin=False))
        web = ServiceManager(ServiceDef(name="web", runlevel=1, deps=["db"], builtin=True))
        system = _bare_init()
        system._managers = {"db": db, "web": web}
        out = system.boot(target_runlevel=1)
        assert "dependency db failed" in out
        assert "✗ dependency failed" in out
        assert db.instance.state == "failed"

    def test_boot_missing_dep_is_satisfied(self, monkeypatch):
        monkeypatch.setattr(init_mod.threading, "Thread", ThreadStub)
        web = ServiceManager(ServiceDef(name="web", runlevel=1, deps=["missing"], builtin=True))
        system = _bare_init()
        system._managers = {"web": web}
        out = system.boot(target_runlevel=1)
        assert "✓" in out
        assert web.instance.state == "running"

    def test_boot_dep_already_running(self, monkeypatch):
        monkeypatch.setattr(init_mod.threading, "Thread", ThreadStub)
        db = ServiceManager(ServiceDef(name="db", runlevel=1, builtin=True))
        db.start()
        web = ServiceManager(ServiceDef(name="web", runlevel=1, deps=["db"], builtin=True))
        system = _bare_init()
        system._managers = {"db": db, "web": web}
        out = system.boot(target_runlevel=1)
        assert "web" in out

    def test_boot_health_check_healthy(self, monkeypatch):
        monkeypatch.setattr(init_mod.time, "sleep", lambda s: None)
        _patch_proc(monkeypatch)
        monkeypatch.setattr(init_mod.subprocess, "run", lambda *a, **k: None)
        svc = ServiceManager(ServiceDef(name="db", runlevel=1, command="sleep 1", builtin=False,
                                        health_check="echo ok", description="db service"))
        system = _bare_init()
        system._managers = {"db": svc}
        out = system.boot(target_runlevel=1)
        assert "✓ db service" in out
        assert svc.instance.state == "running"

    def test_boot_health_check_timeout(self, monkeypatch):
        monkeypatch.setattr(init_mod.time, "sleep", lambda s: None)
        _patch_proc(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("unhealthy")
        monkeypatch.setattr(init_mod.subprocess, "run", boom)
        svc = ServiceManager(ServiceDef(name="db", runlevel=1, command="sleep 1", builtin=False,
                                        health_check="echo ok", timeout=0.01))
        system = _bare_init()
        system._managers = {"db": svc}
        out = system.boot(target_runlevel=1)
        assert svc.instance.state == "failed"

    def test_boot_health_check_breaks_on_failed(self, monkeypatch):
        monkeypatch.setattr(init_mod.time, "sleep", lambda s: None)
        _patch_proc(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("unhealthy")
        monkeypatch.setattr(init_mod.subprocess, "run", boom)
        svc = ServiceManager(ServiceDef(name="db", runlevel=1, command="sleep 1", builtin=False,
                                        health_check="echo ok", timeout=10.0))
        svc.wait_until_healthy = lambda: (setattr(svc.instance, "state", "failed") or False)
        system = _bare_init()
        system._managers = {"db": svc}
        out = system.boot(target_runlevel=1)
        assert svc.instance.state == "failed"


# ── Properties / empty states ────────────────────────────────────────────


class TestMisc:
    def test_runlevel_property(self, monkeypatch):
        system = InitSystem()
        assert system.runlevel == 0
        system.boot(target_runlevel=1)
        assert system.runlevel == 1

    def test_service_table_empty(self):
        system = InitSystem()
        system._managers = {}
        assert system.service_table() == "  No services defined"
