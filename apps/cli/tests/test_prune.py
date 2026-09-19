"""Tests for apps/cli/src/utils/prune.py — stale server garbage collection."""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.prune import (
    _health_ok,
    _is_own_server,
    _proc_port,
    find_stale_servers,
    prune_stale_servers,
)


class TestIsOwnServer:
    def test_cli_serve(self):
        assert _is_own_server(["python", "cli.py", "serve", "--web"]) is True

    def test_uvicorn_main(self):
        assert _is_own_server(["uvicorn", "apps.api.server.main:app", "--port", "8000"]) is True

    def test_other_python(self):
        assert _is_own_server(["python", "-m", "pytest", "tests/"]) is False

    def test_unrelated_uvicorn(self):
        assert _is_own_server(["uvicorn", "other.app:app"]) is False

    def test_empty(self):
        assert _is_own_server([]) is False


class TestProcPort:
    def test_long_form(self):
        assert _proc_port(["uvicorn", "main:app", "--port", "8002"]) == 8002

    def test_equals_form(self):
        assert _proc_port(["uvicorn", "main:app", "--port=8002"]) == 8002

    def test_default(self):
        assert _proc_port(["python", "cli.py", "serve"]) == 8000

    def test_garbage(self):
        assert _proc_port(["uvicorn", "main:app", "--port", "abc"]) == 8000


def _fake_psutil(procs):
    """Build a fake psutil module; procs are dicts(pid, cmdline, create_time)."""
    mod = types.ModuleType("psutil")

    class _NoSuchProcess(Exception):
        pass

    class _AccessDenied(Exception):
        pass

    class _ZombieProcess(Exception):
        pass

    class _TimeoutExpired(Exception):
        pass

    live = {}

    class FakeProc:
        def __init__(self, info):
            self.info = dict(info)
            self.terminated = False
            self.killed = False
            live[self.info["pid"]] = self

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return None

        def parent(self):
            return None

    for p in procs:
        FakeProc(p)

    mod.NoSuchProcess = _NoSuchProcess
    mod.AccessDenied = _AccessDenied
    mod.ZombieProcess = _ZombieProcess
    mod.TimeoutExpired = _TimeoutExpired
    def _by_pid(pid=None):
        if pid in live:
            return live[pid]
        return FakeProc({"pid": pid or os.getpid(), "cmdline": ["pytest"], "create_time": 0})

    mod.Process = _by_pid
    mod.process_iter = lambda attrs=None: list(live.values())
    mod._live = live
    return mod


class TestFindStale:
    def test_skips_healthy(self, monkeypatch):
        import utils.prune as prune_mod

        fake = _fake_psutil(
            [{"pid": 111, "cmdline": ["uvicorn", "apps.api.server.main:app"], "create_time": 0}]
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(prune_mod, "_health_ok", lambda host, port, timeout=3: True)
        assert find_stale_servers() == []

    def test_reports_unhealthy(self, monkeypatch):
        import utils.prune as prune_mod

        fake = _fake_psutil(
            [{"pid": 222, "cmdline": ["uvicorn", "apps.api.server.main:app"], "create_time": 0}]
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(prune_mod, "_health_ok", lambda host, port, timeout=3: False)
        found = find_stale_servers()
        assert [e["pid"] for e in found] == [222]

    def test_never_self(self, monkeypatch):
        import utils.prune as prune_mod

        me = os.getpid()
        fake = _fake_psutil(
            [{"pid": me, "cmdline": ["uvicorn", "apps.api.server.main:app"], "create_time": 0}]
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(prune_mod, "_health_ok", lambda host, port, timeout=3: False)
        assert find_stale_servers() == []

    def test_port_filter(self, monkeypatch):
        import utils.prune as prune_mod

        fake = _fake_psutil(
            [
                {"pid": 333, "cmdline": ["uvicorn", "main:app", "--port", "8001"], "create_time": 0},
                {"pid": 444, "cmdline": ["uvicorn", "main:app", "--port", "8002"], "create_time": 0},
            ]
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(prune_mod, "_health_ok", lambda host, port, timeout=3: False)
        assert [e["pid"] for e in find_stale_servers(port=8002)] == [444]

    def test_supervisor_with_healthy_child_is_kept(self, monkeypatch):
        import utils.prune as prune_mod

        fake = _fake_psutil(
            [
                {"pid": 777, "cmdline": ["python", "cli.py", "serve"], "create_time": 0},
                {
                    "pid": 778,
                    "cmdline": ["uvicorn", "apps.api.server.main:app", "--port", "8002"],
                    "create_time": 0,
                },
            ]
        )
        fake._live[777].children = lambda recursive=False: [fake._live[778]]
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(
            prune_mod, "_health_ok", lambda host, port, timeout=3: port == 8002
        )
        found = [e["pid"] for e in find_stale_servers()]
        assert 777 not in found
        assert 778 not in found


class TestPrune:
    def test_terminates_stale(self, monkeypatch):
        import utils.prune as prune_mod

        fake = _fake_psutil(
            [{"pid": 555, "cmdline": ["uvicorn", "apps.api.server.main:app"], "create_time": 0}]
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(prune_mod, "_health_ok", lambda host, port, timeout=3: False)
        report = prune_stale_servers()
        assert [e["pid"] for e in report["killed"]] == [555]
        assert report["failed"] == []
        assert fake._live[555].terminated is True

    def test_sigkill_fallback(self, monkeypatch):
        import utils.prune as prune_mod

        fake = _fake_psutil(
            [{"pid": 666, "cmdline": ["uvicorn", "apps.api.server.main:app"], "create_time": 0}]
        )

        hung_once = {"n": False}

        def _hang(timeout=None):
            # Hang the SIGTERM wait only; the post-kill wait succeeds.
            if not hung_once["n"]:
                hung_once["n"] = True
                raise fake.TimeoutExpired()
            return None

        fake._live[666].wait = _hang
        monkeypatch.setitem(sys.modules, "psutil", fake)
        monkeypatch.setattr(prune_mod, "_health_ok", lambda host, port, timeout=3: False)
        report = prune_stale_servers(grace_seconds=0)
        assert [e["pid"] for e in report["killed"]] == [666]
        assert fake._live[666].killed is True

    def test_health_ok_before_check(self):
        assert _health_ok("127.0.0.1", 1, timeout=1) is False


class TestServePruneFlag:
    def test_serve_prune_exits_without_starting(self, monkeypatch):
        import types

        import utils.prune as prune_mod

        calls: list = []
        monkeypatch.setattr(
            prune_mod,
            "prune_stale_servers",
            lambda *a, **k: calls.append((a, k)) or {"killed": [], "failed": []},
        )
        from commands.dev import cmd_serve

        args = types.SimpleNamespace(
            model=None, web=False, mobile=False, prune=True,
            host="localhost", port=8000,
        )
        cmd_serve(args)
        assert len(calls) == 1

    def test_dev_prunes_before_start(self, monkeypatch):
        import types

        import utils.prune as prune_mod

        calls: list = []
        monkeypatch.setattr(
            prune_mod,
            "prune_stale_servers",
            lambda *a, **k: calls.append((a, k)) or {"killed": [], "failed": []},
        )
        from commands.dev import cmd_dev

        args = types.SimpleNamespace(
            model=None, web_port=3000, watch_web=False, port=8000,
            host="localhost", auto_download=False, prune=True,
        )
        cmd_dev(args)
        assert len(calls) == 1
