"""Tests for domains/shell/runtime.py — Resource, _probe_api, APIServerProcess, DaitRuntime."""

import signal
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from domains.shell.runtime import (
    APIServerProcess,
    DaitRuntime,
    Resource,
    _probe_api,
)
from domains.shared import find_repo_root


class _FakeProc:
    def __init__(self, pid=9999, stderr=None):
        self.pid = pid
        self.stderr = stderr

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


class TestResource:
    def test_size_str_units(self):
        assert Resource("a", "file", "/x", 512).size_str == "512B"
        assert Resource("a", "file", "/x", 1024).size_str == "1.0K"
        assert Resource("a", "file", "/x", 1048576).size_str == "1.0M"
        assert Resource("a", "file", "/x", 1073741824).size_str == "1.0G"

    def test_defaults(self):
        r = Resource("m", "model", "/models/m")
        assert r.size_bytes == 0
        assert r.metadata == {}


class TestProbeApi:
    def test_success(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"status":"ok","model_loaded":true,"model_id":"gpt2","engine_type":"slo"}'

        with patch("urllib.request.urlopen", return_value=_Resp()):
            r = _probe_api("http://localhost:9999")
        assert r["available"] is True
        assert r["status"] == "ok"
        assert r["model_loaded"] is True
        assert r["model_id"] == "gpt2"
        assert r["engine_type"] == "slo"

    def test_failure(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            r = _probe_api("http://localhost:9999")
        assert r["available"] is False
        assert "refused" in r["error"]


class TestAPIServerProcessStart:
    def test_start_ready(self):
        """Phase 1: server already healthy — returns 'connected', no Popen."""
        with patch("domains.shell.runtime._shared_proc", None), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime._probe_api",
                   return_value={"available": True, "model_id": "gpt2"}):
            r = APIServerProcess("http://x").start(timeout=10)
        assert r["ok"] is True
        assert "connected (gpt2)" in r["message"]

    def test_start_already_running(self):
        fake = _FakeProc(stderr=None)
        with patch("domains.shell.runtime._shared_proc", fake):
            r = APIServerProcess().start()
        assert r == {"ok": True, "message": "already running"}

    def test_start_launch_error(self):
        with patch("domains.shell.runtime._shared_proc", None), \
             patch("domains.shell.runtime._probe_api", return_value={"available": False}), \
             patch("socket.create_connection", side_effect=ConnectionRefusedError), \
             patch("domains.shell.runtime.subprocess.Popen",
                   side_effect=OSError("boom")):
            r = APIServerProcess().start()
        assert r["ok"] is False
        assert "Failed to launch" in r["error"]

    def test_start_timeout(self):
        proc = _FakeProc(stderr=None)
        with patch("domains.shell.runtime._shared_proc", None), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime.subprocess.Popen", return_value=proc), \
             patch("domains.shell.runtime._probe_api", return_value={"available": False}), \
             patch("domains.shell.runtime.time.time",
                   side_effect=[0.0, 0.0, 0.5, 1.0, 1.5, 2.0]), \
             patch("domains.shell.runtime.time.sleep", lambda s: None), \
             patch("socket.create_connection", side_effect=ConnectionRefusedError), \
             patch("sys.stdout", MagicMock()):
            r = APIServerProcess("http://x").start(timeout=1.0)
        assert r["ok"] is False
        assert "Timed out" in r["error"]


class TestAPIServerProcessStop:
    def test_stop_not_running(self):
        with patch("domains.shell.runtime._shared_proc", None), \
             patch("domains.shell.runtime._shared_started_at", 0.0):
            r = APIServerProcess().stop()
        assert r == {"ok": True, "message": "not running"}

    def test_stop_killpg(self):
        proc = _FakeProc(pid=4321)
        with patch("domains.shell.runtime._shared_proc", proc), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime.os.killpg") as killpg, \
             patch("domains.shell.runtime.os.getpgid", return_value=111):
            r = APIServerProcess().stop()
        assert r["ok"] is True
        killpg.assert_any_call(111, signal.SIGTERM)

    def test_stop_timeout_sigkill(self):
        proc = _FakeProc(pid=4321)
        proc.wait = Mock(side_effect=[subprocess.TimeoutExpired("cmd", 10),
                                      RuntimeError("died"), None])
        with patch("domains.shell.runtime._shared_proc", proc), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime.os.killpg") as killpg, \
             patch("domains.shell.runtime.os.getpgid", return_value=111):
            r = APIServerProcess().stop()
        assert r["ok"] is True
        killpg.assert_any_call(111, signal.SIGKILL)

    def test_stop_no_killpg(self):
        class _NoKillpg:
            environ = {}

        proc = _FakeProc(pid=5)
        proc.terminate = Mock()
        proc.wait = Mock(return_value=0)
        with patch("domains.shell.runtime._shared_proc", proc), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime.os", _NoKillpg()):
            r = APIServerProcess().stop()
        assert r["ok"] is True
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()

    def test_stop_no_killpg_timeout(self):
        class _NoKillpg:
            environ = {}

        proc = _FakeProc(pid=5)
        proc.wait = Mock(side_effect=[subprocess.TimeoutExpired("cmd", 10), None])
        proc.kill = Mock()
        with patch("domains.shell.runtime._shared_proc", proc), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime.os", _NoKillpg()):
            r = APIServerProcess().stop()
        assert r["ok"] is True
        proc.kill.assert_called_once()

    def test_stop_wait_exception_ignored(self):
        proc = _FakeProc(pid=5)
        proc.wait = Mock(side_effect=RuntimeError("boom"))
        with patch("domains.shell.runtime._shared_proc", proc), \
             patch("domains.shell.runtime._shared_started_at", 0.0), \
             patch("domains.shell.runtime.os.killpg"), \
             patch("domains.shell.runtime.os.getpgid", return_value=1):
            r = APIServerProcess().stop()
        assert r["ok"] is True


class TestAPIServerProcessMisc:
    def test_is_running_true(self):
        proc = _FakeProc()
        with patch("domains.shell.runtime._shared_proc", proc):
            assert APIServerProcess().is_running is True

    def test_is_running_false_when_exited(self):
        proc = _FakeProc()
        proc.poll = Mock(return_value=0)
        with patch("domains.shell.runtime._shared_proc", proc):
            assert APIServerProcess().is_running is False

    def test_is_running_false_when_none(self):
        with patch("domains.shell.runtime._shared_proc", None), \
             patch("domains.shell.runtime._probe_api", return_value={"available": False}):
            assert APIServerProcess().is_running is False

    def test_is_running_probes_http_when_no_proc(self):
        with patch("domains.shell.runtime._shared_proc", None), \
             patch("domains.shell.runtime._probe_api", return_value={"available": True}):
            assert APIServerProcess().is_running is True

    def test_repr(self):
        with patch("domains.shell.runtime._shared_proc", None):
            r = repr(APIServerProcess("http://x"))
        assert "APIServerProcess" in r
        assert "http://x" in r


class TestFindRepoRoot:
    class _Result:
        def __init__(self, exists=False, is_dir=False):
            self._e = exists
            self._d = is_dir

        def exists(self):
            return self._e

        def is_dir(self):
            return self._d

    class _Parent:
        def __init__(self, name, py=False, apps=False, pkgs=False):
            self.name = name
            self._py = py
            self._a = apps
            self._p = pkgs

        def __truediv__(self, other):
            if other == "pyproject.toml":
                return TestFindRepoRoot._Result(self._py)
            if other == "setup.py":
                return TestFindRepoRoot._Result(False)
            if other == "apps":
                return TestFindRepoRoot._Result(False, self._a)
            if other == "packages":
                return TestFindRepoRoot._Result(False, self._p)
            return TestFindRepoRoot._Result(False)

        def __str__(self):
            return self.name

    class _FakePath:
        def __init__(self, parents):
            self.parents = parents

        def resolve(self):
            return self

    def _patch(self, parents):
        return patch("domains.shared.utils.Path",
                     return_value=self._FakePath(parents))

    def test_finds_pyproject_with_apps(self):
        parents = [self._Parent("A"), self._Parent("B", py=True, apps=True)]
        with self._patch(parents):
            result = find_repo_root()
        assert result.name == "B"

    def test_finds_apps_packages_marker(self):
        parents = [self._Parent("A"),
                   self._Parent("B", apps=True, pkgs=True)]
        with self._patch(parents):
            result = find_repo_root()
        assert result.name == "B"

    def test_apps_packages_beats_pyproject(self):
        parents = [self._Parent("A", apps=True, pkgs=True),
                   self._Parent("B", py=True, apps=True)]
        with self._patch(parents):
            result = find_repo_root()
        assert result.name == "A"

    def test_pyproject_without_apps_skipped(self):
        parents = [self._Parent("A"), self._Parent("B", py=True)]
        with self._patch(parents):
            result = find_repo_root()
        assert result.name == "B"  # falls through to fallback parents[4]

    def test_fallback_to_deep_parent(self):
        parents = [self._Parent("A"), self._Parent("B"),
                   self._Parent("C"), self._Parent("D"),
                   self._Parent("E")]
        with self._patch(parents):
            result = find_repo_root()
        assert result.name == "E"


class TestDaitRuntimeStatus:
    def _runtime(self):
        rt = DaitRuntime()
        rt._api = MagicMock()
        rt._api.status.return_value = {
            "available": True,
            "model_loaded": True,
            "model_id": "gpt2",
        }
        return rt

    def _fake_psutil(self):
        fake = types.ModuleType("psutil")

        class _Proc:
            def memory_info(self):
                return types.SimpleNamespace(rss=1048576, vms=2097152)

        fake.Process = _Proc
        return fake

    def test_status_summary_with_psutil(self):
        rt = self._runtime()
        rt._boot_complete = True
        rt._init = types.SimpleNamespace(status_summary="INIT-SUMMARY")
        with patch.dict("sys.modules", {"psutil": self._fake_psutil()}):
            s = rt.status_summary
        assert "Kernel uptime:" in s
        assert "API: ✓ (gpt2)" in s
        assert "rss=1M vms=2M" in s
        assert "INIT-SUMMARY" in s

    def test_status_summary_without_psutil(self):
        rt = self._runtime()
        rt._boot_complete = False
        with patch.dict("sys.modules", {"psutil": None}):
            s = rt.status_summary
        assert "psutil not available" in s
        assert "Soul:" in s

    def test_init_system_property(self):
        rt = DaitRuntime()
        marker = object()
        rt._init = marker
        assert rt.init_system is marker

    def test_vfs_property(self):
        rt = DaitRuntime()
        marker = object()
        rt._vfs = marker
        assert rt.vfs is marker

    def test_devices_property(self):
        rt = DaitRuntime()
        marker = object()
        rt._devices = marker
        assert rt.devices is marker


class TestDaitRuntimeBoot:
    def test_boot_completes(self):
        rt = DaitRuntime()
        log, status = rt.boot()
        assert isinstance(log, str)
        assert isinstance(status, dict)
        assert rt._boot_complete
        assert rt.init_system is not None
        assert rt.devices is not None
        assert rt.vfs is not None

    def test_shutdown_flips_flag(self):
        rt = DaitRuntime()
        rt.boot()
        log = rt.shutdown()
        assert isinstance(log, str)
        assert not rt._boot_complete

    def test_api_property(self):
        rt = DaitRuntime()
        assert isinstance(rt.api, APIServerProcess)

    def test_api_status_sets_model_flags(self):
        rt = DaitRuntime()
        rt._api = MagicMock()
        rt._api.status.return_value = {
            "available": True,
            "model_loaded": True,
            "model_id": "gpt2",
        }
        result = rt.api_status
        assert result["available"]
        assert rt._model_loaded
        assert rt._model_name == "gpt2"


class TestAPIServerProcessStatusShared:
    def test_status_reports_running_and_uptime(self):
        import time

        import domains.shell.runtime as runtime_mod

        probe = {"available": True, "status": "success", "model_loaded": False}
        saved_proc = runtime_mod._shared_proc
        saved_started = runtime_mod._shared_started_at
        try:
            runtime_mod._shared_proc = object()
            runtime_mod._shared_started_at = time.time() - 100
            api = APIServerProcess(api_url="http://unused:1")
            with patch.object(runtime_mod, "_probe_api", return_value=probe):
                result = api.status()
        finally:
            runtime_mod._shared_proc = saved_proc
            runtime_mod._shared_started_at = saved_started
        assert result["running"] is True
        assert result["uptime"] > 99.0
        assert result["available"] is True
