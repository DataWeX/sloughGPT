"""Tests for shell.runtime — Resource, APIServerProcess, DaitRuntime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from domains.shell.runtime import Resource, APIServerProcess, _probe_api, _default_api_url


# ── Resource ────────────────────────────────────────────────────────────────


class TestResource:

    def test_init(self):
        r = Resource(name="test", kind="model", path="/tmp/test.soul")
        assert r.name == "test"
        assert r.kind == "model"
        assert r.size_bytes == 0
        assert r.metadata == {}

    def test_size_str_bytes(self):
        r = Resource(name="t", kind="m", path="/t", size_bytes=500)
        assert r.size_str == "500B"

    def test_size_str_kb(self):
        r = Resource(name="t", kind="m", path="/t", size_bytes=2048)
        assert r.size_str == "2.0K"

    def test_size_str_mb(self):
        r = Resource(name="t", kind="m", path="/t", size_bytes=2_097_152)
        assert r.size_str == "2.0M"

    def test_size_str_gb(self):
        r = Resource(name="t", kind="m", path="/t", size_bytes=2_147_483_648)
        assert r.size_str == "2.0G"


# ── _default_api_url ───────────────────────────────────────────────────────


class TestDefaultApiUrl:

    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            url = _default_api_url()
            assert "localhost" in url

    def test_from_env(self):
        with patch.dict("os.environ", {"API_BASE": "http://custom:9000"}):
            assert _default_api_url() == "http://custom:9000"


# ── _probe_api ──────────────────────────────────────────────────────────────


class TestProbeApi:

    def test_available(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "status": "success",
            "data": {"status": "ready", "model_loaded": True, "model_type": "slonet"},
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _probe_api("http://localhost:8000")
        assert result["available"] is True
        assert result["model_loaded"] is True
        assert result["model_id"] == "slonet"

    def test_unavailable(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = _probe_api("http://localhost:8000")
        assert result["available"] is False
        assert "error" in result


# ── APIServerProcess ───────────────────────────────────────────────────────


class TestAPIServerProcess:

    def setup_method(self):
        import domains.shell.runtime as mod
        mod._shared_proc = None
        mod._shared_started_at = 0.0

    def test_init(self):
        api = APIServerProcess()
        assert api._port == 8000

    def test_init_custom_port(self):
        api = APIServerProcess("http://localhost:9000")
        assert api._port == 9000

    def test_status_not_running(self):
        api = APIServerProcess()
        with patch("domains.shell.runtime._probe_api", return_value={"available": False}):
            result = api.status()
        assert result["running"] is False
        assert result["available"] is False

    def test_stop_not_running(self):
        api = APIServerProcess()
        result = api.stop()
        assert result["ok"] is True
        assert "not running" in result["message"]

    def test_repr(self):
        api = APIServerProcess()
        r = repr(api)
        assert "APIServerProcess" in r

    def test_is_running_false(self):
        api = APIServerProcess()
        with patch("domains.shell.runtime._probe_api", return_value={"available": False}):
            assert api.is_running is False

    def test_start_already_healthy(self):
        api = APIServerProcess()
        with patch("domains.shell.runtime._probe_api", return_value={"available": True, "model_id": "slonet"}):
            result = api.start()
        assert result["ok"] is True
        assert "connected" in result["message"]

    def test_start_already_spawned(self):
        import domains.shell.runtime as mod
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mod._shared_proc = mock_proc
        mod._shared_started_at = time.time()
        api = APIServerProcess()
        result = api.start()
        assert result["ok"] is True
        assert "already running" in result["message"]
        mod._shared_proc = None

    def test_start_stale_process(self):
        import domains.shell.runtime as mod
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        mod._shared_proc = mock_proc
        mod._shared_started_at = time.time() - 100
        api = APIServerProcess()
        with patch("domains.shell.runtime._probe_api", return_value={"available": True}):
            result = api.start()
        assert result["ok"] is True
        mod._shared_proc = None


# ── DaitRuntime ─────────────────────────────────────────────────────────────


class TestDaitRuntime:

    def setup_method(self):
        import domains.shell.runtime as mod
        mod._shared_proc = None
        mod._shared_started_at = 0.0

    def test_init(self):
        with patch("domains.shell.kernel.Kernel"):
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            assert rt._model_loaded is False
            assert rt._boot_complete is False

    def test_api_property(self):
        with patch("domains.shell.kernel.Kernel"):
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            assert isinstance(rt.api, APIServerProcess)

    def test_api_status(self):
        with patch("domains.shell.kernel.Kernel"):
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            with patch.object(rt._api, "status", return_value={"available": True, "model_loaded": True}):
                result = rt.api_status
            assert result["available"] is True
            assert rt._model_loaded is True

    def test_status_summary(self):
        with patch("domains.shell.kernel.Kernel") as MockKernel:
            mock_kernel = MockKernel.return_value
            mock_kernel.uptime = 100
            mock_kernel.list_processes.return_value = []
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            with patch.object(rt._api, "status", return_value={"available": False}):
                summary = rt.status_summary
            assert "Kernel uptime" in summary

    def test_init_system_property(self):
        with patch("domains.shell.kernel.Kernel"):
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            assert rt.init_system is None

    def test_devices_property(self):
        with patch("domains.shell.kernel.Kernel"):
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            assert rt.devices is None

    def test_vfs_property(self):
        with patch("domains.shell.kernel.Kernel"):
            from domains.shell.runtime import DaitRuntime
            rt = DaitRuntime()
            assert rt.vfs is None
