"""Tests for self_train router — start, stop, status subprocess lifecycle."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

pytest.importorskip("fastapi")

# Ensure apps/api/server is on the path for schemas.common + state import
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi.testclient import TestClient

import apps.api.server.routers.self_train as self_train_mod
import state as server_state
from tests.conftest import build_test_app


@pytest.fixture(autouse=True)
def reset_state():
    """Reset server state before each test."""
    server_state._self_train_proc = None
    yield
    server_state._self_train_proc = None


@pytest.fixture
def app():
    """Create FastAPI app with self_train router."""
    from apps.api.server.routers.self_train import SelfTrainRouter
    router_instance = SelfTrainRouter()
    return build_test_app(router_instance.router)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestGetStatus:
    def test_not_started(self, client):
        resp = client.get("/self-train/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "not_started"

    def test_running(self, client):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        server_state._self_train_proc = mock_proc

        resp = client.get("/self-train/status")
        assert resp.json()["data"]["status"] == "running"
        assert resp.json()["data"]["pid"] == 12345

    def test_exited(self, client):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        server_state._self_train_proc = mock_proc

        resp = client.get("/self-train/status")
        assert resp.json()["data"]["status"] == "exited"
        assert resp.json()["data"]["returncode"] == 0


class TestStartSelfTrain:
    def test_start(self, client):
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        with patch("subprocess.Popen", return_value=mock_proc):
            resp = client.post("/self-train/start", json={})
            assert resp.status_code == 200
            assert resp.json()["data"]["status"] == "started"
            assert resp.json()["data"]["pid"] == 99999

    def test_already_running(self, client):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 11111
        server_state._self_train_proc = mock_proc

        resp = client.post("/self-train/start", json={})
        assert resp.json()["data"]["status"] == "already_running"

    def test_start_with_model(self, client):
        mock_proc = MagicMock()
        mock_proc.pid = 22222
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            resp = client.post("/self-train/start", json={"model": "gpt2"})
            assert resp.status_code == 200
            cmd = mock_popen.call_args[0][0]
            assert "--model" in cmd
            assert "gpt2" in cmd

    def test_start_with_temperature(self, client):
        mock_proc = MagicMock()
        mock_proc.pid = 33333
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            resp = client.post("/self-train/start", json={"temperature": 0.8})
            assert resp.status_code == 200
            cmd = mock_popen.call_args[0][0]
            assert "--temperature" in cmd
            assert "0.8" in cmd

    def test_start_forever(self, client):
        mock_proc = MagicMock()
        mock_proc.pid = 44444
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            resp = client.post("/self-train/start", json={"forever": True})
            assert resp.status_code == 200
            cmd = mock_popen.call_args[0][0]
            assert "--forever" in cmd

    def test_invalid_model_name(self, client):
        resp = client.post("/self-train/start", json={"model": "rm -rf /"})
        assert resp.status_code == 422


class TestStopSelfTrain:
    def test_stop_running(self, client):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 55555
        server_state._self_train_proc = mock_proc

        resp = client.post("/self-train/stop")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "stopped"
        mock_proc.terminate.assert_called_once()
        assert server_state._self_train_proc is None

    def test_stop_not_running(self, client):
        resp = client.post("/self-train/stop")
        assert resp.json()["data"]["status"] == "not_running"

    def test_stop_already_exited(self, client):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        server_state._self_train_proc = mock_proc

        resp = client.post("/self-train/stop")
        assert resp.json()["data"]["status"] == "not_running"

    def test_stop_force_kill(self, client):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 66666
        mock_proc.terminate.side_effect = OSError("process gone")
        server_state._self_train_proc = mock_proc

        resp = client.post("/self-train/stop")
        assert resp.status_code == 503
        assert resp.json()["error"] == "process gone"
        mock_proc.kill.assert_called_once()
        assert server_state._self_train_proc is None
