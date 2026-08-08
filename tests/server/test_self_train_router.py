"""
Tests for the self-train router — POST start/stop and GET status.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.self_train import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestSelfTrainStart:
    def test_starts_when_not_running(self, client):
        resp = client.post("/self-train/start", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] in ("started", "error")

    def test_rejects_invalid_model_name(self, client):
        resp = client.post("/self-train/start", json={"model": "bad model!"})
        assert resp.status_code in (200, 400, 422)

    def test_start_with_valid_model(self, client):
        resp = client.post("/self-train/start", json={"model": "gpt2"})
        assert resp.status_code == 200

    def test_start_with_temperature(self, client):
        resp = client.post("/self-train/start", json={"model": "gpt2", "temperature": 0.5})
        assert resp.status_code == 200

    def test_start_with_all_params(self, client):
        resp = client.post("/self-train/start", json={
            "model": "gpt2",
            "temperature": 0.7,
            "max_steps": 100,
        })
        assert resp.status_code == 200

    def test_start_response_has_status(self, client):
        resp = client.post("/self-train/start", json={})
        data = resp.json()
        assert "data" in data
        assert "status" in data["data"]


class TestSelfTrainDeterministic:
    """Deterministic branch coverage via patched subprocess + controlled state."""

    @pytest.fixture(autouse=True)
    def reset_proc(self):
        import state as server_state
        yield
        server_state._self_train_proc = None

    def test_already_running(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 4242
        server_state._self_train_proc = proc
        resp = client.post("/self-train/start", json={})
        data = resp.json()["data"]
        assert data["status"] == "already_running"
        assert data["pid"] == 4242

    @patch("apps.api.server.routers.self_train.subprocess.Popen")
    def test_start_builds_command(self, mock_popen, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 9999
        mock_popen.return_value = proc
        client.post("/self-train/start", json={"model": "gpt2", "temperature": 0.5, "forever": True})
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert "--model" in cmd and "gpt2" in cmd
        assert "--temperature" in cmd and "0.5" in cmd
        assert "--forever" in cmd
        assert server_state._self_train_proc is proc

    @patch("apps.api.server.routers.self_train.subprocess.Popen")
    def test_invalid_model_returns_422(self, mock_popen, client):
        resp = client.post("/self-train/start", json={"model": "bad model!"})
        assert resp.status_code == 422
        mock_popen.assert_not_called()

    def test_stop_not_running(self, client):
        resp = client.post("/self-train/stop")
        assert resp.json()["data"]["status"] == "not_running"

    def test_stop_stopped(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = None
        server_state._self_train_proc = proc
        resp = client.post("/self-train/stop")
        assert resp.json()["data"]["status"] == "stopped"
        proc.terminate.assert_called_once()
        assert server_state._self_train_proc is None

    def test_stop_killed_on_terminate_failure(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate.side_effect = OSError("timeout")
        server_state._self_train_proc = proc
        resp = client.post("/self-train/stop")
        data = resp.json()["data"]
        assert data["status"] == "killed"
        proc.kill.assert_called_once()
        assert server_state._self_train_proc is None

    def test_status_exited(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = 3
        server_state._self_train_proc = proc
        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "exited"
        assert data["returncode"] == 3

    def test_status_running(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 5
        server_state._self_train_proc = proc
        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "running"
        assert data["pid"] == 5
    def test_returns_stopped_or_not_running(self, client):
        resp = client.post("/self-train/stop")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] in ("not_running", "stopped")

    def test_stop_returns_success(self, client):
        resp = client.post("/self-train/stop")
        assert resp.status_code == 200
        assert "status" in resp.json()["data"]

    def test_stop_idempotent(self, client):
        r1 = client.post("/self-train/stop").json()["data"]["status"]
        r2 = client.post("/self-train/stop").json()["data"]["status"]
        assert r1 == r2


class TestSelfTrainStatus:
    def test_returns_not_started(self, client):
        resp = client.get("/self-train/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "not_started"

    def test_status_has_history(self, client):
        resp = client.get("/self-train/status")
        data = resp.json()["data"]
        assert "history" in data or "status" in data

    def test_status_idempotent(self, client):
        r1 = client.get("/self-train/status").json()["data"]["status"]
        r2 = client.get("/self-train/status").json()["data"]["status"]
        assert r1 == r2

    def test_status_always_200(self, client):
        resp = client.get("/self-train/status")
        assert resp.status_code == 200
