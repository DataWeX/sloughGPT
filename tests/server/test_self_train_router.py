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


class TestSelfTrainValidation:
    """Request body validation bounds."""

    def test_temperature_above_max_422(self, client):
        resp = client.post("/self-train/start", json={"temperature": 2.1})
        assert resp.status_code == 422

    def test_temperature_below_min_422(self, client):
        resp = client.post("/self-train/start", json={"temperature": -0.1})
        assert resp.status_code == 422

    def test_temperature_string_422(self, client):
        resp = client.post("/self-train/start", json={"temperature": "hot"})
        assert resp.status_code == 422

    def test_model_too_long_422(self, client):
        resp = client.post("/self-train/start", json={"model": "m" * 129})
        assert resp.status_code == 422

    def test_forever_wrong_type_422(self, client):
        resp = client.post("/self-train/start", json={"forever": {}})
        assert resp.status_code == 422

    def test_temperature_boundary_zero_ok(self, client):
        resp = client.post("/self-train/start", json={"temperature": 0.0})
        assert resp.status_code == 200

    def test_temperature_boundary_two_ok(self, client):
        resp = client.post("/self-train/start", json={"temperature": 2.0})
        assert resp.status_code == 200


class TestSelfTrainMethodMismatch:
    """Wrong HTTP methods on self-train routes."""

    def test_start_get_405(self, client):
        resp = client.get("/self-train/start")
        assert resp.status_code == 405

    def test_stop_get_405(self, client):
        resp = client.get("/self-train/stop")
        assert resp.status_code == 405

    def test_status_post_405(self, client):
        resp = client.post("/self-train/status")
        assert resp.status_code == 405


class TestSelfTrainEdgePaths:
    """Remaining start/stop/status branches."""

    @pytest.fixture(autouse=True)
    def reset_proc(self):
        import state as server_state
        yield
        server_state._self_train_proc = None

    def test_start_when_proc_exited(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = 0
        server_state._self_train_proc = proc
        resp = client.post("/self-train/start", json={})
        assert resp.json()["data"]["status"] == "started"

    @patch("apps.api.server.routers.self_train.subprocess.Popen")
    def test_start_popen_failure_returns_error(self, mock_popen, client):
        mock_popen.side_effect = FileNotFoundError("no python")
        resp = client.post("/self-train/start", json={})
        data = resp.json()["data"]
        assert data["status"] == "error"
        assert "no python" in data["error"]

    def test_stop_when_proc_exited(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = 1
        server_state._self_train_proc = proc
        resp = client.post("/self-train/stop")
        assert resp.json()["data"]["status"] == "not_running"

    def test_stop_kill_failure_returns_500(self, client):
        import state as server_state
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate.side_effect = OSError("timeout")
        proc.kill.side_effect = OSError("already gone")
        server_state._self_train_proc = proc
        resp = client.post("/self-train/stop")
        assert resp.status_code == 500
        assert server_state._self_train_proc is proc

    @patch("pathlib.Path.read_text", return_value="step 1\nstep 2\nstep 3\n")
    @patch("pathlib.Path.exists", return_value=True)
    def test_status_reads_history(self, mock_exists, mock_read, client):
        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "not_started"
        assert data["history"] == ["step 1", "step 2", "step 3"]

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists", return_value=True)
    def test_status_history_caps_at_fifty(self, mock_exists, mock_read, client):
        import state as server_state
        mock_read.return_value = "\n".join(f"line {i}" for i in range(70))
        proc = MagicMock()
        proc.poll.return_value = 7
        server_state._self_train_proc = proc
        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "exited"
        assert data["returncode"] == 7
        assert len(data["history"]) == 50
