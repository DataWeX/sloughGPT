"""Tests for the /self-train router (start/stop/status)."""

from unittest.mock import patch, MagicMock
from test_support import get_test_client
import state as server_state


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


def _cleanup():
    """Reset the self-train process in server state."""
    server_state._self_train_proc = None


class TestStatus:
    def setup_method(self):
        _cleanup()

    def test_status_not_started(self):
        client = get_test_client()
        resp = client.get("/self-train/status")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "not_started"
        assert "history" in data


class TestStart:
    def setup_method(self):
        _cleanup()

    def test_start_default(self):
        client = get_test_client()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc):
            resp = client.post("/self-train/start")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "started"
        assert data["pid"] == 12345
        _cleanup()

    def test_start_with_model(self):
        client = get_test_client()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            resp = client.post("/self-train/start", json={"model": "test-model"})
        assert resp.status_code == 200
        cmd = mock_popen.call_args[0][0]
        assert "--model" in cmd
        assert "test-model" in cmd
        _cleanup()

    def test_start_invalid_model_name(self):
        client = get_test_client()
        resp = client.post("/self-train/start", json={"model": "rm -rf /"})
        assert resp.status_code == 422
        _cleanup()

    def test_start_already_running(self):
        client = get_test_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 111
        server_state._self_train_proc = mock_proc
        resp = client.post("/self-train/start")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "already_running"
        _cleanup()

    def test_start_with_temperature(self):
        client = get_test_client()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            resp = client.post("/self-train/start", json={"temperature": 1.5})
        assert resp.status_code == 200
        cmd = mock_popen.call_args[0][0]
        assert "--temperature" in cmd
        assert "1.5" in cmd
        _cleanup()

    def test_start_with_forever(self):
        client = get_test_client()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            resp = client.post("/self-train/start", json={"forever": True})
        assert resp.status_code == 200
        cmd = mock_popen.call_args[0][0]
        assert "--forever" in cmd
        _cleanup()

    def test_temperature_out_of_range(self):
        client = get_test_client()
        resp = client.post("/self-train/start", json={"temperature": 3.0})
        assert resp.status_code == 422
        _cleanup()


class TestStop:
    def setup_method(self):
        _cleanup()

    def test_stop_not_running(self):
        client = get_test_client()
        resp = client.post("/self-train/stop")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "not_running"

    def test_stop_running(self):
        client = get_test_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 999
        server_state._self_train_proc = mock_proc
        resp = client.post("/self-train/stop")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "stopped"
        mock_proc.terminate.assert_called_once()
        assert server_state._self_train_proc is None
