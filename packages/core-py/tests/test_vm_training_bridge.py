"""Tests for shell.vm_training_bridge — VMTrainingBridge job tracking."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from domains.shell.vm_training_bridge import VMTrainingBridge, get_bridge


# ── VMTrainingBridge ───────────────────────────────────────────────────────


class TestVMTrainingBridge:

    def setup_method(self):
        self.bridge = VMTrainingBridge()

    def test_init(self):
        assert self.bridge._jobs == {}
        assert self.bridge._next_job_id == 1

    def test_start_valid_json(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"job_id": "api-123"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(self.bridge._session, "post", return_value=mock_resp):
            job_id = self.bridge.start(json.dumps({"dataset": "test", "epochs": 5}))
            assert job_id == 1
            assert "api-123" in str(self.bridge._jobs[1])

    def test_start_invalid_json(self):
        assert self.bridge.start("not json") == -1

    def test_start_non_dict_json(self):
        assert self.bridge.start("[1,2,3]") == -1

    def test_start_api_error(self):
        import requests
        with patch.object(self.bridge._session, "post", side_effect=requests.RequestException("fail")):
            assert self.bridge.start(json.dumps({"dataset": "test"})) == -1

    def test_start_multiple_jobs(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"job_id": "api-1"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(self.bridge._session, "post", return_value=mock_resp):
            j1 = self.bridge.start(json.dumps({"dataset": "a"}))
            j2 = self.bridge.start(json.dumps({"dataset": "b"}))
            assert j1 == 1
            assert j2 == 2

    def test_status_not_found(self):
        result = self.bridge.status(999)
        assert result["status"] == "not_found"

    def test_status_completed(self):
        self.bridge._jobs[1] = {"status": "completed", "progress": 1.0}
        result = self.bridge.status(1)
        assert result["status"] == "completed"

    def test_status_failed(self):
        self.bridge._jobs[1] = {"status": "failed", "error": "oom"}
        result = self.bridge.status(1)
        assert result["status"] == "failed"
        assert result["error"] == "oom"

    def test_status_polls_api(self):
        self.bridge._jobs[1] = {"api_job_id": "api-1", "status": "running"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "completed", "progress": 100}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(self.bridge._session, "get", return_value=mock_resp):
            result = self.bridge.status(1)
            assert result["status"] == "completed"
            assert result["progress"] == 1.0

    def test_status_api_404(self):
        self.bridge._jobs[1] = {"api_job_id": "api-1", "status": "running"}
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status = MagicMock(side_effect=requests.RequestException("404"))

        with patch.object(self.bridge._session, "get", return_value=mock_resp):
            result = self.bridge.status(1)
            assert result["status"] == "not_found"

    def test_status_api_error(self):
        self.bridge._jobs[1] = {"api_job_id": "api-1", "status": "running"}
        import requests
        with patch.object(self.bridge._session, "get", side_effect=requests.RequestException("fail")):
            result = self.bridge.status(1)
            assert result["status"] == "running"

    def test_get_result_json_not_completed(self):
        self.bridge._jobs[1] = {"status": "running"}
        assert self.bridge.get_result_json(1) is None

    def test_get_result_json_completed(self):
        self.bridge._jobs[1] = {
            "status": "completed",
            "_result_data": {"status": "completed", "loss": 1.5, "checkpoint": "/path/ckpt"},
        }
        result = self.bridge.get_result_json(1)
        data = json.loads(result)
        assert data["success"] is True
        assert data["final_loss"] == 1.5
        assert data["model_path"] == "/path/ckpt"

    def test_get_result_json_not_found(self):
        assert self.bridge.get_result_json(999) is None

    def test_stop_not_found(self):
        assert self.bridge.stop(999) is False

    def test_stop_no_api_job(self):
        self.bridge._jobs[1] = {"api_job_id": ""}
        assert self.bridge.stop(1) is False

    def test_stop_success(self):
        self.bridge._jobs[1] = {"api_job_id": "api-1", "status": "running"}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(self.bridge._session, "post", return_value=mock_resp):
            assert self.bridge.stop(1) is True
            assert self.bridge._jobs[1]["status"] == "stopping"

    def test_stop_api_error(self):
        self.bridge._jobs[1] = {"api_job_id": "api-1", "status": "running"}
        import requests
        with patch.object(self.bridge._session, "post", side_effect=requests.RequestException("fail")):
            assert self.bridge.stop(1) is False

    def test_remove_existing(self):
        self.bridge._jobs[1] = {"status": "completed"}
        assert self.bridge.remove(1) is True
        assert 1 not in self.bridge._jobs

    def test_remove_not_found(self):
        assert self.bridge.remove(999) is False

    def test_job_info(self):
        self.bridge._jobs[1] = {"api_job_id": "api-1", "status": "running", "progress": 0.5}
        info = self.bridge.job_info(1)
        assert info["api_job_id"] == "api-1"
        assert info["status"] == "running"
        assert info["progress"] == 0.5

    def test_job_info_not_found(self):
        assert self.bridge.job_info(999) is None

    def test_alive_count(self):
        self.bridge._jobs[1] = {"status": "running"}
        self.bridge._jobs[2] = {"status": "completed"}
        self.bridge._jobs[3] = {"status": "running"}
        assert self.bridge.alive_count() == 2

    def test_alive_count_empty(self):
        assert self.bridge.alive_count() == 0


# ── Singleton ─────────────────────────────────────────────────────────────


class TestSingleton:

    def test_get_returns_same(self):
        import domains.shell.vm_training_bridge as mod
        mod._bridge = None
        b1 = get_bridge()
        b2 = get_bridge()
        assert b1 is b2
        mod._bridge = None
