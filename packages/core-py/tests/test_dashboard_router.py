"""
Dashboard Router Tests
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from routers.dashboard import DashboardRouter, _get_health_summary, _get_active_processes

app = FastAPI()
router_obj = DashboardRouter()
app.include_router(router_obj.router)

original_imports = {}


def _mock(mod_name, attrs):
    m = MagicMock()
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[mod_name] = m
    original_imports[mod_name] = m
    return m


def _cleanup():
    for mod in list(original_imports):
        sys.modules.pop(mod, None)
    original_imports.clear()


@pytest.fixture(autouse=True)
def _setup():
    _cleanup()
    yield
    _cleanup()


class TestGetHealthSummary:
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent", return_value=45.0)
    def test_returns_health_dict(self, mock_cpu, mock_mem):
        mock_mem.return_value = MagicMock(percent=72.0, used=4 * 1024**3)

        ss = MagicMock()
        ss.model.get.return_value = "gpt2"
        ss.provider.get.return_value = None
        ss.model_type.get.return_value = "llama"
        ss.uptime_seconds = 3600.0
        ss.request_count = 150
        ss.error_count = 3
        ss.get_tokens_per_second.return_value = 45.5
        ss.get_avg_latency.return_value = 120.5
        ss.get_requests_per_minute.return_value = 25.0

        _mock("state", {"_self_train_proc": None})
        _mock("domains.infrastructure.server_state", {"get_server_state": lambda: ss})

        result = _get_health_summary()
        assert result["model_loaded"] is True
        assert result["model_type"] == "llama"
        assert result["request_count"] == 150
        assert result["error_count"] == 3
        assert result["tokens_per_sec"] == 45.5
        assert result["cpu_percent"] == 45.0
        assert result["memory_percent"] == 72.0

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent", return_value=10.0)
    def test_no_model_loaded(self, mock_cpu, mock_mem):
        mock_mem.return_value = MagicMock(percent=50.0, used=2 * 1024**3)

        ss = MagicMock()
        ss.model.get.return_value = None
        ss.provider.get.return_value = None
        ss.model_type.get.return_value = None
        ss.uptime_seconds = 0.0
        ss.request_count = 0
        ss.error_count = 0
        ss.get_tokens_per_second.return_value = 0.0
        ss.get_avg_latency.return_value = 0.0
        ss.get_requests_per_minute.return_value = 0.0

        _mock("state", {"_self_train_proc": None})
        _mock("domains.infrastructure.server_state", {"get_server_state": lambda: ss})

        result = _get_health_summary()
        assert result["model_loaded"] is False
        assert result["model_type"] == ""


class TestGetActiveProcesses:
    def test_empty_when_no_jobs(self):
        _mock("state", {"_self_train_proc": None})
        _mock("training.jobs", {"training_jobs": {}})

        result = _get_active_processes()
        assert "self-train" not in result

    def test_running_training_job(self):
        _mock("state", {"_self_train_proc": None})
        _mock("training.jobs", {
            "training_jobs": {
                "abc123": {
                    "status": "running",
                    "progress": 50,
                    "current_step": "100",
                    "total_steps": "200",
                    "model": "gpt2",
                    "name": "my-run",
                }
            }
        })

        result = _get_active_processes()
        assert "train:abc123" in result
        assert result["train:abc123"]["status"] == "running"
        assert result["train:abc123"]["progress"] == 50
        assert "gpt2" in result["train:abc123"]["detail"]

    def test_completed_job_not_included(self):
        _mock("state", {"_self_train_proc": None})
        _mock("training.jobs", {
            "training_jobs": {
                "old1": {"status": "completed", "progress": 100}
            }
        })

        result = _get_active_processes()
        assert not any(k.startswith("train:") for k in result)


class TestDashboardSummary:
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent", return_value=25.0)
    def test_returns_envelope(self, mock_cpu, mock_mem):
        mock_mem.return_value = MagicMock(percent=60.0, used=3 * 1024**3)

        ss = MagicMock()
        ss.model.get.return_value = "gpt2"
        ss.provider.get.return_value = None
        ss.model_type.get.return_value = "llama"
        ss.uptime_seconds = 100.0
        ss.request_count = 10
        ss.error_count = 0
        ss.get_tokens_per_second.return_value = 20.0
        ss.get_avg_latency.return_value = 50.0
        ss.get_requests_per_minute.return_value = 5.0

        _mock("state", {"_self_train_proc": None})
        _mock("domains.infrastructure.server_state", {"get_server_state": lambda: ss})
        _mock("training.jobs", {"training_jobs": {}})
        _mock("domains.training.outcome_tracker", {
            "TrainingOutcomeTracker": MagicMock(return_value=MagicMock(get_stats=lambda: {"total_runs": 3}))
        })
        _mock("domains.settings.persistent", {
            "get_settings": MagicMock(return_value={"settings": True})
        })

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/dashboard/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert "health" in body["data"]
        assert "processes" in body["data"]
        assert "services" in body["data"]

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent", return_value=25.0)
    def test_services_has_total_and_healthy(self, mock_cpu, mock_mem):
        mock_mem.return_value = MagicMock(percent=60.0, used=3 * 1024**3)

        ss = MagicMock()
        ss.model.get.return_value = None
        ss.provider.get.return_value = None
        ss.model_type.get.return_value = None
        ss.uptime_seconds = 0.0
        ss.request_count = 0
        ss.error_count = 0
        ss.get_tokens_per_second.return_value = 0.0
        ss.get_avg_latency.return_value = 0.0
        ss.get_requests_per_minute.return_value = 0.0

        _mock("state", {"_self_train_proc": None})
        _mock("domains.infrastructure.server_state", {"get_server_state": lambda: ss})
        _mock("training.jobs", {"training_jobs": {}})
        _mock("domains.training.outcome_tracker", {
            "TrainingOutcomeTracker": MagicMock(return_value=MagicMock(get_stats=lambda: {"total_runs": 0}))
        })
        _mock("domains.settings.persistent", {
            "get_settings": MagicMock(return_value={"settings": True})
        })

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/dashboard/summary")
        body = response.json()
        services = body["data"]["services"]
        assert "total" in services
        assert "healthy" in services
        assert services["total"] >= 0

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent", return_value=25.0)
    def test_health_has_model_loaded(self, mock_cpu, mock_mem):
        mock_mem.return_value = MagicMock(percent=60.0, used=3 * 1024**3)

        ss = MagicMock()
        ss.model.get.return_value = "gpt2"
        ss.provider.get.return_value = None
        ss.model_type.get.return_value = "llama"
        ss.uptime_seconds = 100.0
        ss.request_count = 10
        ss.error_count = 0
        ss.get_tokens_per_second.return_value = 20.0
        ss.get_avg_latency.return_value = 50.0
        ss.get_requests_per_minute.return_value = 5.0

        _mock("state", {"_self_train_proc": None})
        _mock("domains.infrastructure.server_state", {"get_server_state": lambda: ss})
        _mock("training.jobs", {"training_jobs": {}})
        _mock("domains.training.outcome_tracker", {
            "TrainingOutcomeTracker": MagicMock(return_value=MagicMock(get_stats=lambda: {"total_runs": 3}))
        })
        _mock("domains.settings.persistent", {
            "get_settings": MagicMock(return_value={"settings": True})
        })

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/dashboard/summary")
        body = response.json()
        assert "model_loaded" in body["data"]["health"]


class TestDashboardEvents:
    def test_returns_events_list(self):
        _mock("domains.infrastructure.event_buffer", {
            "get_event_buffer": MagicMock(return_value=MagicMock(recent=lambda n: [{"type": "info", "msg": "test"}]))
        })

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/dashboard/events")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert isinstance(body["data"]["events"], list)
        assert "count" in body["data"]
