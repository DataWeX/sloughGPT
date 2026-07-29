"""
Tests for the system router — metrics, info, disk, lifecycle, executor, inference pool.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.system import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestMetrics:
    """GET /system/metrics"""

    def test_returns_cpu_and_memory(self, client):
        resp = client.get("/system/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "cpu_count_logical" in data
        assert isinstance(data["cpu_count_logical"], int)

    def test_cache_hits(self, client):
        a = client.get("/system/metrics").json()
        b = client.get("/system/metrics").json()
        assert a["data"] == b["data"]


class TestInfo:
    """GET /system/info"""

    def test_returns_platform_info(self, client):
        resp = client.get("/system/info")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "platform" in data
        assert "cpu_count" in data
        assert isinstance(data["cpu_count"], int)


class TestDisk:
    """GET /system/disk"""

    def test_returns_disk_usage(self, client):
        resp = client.get("/system/disk")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_gb" in data
        assert "used_gb" in data
        assert "free_gb" in data
        assert "percent" in data
        assert 0 <= data["percent"] <= 100


class TestLifecycle:
    """GET /system/lifecycle"""

    def test_returns_lifecycle_state(self, client):
        resp = client.get("/system/lifecycle")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "phase" in data


class TestTailOutput:
    """GET /system/output"""

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_returns_output_lines(self, mock_get_buf, client):
        buf = MagicMock()
        buf.tail_dicts.return_value = []
        buf.count = 0
        buf.seq = 0
        mock_get_buf.return_value = buf
        resp = client.get("/system/output")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "lines" in data
        assert "size" in data


class TestExecutor:
    """GET /system/executor"""

    def test_returns_uninitialized_when_not_setup(self, client):
        resp = client.get("/system/executor")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["initialized"] is False

    def test_executor_job_not_found(self, client):
        resp = client.get("/system/executor/nonexistent")
        assert resp.status_code == 200
        assert "error" in resp.json()["data"]

    def test_executor_job_result_not_found(self, client):
        resp = client.get("/system/executor/nonexistent/result")
        assert resp.status_code == 200
        assert "error" in resp.json()["data"]

    def test_purge_when_uninitialized(self, client):
        resp = client.post("/system/executor/purge")
        assert resp.status_code == 200
        assert resp.json()["data"]["purged"] == 0

    def test_cancel_when_uninitialized(self, client):
        resp = client.post("/system/executor/job-1/cancel")
        assert resp.status_code == 200
        assert resp.json()["data"]["cancelled"] is False


class TestInferencePool:
    """GET /system/inference-pool"""

    def test_returns_unavailable_when_not_initialized(self, client):
        resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "initialized" in data



