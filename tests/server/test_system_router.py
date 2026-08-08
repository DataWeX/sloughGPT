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

    def test_memory_fields_present(self, client):
        data = client.get("/system/metrics").json()["data"]
        assert "memory_used_gb" in data
        assert "memory_total_gb" in data
        assert "cpu_count_physical" in data


class TestInfo:
    """GET /system/info"""

    def test_returns_platform_info(self, client):
        resp = client.get("/system/info")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "platform" in data
        assert "cpu_count" in data
        assert isinstance(data["cpu_count"], int)

    def test_returns_architecture_and_processor(self, client):
        data = client.get("/system/info").json()["data"]
        assert "architecture" in data
        assert "processor" in data


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

    def test_gb_fields_are_positive(self, client):
        data = client.get("/system/disk").json()["data"]
        assert data["total_gb"] > 0
        assert data["free_gb"] >= 0


class TestLifecycle:
    """GET /system/lifecycle"""

    def test_returns_lifecycle_state(self, client):
        resp = client.get("/system/lifecycle")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "phase" in data

    def test_returns_profile(self, client):
        data = client.get("/system/lifecycle").json()["data"]
        assert "profile" in data


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

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_lists_actual_lines(self, mock_get_buf, client):
        buf = MagicMock()
        buf.tail_dicts.return_value = [{"text": "hello", "level": "info", "ts": 1.0}]
        buf.count = 1
        buf.seq = 5
        mock_get_buf.return_value = buf
        data = client.get("/system/output?n=10").json()["data"]
        assert data["size"] == 1
        assert data["seq"] == 5
        assert data["lines"][0]["text"] == "hello"


class TestExecutor:
    """GET /system/executor"""

    def test_returns_uninitialized_when_not_setup(self, client):
        import domains.training.executor as executor_mod
        old = executor_mod._instance
        try:
            executor_mod._instance = None
            resp = client.get("/system/executor")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["initialized"] is False
        finally:
            executor_mod._instance = old

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


class TestExecutorInitialized:
    """GET /system/executor with a real TrainingExecutor instance."""

    def _install(self):
        import domains.training.executor as executor_mod
        self._old = executor_mod._instance
        executor_mod._instance = executor_mod.TrainingExecutor(max_workers=2)
        return executor_mod

    def _restore(self):
        import domains.training.executor as executor_mod
        executor_mod._instance = self._old

    def test_initialized_status(self, client):
        mod = self._install()
        try:
            resp = client.get("/system/executor")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["initialized"] is True
            assert data["max_workers"] == 2
            assert data["jobs"] == []
        finally:
            self._restore()

    def test_job_status_unknown_id(self, client):
        mod = self._install()
        try:
            resp = client.get("/system/executor/ghost")
            assert resp.status_code == 200
            assert "error" in resp.json()["data"]
        finally:
            self._restore()

    def test_job_result_unknown_id(self, client):
        mod = self._install()
        try:
            resp = client.get("/system/executor/ghost/result")
            assert resp.status_code == 200
            assert "error" in resp.json()["data"]
        finally:
            self._restore()


class TestInferencePool:
    """GET /system/inference-pool"""

    def test_returns_unavailable_when_not_initialized(self, client):
        resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "initialized" in data



