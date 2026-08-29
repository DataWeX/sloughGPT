"""
Tests for the system router — metrics, info, disk, lifecycle, executor, inference pool.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.system import router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
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
        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body

    def test_executor_job_result_not_found(self, client):
        resp = client.get("/system/executor/nonexistent/result")
        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body

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
            assert resp.status_code == 404
            body = resp.json()
            assert "error" in body
        finally:
            self._restore()

    def test_job_result_unknown_id(self, client):
        mod = self._install()
        try:
            resp = client.get("/system/executor/ghost/result")
            assert resp.status_code == 404
            body = resp.json()
            assert "error" in body
        finally:
            self._restore()


class TestInferencePool:
    """GET /system/inference-pool"""

    def test_returns_unavailable_when_not_initialized(self, client):
        resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "initialized" in data


class TestOutputStream:
    """GET /system/stream — SSE output stream."""

    def _make_sub(self, lines=(), timeout_raises=False):
        import asyncio

        class FakeSub:
            name = "fake-sub"

            def __init__(self, items):
                self._items = list(items)

            def read(self, timeout=0.1):
                if timeout_raises:
                    raise Exception("timeout")
                out = self._items
                self._items = []
                return out

            async def async_read(self, timeout=0.1):
                if timeout_raises:
                    raise Exception("timeout")
                out = self._items
                self._items = []
                return out

        return FakeSub(lines)

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_stream_emits_history_then_exits(self, mock_get_buf, client):
        import asyncio
        from unittest.mock import AsyncMock

        buf = MagicMock()
        hist = [MagicMock(to_sse=lambda: '{"text": "boot"}')]
        buf.tail.return_value = hist
        sub = self._make_sub([])
        buf.subscribe.return_value = sub
        mock_get_buf.return_value = buf

        with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])), \
             patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            with client.stream("GET", "/system/stream") as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                body = resp.read().decode()
                assert '{"text": "boot"}' in body

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_stream_pushes_live_lines(self, mock_get_buf, client):
        import asyncio
        from unittest.mock import AsyncMock

        buf = MagicMock()
        buf.tail.return_value = []
        sub = self._make_sub([MagicMock(to_sse=lambda: '{"text": "live"}')])
        buf.subscribe.return_value = sub
        mock_get_buf.return_value = buf

        with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])), \
             patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            with client.stream("GET", "/system/stream") as resp:
                body = resp.read().decode()
                assert '{"text": "live"}' in body

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_stream_unsubscribes_on_close(self, mock_get_buf, client):
        import asyncio
        from unittest.mock import AsyncMock

        buf = MagicMock()
        buf.tail.return_value = []
        buf.subscribe.return_value = self._make_sub([])
        mock_get_buf.return_value = buf

        with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])), \
             patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            with client.stream("GET", "/system/stream") as resp:
                resp.read()
        buf.unsubscribe.assert_called_once()


class TestSystemValidation:
    """Validation bounds and method mismatches."""

    def test_metrics_wrong_method_405(self, client):
        resp = client.post("/system/metrics")
        assert resp.status_code == 405

    def test_disk_wrong_method_405(self, client):
        resp = client.post("/system/disk")
        assert resp.status_code == 405

    def test_info_wrong_method_405(self, client):
        resp = client.post("/system/info")
        assert resp.status_code == 405

    def test_output_wrong_method_405(self, client):
        resp = client.post("/system/output")
        assert resp.status_code == 405

    def test_executor_wrong_method_405(self, client):
        resp = client.post("/system/executor")
        assert resp.status_code == 405

    def test_output_n_above_max_422(self, client):
        resp = client.get("/system/output?n=1001")
        assert resp.status_code == 422

    def test_output_n_below_min_422(self, client):
        resp = client.get("/system/output?n=0")
        assert resp.status_code == 422

    def test_stream_tail_above_max_422(self, client):
        resp = client.get("/system/stream?tail=501")
        assert resp.status_code == 422

    def test_purge_max_age_below_zero_422(self, client):
        resp = client.post("/system/executor/purge?max_age_s=0")
        assert resp.status_code == 422



