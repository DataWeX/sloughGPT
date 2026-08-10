"""Tests for the system API router (routers/system.py).

Covers: metrics, info, disk, lifecycle, executor, tail, output, inference-pool.
psutil and domain deps are mocked; only HTTP-level behavior is tested.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import asyncio
import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.system import SystemRouter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system_router() -> SystemRouter:
    return SystemRouter()


def _app(sr: SystemRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(sr.router)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_returns_cpu_memory(self):
        sr = _make_system_router()
        client = TestClient(_app(sr))
        resp = client.get("/system/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "memory_used_gb" in data
        assert "memory_total_gb" in data
        assert "cpu_count_logical" in data
        assert "cpu_count_physical" in data

    def test_metrics_caching(self):
        sr = _make_system_router()
        client = TestClient(_app(sr))
        r1 = client.get("/system/metrics").json()["data"]
        r2 = client.get("/system/metrics").json()["data"]
        assert r1["cpu_percent"] == r2["cpu_percent"]


class TestInfo:
    def test_info_platform_fields(self):
        sr = _make_system_router()
        client = TestClient(_app(sr))
        resp = client.get("/system/info")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "platform" in data
        assert "architecture" in data
        assert "cpu_count" in data


class TestDisk:
    def test_disk_usage_fields(self):
        sr = _make_system_router()
        client = TestClient(_app(sr))
        resp = client.get("/system/disk")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_gb" in data
        assert "used_gb" in data
        assert "free_gb" in data
        assert "percent" in data


class TestLifecycle:
    def test_lifecycle_ok(self):
        sr = _make_system_router()
        mock_mgr = MagicMock()
        mock_mgr.get_results.return_value = {"phase": "ready", "profile": "default"}
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", return_value=mock_mgr):
            client = TestClient(_app(sr))
            resp = client.get("/system/lifecycle")
        assert resp.status_code == 200
        assert resp.json()["data"]["phase"] == "ready"

    def test_lifecycle_unavailable(self):
        sr = _make_system_router()
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager", side_effect=RuntimeError("not init")):
            client = TestClient(_app(sr))
            resp = client.get("/system/lifecycle")
        assert resp.status_code == 200
        assert resp.json()["data"]["phase"] == "unavailable"


class TestExecutor:
    def test_executor_not_initialized(self):
        sr = _make_system_router()
        with patch("domains.training.executor._instance", None):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["initialized"] is False
        assert data["active_jobs"] == 0

    def test_executor_initialized(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.active_count.return_value = 2
        mock_inst._max_workers = 4
        mock_inst._jobs = {"j1": {}, "j2": {}}
        mock_inst.list_jobs.return_value = [{"id": "j1"}, {"id": "j2"}]
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["initialized"] is True
        assert data["active_jobs"] == 2
        assert data["max_workers"] == 4

    def test_executor_job_not_found(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.status.return_value = None
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor/nonexistent")
        assert resp.status_code == 200
        assert "not found" in resp.json()["data"]["error"]

    def test_executor_job_found(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.status.return_value = {"id": "j1", "status": "running"}
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor/j1")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "running"

    def test_executor_result_not_found(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.result_summary.return_value = None
        mock_inst.status.return_value = None
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor/j1/result")
        assert resp.status_code == 200
        assert "not found" in resp.json()["data"]["error"]

    def test_executor_result_no_weight(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.result_summary.return_value = None
        mock_inst.status.return_value = {"id": "j1", "status": "running"}
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor/j1/result")
        assert resp.status_code == 200
        assert "not completed" in resp.json()["data"]["error"]

    def test_executor_result_ok(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.result_summary.return_value = {"weights": ["W_ih"], "total_bytes": 1024}
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.get("/system/executor/j1/result")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_bytes"] == 1024

    def test_purge_not_initialized(self):
        sr = _make_system_router()
        with patch("domains.training.executor._instance", None):
            client = TestClient(_app(sr))
            resp = client.post("/system/executor/purge")
        assert resp.status_code == 200
        assert resp.json()["data"]["purged"] == 0

    def test_purge_ok(self):
        sr = _make_system_router()
        mock_inst = MagicMock()
        mock_inst.purge_completed.return_value = 3
        with patch("domains.training.executor._instance", mock_inst):
            client = TestClient(_app(sr))
            resp = client.post("/system/executor/purge")
        assert resp.status_code == 200
        assert resp.json()["data"]["purged"] == 3

    def test_cancel_not_initialized(self):
        sr = _make_system_router()
        with patch("domains.training.executor._instance", None):
            client = TestClient(_app(sr))
            resp = client.post("/system/executor/j1/cancel")
        assert resp.status_code == 200
        assert resp.json()["data"]["cancelled"] is False


class TestTailOutput:
    def test_tail_output(self):
        sr = _make_system_router()
        mock_buf = MagicMock()
        mock_buf.tail_dicts.return_value = [{"text": "line1"}]
        mock_buf.count = 1
        mock_buf.seq = 1
        with patch("domains.infrastructure.output_buffer.get_server_buffer", return_value=mock_buf):
            client = TestClient(_app(sr))
            resp = client.get("/system/output")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["lines"]) == 1
        assert data["size"] == 1


class TestInferencePool:
    def test_pool_not_initialized(self):
        sr = _make_system_router()
        async def _raise():
            raise RuntimeError("no pool")
        with patch("infrastructure.inference_pool.InferencePool.get_instance", _raise):
            client = TestClient(_app(sr))
            resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        assert resp.json()["data"]["initialized"] is False

    def test_pool_initialized(self):
        sr = _make_system_router()
        mock_pool = MagicMock()
        mock_pool._max_workers = 4
        mock_pool._queue_timeout = 30
        async def _get():
            return mock_pool
        with patch("infrastructure.inference_pool.InferencePool.get_instance", _get):
            client = TestClient(_app(sr))
            resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["initialized"] is True
        assert data["max_workers"] == 4
