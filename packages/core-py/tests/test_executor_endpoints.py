"""Tests for /system/executor and /system/inference-pool API endpoints."""

import sys
import time
import threading
from pathlib import Path

# Ensure repo root AND apps/api/server are on sys.path
_repo_root = str(Path(__file__).resolve().parents[3])
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
for _p in (_repo_root, _server_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.system import router


@pytest.fixture(autouse=True)
def reset_executor():
    """Reset TrainingExecutor singleton before each test."""
    import domains.training.executor as exec_mod
    old = exec_mod._instance
    exec_mod._instance = None
    yield
    if old is not None:
        old.shutdown(wait=False)
    exec_mod._instance = None


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestExecutorEndpoints:
    """GET /system/executor, /system/executor/{id}, purge, cancel."""

    def test_get_executor_empty(self, client):
        resp = client.get("/system/executor")
        assert resp.status_code == 200
        body = resp.json()
        data = body["data"]
        assert data["initialized"] is False
        assert data["total_tracked"] == 0
        assert data["jobs"] == []

    def test_get_executor_with_jobs(self, client):
        from domains.training.executor import get_training_executor

        ex = get_training_executor()

        def noop(job_id):
            pass

        ex.submit(noop, "ep_a")
        ex.submit(noop, "ep_b")
        time.sleep(0.15)

        resp = client.get("/system/executor")
        data = resp.json()["data"]
        assert data["initialized"] is True
        assert data["total_tracked"] == 2
        assert data["max_workers"] >= 1
        ids = {j["job_id"] for j in data["jobs"]}
        assert ids == {"ep_a", "ep_b"}

    def test_get_single_job(self, client):
        from domains.training.executor import get_training_executor

        ex = get_training_executor()

        def noop(job_id):
            pass

        ex.submit(noop, "single_1")
        time.sleep(0.1)

        resp = client.get("/system/executor/single_1")
        data = resp.json()["data"]
        assert data["job_id"] == "single_1"
        assert data["status"] == "completed"
        assert data["cancel_requested"] is False

    def test_get_single_job_not_found(self, client):
        from domains.training.executor import get_training_executor
        get_training_executor()  # ensure initialized
        resp = client.get("/system/executor/nonexistent")
        data = resp.json()["data"]
        assert "not found" in data["error"]

    def test_get_result_completed_job(self, client):
        import numpy as np
        from domains.training.executor import get_training_executor

        ex = get_training_executor()

        def train_fn(job_id):
            return {"w1": np.zeros(16, dtype=np.float32)}

        ex.submit(train_fn, "res_1")
        time.sleep(0.1)

        resp = client.get("/system/executor/res_1/result")
        data = resp.json()["data"]
        assert data["job_id"] == "res_1"
        assert "w1" in data["weights"]
        assert data["weights"]["w1"]["shape"] == [16]
        assert data["weights"]["w1"]["dtype"] == "float32"
        assert data["total_bytes"] > 0

    def test_get_result_running_job(self, client):
        from domains.training.executor import get_training_executor

        ex = get_training_executor()
        evt = threading.Event()

        def slow(job_id):
            evt.wait(timeout=2)

        ex.submit(slow, "run_res")
        time.sleep(0.05)

        resp = client.get("/system/executor/run_res/result")
        data = resp.json()["data"]
        assert "error" in data
        evt.set()
        time.sleep(0.1)

    def test_purge_old_jobs(self, client):
        from domains.training.executor import get_training_executor

        ex = get_training_executor()

        def noop(job_id):
            pass

        ex.submit(noop, "old_purge")
        time.sleep(0.1)
        ex._jobs["old_purge"].completed_at = time.time() - 7200

        resp = client.post("/system/executor/purge?max_age_s=3600")
        data = resp.json()["data"]
        assert data["purged"] == 1
        assert ex.status("old_purge") is None

    def test_cancel_queued_job(self, client):
        from domains.training.executor import get_training_executor

        ex = get_training_executor()
        evt = threading.Event()

        def blocker(job_id):
            evt.wait(timeout=2)

        ex.submit(blocker, "block")
        time.sleep(0.05)
        job_id = ex.submit(blocker, "to_cancel")
        time.sleep(0.01)

        resp = client.post(f"/system/executor/{job_id}/cancel")
        data = resp.json()["data"]
        assert data["cancelled"] is True
        evt.set()
        time.sleep(0.1)

    def test_cancel_nonexistent_job(self, client):
        resp = client.post("/system/executor/no_such_job/cancel")
        data = resp.json()["data"]
        assert data["cancelled"] is False

    def test_cancel_running_job_sets_flag(self, client):
        from domains.training.executor import get_training_executor

        ex = get_training_executor()
        evt = threading.Event()

        def blocker(job_id):
            evt.wait(timeout=2)

        job_id = ex.submit(blocker, "run_cancel")
        time.sleep(0.05)

        resp = client.post(f"/system/executor/{job_id}/cancel")
        data = resp.json()["data"]
        assert data["cancelled"] is True

        # Job status should still be running but cancel_requested should be True
        resp = client.get(f"/system/executor/{job_id}")
        job_data = resp.json()["data"]
        assert job_data["status"] == "running"
        assert job_data["cancel_requested"] is True

        evt.set()
        time.sleep(0.1)

    def test_executor_not_initialized(self, client):
        """Endpoints return gracefully when executor hasn't been created."""
        # _instance is None due to reset_executor fixture
        resp = client.get("/system/executor")
        data = resp.json()["data"]
        assert data["initialized"] is False

        resp = client.get("/system/executor/any_id")
        assert resp.status_code == 200

        resp = client.post("/system/executor/purge?max_age_s=1")
        data = resp.json()["data"]
        assert data["purged"] == 0

        resp = client.post("/system/executor/any_id/cancel")
        data = resp.json()["data"]
        assert data["cancelled"] is False


class TestInferencePoolEndpoint:
    """GET /system/inference-pool."""

    def test_inference_pool_status(self, client):
        resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "initialized" in data
