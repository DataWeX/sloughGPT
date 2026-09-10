"""Tests for the Cloud Training router — 4 endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.cloud_training import CloudTrainingRouter

    router_obj = CloudTrainingRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


# ── List jobs ────────────────────────────────────────────────────────────────


class TestListJobs:
    def test_list_jobs(self):
        _app = _build_app()
        mock_provider = MagicMock()
        mock_provider.list_jobs.return_value = []
        with patch("domains.training.cloud.get_provider", return_value=mock_provider):
            resp = TestClient(_app).get("/cloud-training/jobs")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "jobs" in data


# ── Submit job ───────────────────────────────────────────────────────────────


class TestSubmitJob:
    def test_submit_job(self):
        _app = _build_app()
        mock_provider = MagicMock()
        mock_provider.submit_job.return_value = "job-123"
        with patch("domains.training.cloud.get_provider", return_value=mock_provider):
            resp = TestClient(_app).post("/cloud-training/submit?provider=local&dataset_id=ds1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["job_id"] == "job-123"
        assert data["status"] == "submitted"


# ── Job status ───────────────────────────────────────────────────────────────


class TestJobStatus:
    def test_job_status(self):
        _app = _build_app()
        mock_provider = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-123"
        mock_job.provider = "local"
        mock_job.status = "running"
        mock_job.progress = 50
        mock_job.error = None
        mock_provider.get_status.return_value = mock_job
        with patch("domains.training.cloud.get_provider", return_value=mock_provider):
            resp = TestClient(_app).get("/cloud-training/job-123/status")
        assert resp.status_code == 200


# ── Cancel job ───────────────────────────────────────────────────────────────


class TestCancelJob:
    def test_cancel_job(self):
        _app = _build_app()
        mock_provider = MagicMock()
        mock_provider.cancel_job.return_value = True
        with patch("domains.training.cloud.get_provider", return_value=mock_provider):
            resp = TestClient(_app).post("/cloud-training/job-123/cancel")
        assert resp.status_code == 200
        assert resp.json()["data"]["cancelled"] is True
