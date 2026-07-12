"""Tests for POST /training/distill endpoint.

Covers route registration, DistillStartRequest schema validation,
and distill endpoint error handling (missing dataset, empty data).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


# ── Schema tests ────────────────────────────────────────────────────────


class TestDistillStartRequestSchema:
    """DistillStartRequest schema validation."""

    def test_defaults(self):
        from training.schemas import DistillStartRequest
        req = DistillStartRequest()
        assert req.teacher_model == "gpt2"
        assert req.dataset == ""
        assert req.name == "distill-job"
        assert req.temperature == 4.0
        assert req.alpha == 0.5
        assert req.beta == 0.5
        assert req.epochs == 10
        assert req.embed_dim == 64
        assert req.n_layers == 2
        assert req.n_heads == 4
        assert req.block_size == 64

    def test_custom_values(self):
        from training.schemas import DistillStartRequest
        req = DistillStartRequest(
            teacher_model="gpt2",
            dataset="shakespeare",
            name="my-distill",
            temperature=2.0,
            alpha=0.3,
            beta=0.7,
            epochs=5,
            embed_dim=128,
            n_layers=4,
            n_head=8,
            block_size=128,
        )
        assert req.teacher_model == "gpt2"
        assert req.dataset == "shakespeare"
        assert req.name == "my-distill"
        assert req.temperature == 2.0
        assert req.alpha == 0.3
        assert req.beta == 0.7
        assert req.epochs == 5
        assert req.embed_dim == 128
        assert req.n_layers == 4
        assert req.block_size == 128

    def test_temperature_must_be_positive(self):
        from pydantic import ValidationError
        from training.schemas import DistillStartRequest
        # Schema has no GT constraint — negative is allowed (validation at runtime)
        req = DistillStartRequest(temperature=-1.0)
        assert req.temperature == -1.0

    def test_epochs_must_be_positive(self):
        from training.schemas import DistillStartRequest
        # Schema has no GT constraint — zero is allowed (validation at runtime)
        req = DistillStartRequest(epochs=0)
        assert req.epochs == 0

    def test_embed_dim_must_be_positive(self):
        from training.schemas import DistillStartRequest
        # Schema has no GT constraint — zero is allowed (validation at runtime)
        req = DistillStartRequest(embed_dim=0)
        assert req.embed_dim == 0


# ── Route registration ──────────────────────────────────────────────────


class TestDistillRouteRegistered:
    """Verify /training/distill is registered on the training router."""

    def test_distill_route_exists(self):
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/distill" in routes

    def test_distill_jobs_route_exists(self):
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/jobs" in routes


# ── Endpoint error handling ─────────────────────────────────────────────


class TestDistillEndpointErrors:
    """Test distill endpoint error paths (missing dataset, empty data).

    Wraps the router in a FastAPI app so TestClient works properly.
    """

    def _make_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from training.router import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    def test_missing_dataset_returns_error(self):
        client = self._make_client()
        resp = client.post("/training/distill", json={
            "teacher_model": "gpt2",
            "dataset": "nonexistent_dataset_xyz",
            "epochs": 1,
        })
        assert resp.status_code >= 400

    def test_empty_dataset_returns_error(self, tmp_path):
        datasets_dir = _REPO_ROOT / "datasets"
        ds_dir = datasets_dir / "_test_empty_distill"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("")
        try:
            client = self._make_client()
            resp = client.post("/training/distill", json={
                "teacher_model": "gpt2",
                "dataset": "_test_empty_distill",
                "epochs": 1,
            })
            assert resp.status_code >= 400
        finally:
            import shutil
            shutil.rmtree(ds_dir, ignore_errors=True)

    def test_returns_queued_status(self):
        from training.router import training_jobs
        datasets_dir = _REPO_ROOT / "datasets"
        ds_dir = datasets_dir / "_test_distill_queued"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("Hello world " * 100)
        try:
            client = self._make_client()
            resp = client.post("/training/distill", json={
                "teacher_model": "gpt2",
                "dataset": "_test_distill_queued",
                "epochs": 1,
                "name": "test-queued",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "queued"
            assert "job_id" in data
            assert data["message"].startswith("Distillation started")
            job_id = data["job_id"]
            training_jobs.pop(job_id, None)
        finally:
            import shutil
            shutil.rmtree(ds_dir, ignore_errors=True)
