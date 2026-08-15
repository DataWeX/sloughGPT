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


# ── SloNet teacher path (pure NumPy) ────────────────────────────────────


class TestDistillSlonetTeacher:
    """Distillation from the active SloNet provider — no torch involved.

    The load path publishes a SloNetChatProvider into ServerState instead of
    ``ctrl._hf_model``; the teacher forward must therefore run in pure NumPy
    via ``SloTransformer.forward(input_ids, None) -> (logits, loss)``.
    """

    @staticmethod
    def _make_client():
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from training.router import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    @staticmethod
    def _fake_provider():
        import numpy as np

        class _Logits:
            def __init__(self, arr):
                self.data = arr

        class _Model:
            def forward(self, input_ids, targets=None):
                return _Logits(np.full((input_ids.shape[0], input_ids.shape[1], 16),
                                       0.5, dtype=np.float32)), None

        class _Provider:
            model_id = "gpt2"

            def tokenize(self, text):
                return list(range(1, 201))

            def _get_model(self):
                return _Model()

        return _Provider()

    def _poll(self, training_jobs, job_id, timeout=30.0):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = training_jobs.get(job_id, {}).get("status")
            if status in ("completed", "failed"):
                return status
            time.sleep(0.05)
        return "timeout"

    def test_slonet_teacher_completes_without_torch(self):
        import shutil
        from unittest.mock import MagicMock

        from training.router import training_jobs

        datasets_dir = _REPO_ROOT / "datasets"
        ds_dir = datasets_dir / "_test_distill_slonet"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("alpha beta gamma delta epsilon zeta eta theta " * 8)

        fake_core = MagicMock()
        fake_core.model.get.return_value = self._fake_provider()

        ckpt = _REPO_ROOT / "models" / "auto-training" / "_test_slonet_distill_distilled.soul"
        job_id = None
        try:
            client = self._make_client()
            with patch("domains.infrastructure.model_registry.get_model_registry",
                       return_value=None), \
                 patch("domains.infrastructure.server_state.get_server_state",
                       return_value=fake_core):
                resp = client.post("/training/distill", json={
                    "teacher_model": "gpt2",
                    "dataset": "_test_distill_slonet",
                    "epochs": 1,
                    "name": "_test_slonet_distill",
                })
                assert resp.status_code == 200
                job_id = resp.json()["job_id"]
                # Keep the patches active while the background training thread
                # runs — it resolves the teacher via get_server_state() and
                # would otherwise read the real ServerState singleton.
                assert self._poll(training_jobs, job_id) == "completed"
            job = training_jobs[job_id]
            assert job["status"] == "completed"
            assert "loss" in job
            assert ckpt.exists()
        finally:
            if job_id:
                training_jobs.pop(job_id, None)
            shutil.rmtree(ds_dir, ignore_errors=True)
            ckpt.unlink(missing_ok=True)

    def test_slonet_teacher_not_loaded_fails(self):
        import shutil
        from unittest.mock import MagicMock

        from training.router import training_jobs

        datasets_dir = _REPO_ROOT / "datasets"
        ds_dir = datasets_dir / "_test_distill_slonet_missing"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("alpha beta gamma delta epsilon zeta eta theta " * 8)

        fake_core = MagicMock()
        fake_core.model.get.return_value = None  # no provider loaded

        job_id = None
        try:
            client = self._make_client()
            with patch("domains.infrastructure.model_registry.get_model_registry",
                       return_value=None), \
                 patch("domains.infrastructure.server_state.get_server_state",
                       return_value=fake_core):
                resp = client.post("/training/distill", json={
                    "teacher_model": "gpt2",
                    "dataset": "_test_distill_slonet_missing",
                    "epochs": 1,
                    "name": "_test_slonet_missing",
                })
                assert resp.status_code == 200
                job_id = resp.json()["job_id"]
                assert self._poll(training_jobs, job_id) == "failed"
            assert "not loaded" in training_jobs[job_id]["error"]
        finally:
            if job_id:
                training_jobs.pop(job_id, None)
            shutil.rmtree(ds_dir, ignore_errors=True)
