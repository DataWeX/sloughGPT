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


@pytest.fixture(autouse=True)
def _in_memory_jobs(monkeypatch):
    """Training jobs must behave like a plain in-memory dict for these tests.

    The production ``training_jobs`` wrapper persists to MogDB, which is not
    suitable in a unit-test context (in-place ``.update()`` on the wrapper's
    returned docs would be lost, and runtime-only fields like ``_cancel_event``
    are not serializable). Forcing the wrapper's in-memory fallback preserves
    the dict contract these tests were written against.
    """
    monkeypatch.setattr("training.job_store.get_job_store", lambda: None)


# ── Schema tests ────────────────────────────────────────────────────────


class TestDistillStartRequestSchema:
    """DistillStartRequest schema validation."""

    def test_defaults(self):
        from training.schemas import DistillStartRequest

        req = DistillStartRequest(dataset="shakespeare")
        assert req.teacher_model == "gpt2"
        assert req.dataset == "shakespeare"
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

        # Schema constrains temperature to >= 0.1.
        with pytest.raises(ValidationError):
            DistillStartRequest(dataset="shakespeare", temperature=-1.0)

    def test_epochs_must_be_positive(self):
        from pydantic import ValidationError
        from training.schemas import DistillStartRequest

        # Schema constrains epochs to >= 1.
        with pytest.raises(ValidationError):
            DistillStartRequest(dataset="shakespeare", epochs=0)

    def test_embed_dim_must_be_positive(self):
        from pydantic import ValidationError
        from training.schemas import DistillStartRequest

        # Schema constrains embed_dim to >= 16.
        with pytest.raises(ValidationError):
            DistillStartRequest(dataset="shakespeare", embed_dim=0)


# ── Route registration ──────────────────────────────────────────────────


class TestDistillRouteRegistered:
    """Verify /training/distill is registered on the training router."""

    @staticmethod
    def _paths(router, out=None):
        out = [] if out is None else out
        for r in router.routes:
            orig = getattr(r, "original_router", None)
            if orig is not None:
                TestDistillRouteRegistered._paths(orig, out)
            elif getattr(r, "path", None):
                out.append(r.path)
        return out

    def test_distill_route_exists(self):
        from training.router import router

        assert "/training/distill" in list(self._paths(router))

    def test_distill_jobs_route_exists(self):
        from training.router import router

        assert "/training/jobs" in list(self._paths(router))


# ── Endpoint error handling ─────────────────────────────────────────────


class TestDistillEndpointErrors:
    """Test distill endpoint error paths (missing dataset, empty data).

    Wraps the router in a FastAPI app so TestClient works properly.
    """

    def _make_client(self):
        # build_test_app registers the production exception handlers; a bare
        # FastAPI() let raise_error()'s 400s escape as unhandled 500s, which
        # made the >=400 assertions pass while ==200 ones failed (the
        # misleading rotating signature in card 5473eb90).
        from fastapi.testclient import TestClient
        from training.router import router

        from tests.conftest import build_test_app

        return TestClient(build_test_app(router), raise_server_exceptions=False)

    def test_missing_dataset_returns_error(self):
        client = self._make_client()
        resp = client.post(
            "/training/distill",
            json={
                "teacher_model": "gpt2",
                "dataset": "nonexistent_dataset_xyz",
                "epochs": 1,
            },
        )
        assert resp.status_code >= 400

    def test_empty_dataset_returns_error(self, tmp_path):
        datasets_dir = _REPO_ROOT / "data"
        ds_dir = datasets_dir / "_test_empty_distill"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("")
        try:
            client = self._make_client()
            resp = client.post(
                "/training/distill",
                json={
                    "teacher_model": "gpt2",
                    "dataset": "_test_empty_distill",
                    "epochs": 1,
                },
            )
            assert resp.status_code >= 400
        finally:
            import shutil

            shutil.rmtree(ds_dir, ignore_errors=True)

    def test_returns_queued_status(self):
        from training.router import training_jobs

        datasets_dir = _REPO_ROOT / "data"
        ds_dir = datasets_dir / "_test_distill_queued"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("Hello world " * 100)
        try:
            client = self._make_client()
            resp = client.post(
                "/training/distill",
                json={
                    "teacher_model": "gpt2",
                    "dataset": "_test_distill_queued",
                    "epochs": 1,
                    "name": "test-queued",
                },
            )
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
        # Production exception handlers registered — see TestDistillEndpointErrors.
        from fastapi.testclient import TestClient
        from training.router import router

        from tests.conftest import build_test_app

        return TestClient(build_test_app(router), raise_server_exceptions=False)

    @staticmethod
    def _fake_provider():
        import numpy as np

        class _Logits:
            def __init__(self, arr):
                self.data = arr

        class _Model:
            def forward(self, input_ids, targets=None):
                # Logits dim is larger than the token vocab so the
                # [..., :vocab_size] slice in the teacher wrapper is a no-op.
                return _Logits(
                    np.full((input_ids.shape[0], input_ids.shape[1], 640), 0.5, dtype=np.float32)
                ), None

        class _Provider:
            model_id = "gpt2"

            def tokenize(self, text):
                # >=352 tokens to yield the 10-sample minimum with block=64.
                return list(range(1, 600))

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

        datasets_dir = _REPO_ROOT / "data"
        ds_dir = datasets_dir / "_test_distill_slonet"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("alpha beta gamma delta epsilon zeta eta theta " * 8)

        fake_core = MagicMock()
        fake_core.model.get.return_value = self._fake_provider()

        ckpt = _REPO_ROOT / "models" / "auto-training" / "_test_slonet_distill_distilled.soul"
        job_id = None
        try:
            client = self._make_client()
            with (
                patch(
                    "domain.infrastructure._internal.model_registry.get_model_registry",
                    return_value=None,
                ),
                patch(
                    "domain.infrastructure._internal.server_state.get_server_state",
                    return_value=fake_core,
                ),
            ):
                resp = client.post(
                    "/training/distill",
                    json={
                        "teacher_model": "gpt2",
                        "dataset": "_test_distill_slonet",
                        "epochs": 1,
                        "name": "_test_slonet_distill",
                    },
                )
                assert resp.status_code == 200
                job_id = resp.json()["job_id"]
                # Keep the patches active while the background training thread
                # runs — it resolves the teacher via get_server_state() and
                # would otherwise read the real ServerState singleton.
                assert self._poll(training_jobs, job_id) == "completed"
            job = training_jobs[job_id]
            assert job["status"] == "completed"
            assert "train_loss" in job
            assert job["train_loss"] >= 0
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

        datasets_dir = _REPO_ROOT / "data"
        ds_dir = datasets_dir / "_test_distill_slonet_missing"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("alpha beta gamma delta epsilon zeta eta theta " * 8)

        fake_core = MagicMock()
        fake_core.model.get.return_value = None  # no provider loaded

        job_id = None
        try:
            client = self._make_client()
            with (
                patch(
                    "domain.infrastructure._internal.model_registry.get_model_registry",
                    return_value=None,
                ),
                patch(
                    "domain.infrastructure._internal.server_state.get_server_state",
                    return_value=fake_core,
                ),
            ):
                resp = client.post(
                    "/training/distill",
                    json={
                        "teacher_model": "gpt2",
                        "dataset": "_test_distill_slonet_missing",
                        "epochs": 1,
                        "name": "_test_slonet_missing",
                    },
                )
                assert resp.status_code == 200
                job_id = resp.json()["job_id"]
                assert self._poll(training_jobs, job_id) == "failed"
            assert "not loaded" in training_jobs[job_id]["error"]
        finally:
            if job_id:
                training_jobs.pop(job_id, None)
            shutil.rmtree(ds_dir, ignore_errors=True)


# ── Data size caps (card 20261126_093028) ──────────────────────────────


class TestDataSizeCaps:
    """``SLO_DISTILL_MAX_DATA_BYTES`` + full-text tokenization (no 100k-char slice).

    Two stacked silent caps made the formalized distill corpus unusable:
    a 200 KB file gate rejected it outright (495 KB), and a 100k-char slice
    taken before tokenization would have ignored 80% of whatever fit.
    """

    def _make_client(self):
        from fastapi.testclient import TestClient
        from training.router import router

        from tests.conftest import build_test_app

        return TestClient(build_test_app(router), raise_server_exceptions=False)

    @staticmethod
    def _write_dataset(name: str, content: str):
        ds_dir = _REPO_ROOT / "data" / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text(content, encoding="utf-8")
        return ds_dir

    def test_low_env_cap_rejects_oversized_dataset(self, monkeypatch):
        """The env cap is the ENFORCED number — the message proves which cap fired."""
        import shutil

        monkeypatch.setenv("SLO_DISTILL_MAX_DATA_BYTES", "100000")
        ds_dir = self._write_dataset("_test_distill_cap_low", "Hello world " * 25_000)
        try:
            client = self._make_client()
            resp = client.post(
                "/training/distill",
                json={"teacher_model": "gpt2", "dataset": "_test_distill_cap_low", "epochs": 1},
            )
            assert resp.status_code >= 400
            assert "too large" in resp.text
            assert "100000" in resp.text, "the env cap, not the old hardcoded 200000, must fire"
        finally:
            shutil.rmtree(ds_dir, ignore_errors=True)

    def test_default_cap_fits_corpus_scale(self, monkeypatch):
        """300 KB (corpus scale) must pass the size gate — the old 200 KB cap rejected it."""
        import shutil

        from training.router import training_jobs

        monkeypatch.delenv("SLO_DISTILL_MAX_DATA_BYTES", raising=False)
        ds_dir = self._write_dataset("_test_distill_cap_default", "Hello world " * 25_000)
        job_id = None
        try:
            client = self._make_client()
            resp = client.post(
                "/training/distill",
                json={"teacher_model": "gpt2", "dataset": "_test_distill_cap_default", "epochs": 1},
            )
            assert "too large" not in resp.text, "default cap must admit corpus-scale data"
            assert resp.status_code == 200, resp.text
            job_id = resp.json().get("job_id")
        finally:
            if job_id:
                training_jobs.pop(job_id, None)
            shutil.rmtree(ds_dir, ignore_errors=True)

    def test_cap_env_fails_safe(self, monkeypatch):
        """Malformed/non-positive env falls back to the default (SLO_PREFILL_MAX_TOKENS precedent)."""
        from training.distill import _max_data_read_bytes

        monkeypatch.delenv("SLO_DISTILL_MAX_DATA_BYTES", raising=False)
        assert _max_data_read_bytes() == 4_000_000
        monkeypatch.setenv("SLO_DISTILL_MAX_DATA_BYTES", "not-a-number")
        assert _max_data_read_bytes() == 4_000_000
        monkeypatch.setenv("SLO_DISTILL_MAX_DATA_BYTES", "-5")
        assert _max_data_read_bytes() == 4_000_000
        monkeypatch.setenv("SLO_DISTILL_MAX_DATA_BYTES", "12345")
        assert _max_data_read_bytes() == 12345

    def test_tokenizer_sees_full_text(self):
        """No pre-tokenization slice — every char inside the byte cap reaches the tokenizer."""
        from training.distill import _tokenize_training_text

        text = "abcdefghij" * 15_000  # 150_000 chars — 1.5x the old 100k slice
        tokens = _tokenize_training_text(text, teacher_tokenizer=None, slonet_provider=None)
        assert len(tokens) == len(text)
