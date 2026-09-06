from infrastructure.exception_handlers import register_app_error_handler

"""
Tests for the POST /training/recovery/recover/{job_id} endpoint.

Covers the checkpoint resolution contract: an explicit recorded path is loaded
strictly in the request handler (corrupt/missing -> 422, before any job is
created), a fallback uses ``load_latest_with_path`` (skip-corrupt), and the
pre-loaded bundle is handed to ``train(resume=True, resume_checkpoint=bundle)``
so no second load happens in the worker thread.
"""

import contextlib
import importlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

router_mod = importlib.import_module("training.router")

app = FastAPI()
register_app_error_handler(app)
app.include_router(router_mod.router)
client = TestClient(app)


class _FakeJobStore:
    """In-memory job store satisfying the recovery endpoint's needs.

    Records every mutation in ``calls`` (method, args, kwargs) so tests can
    assert the persistent-lifecycle contract (terminal writes target the
    original job's store row, never the ephemeral recovery id).
    """

    def __init__(self, job):
        self._job = job
        self.calls = []

    def _record(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))

    def get(self, job_id):
        return self._job if job_id == "job-1" else None

    def update(self, *args, **kwargs):
        self._record("update", args, kwargs)

    def update_progress(self, *args, **kwargs):
        self._record("update_progress", args, kwargs)

    def mark_completed(self, *args, **kwargs):
        self._record("mark_completed", args, kwargs)

    def mark_failed(self, *args, **kwargs):
        self._record("mark_failed", args, kwargs)

    def mark_recovering(self, *args, **kwargs):
        self._record("mark_recovering", args, kwargs)

    @staticmethod
    def is_stale_heartbeat(job):
        from training.job_store import JobStore

        return JobStore.is_stale_heartbeat(job)


class _FakeController:
    def start(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        pass

    def fail(self, *args, **kwargs):
        pass


class _SyncExecutor:
    """Runs submitted recovery threads synchronously for deterministic asserts."""

    def __init__(self):
        self.submitted = []

    def submit(self, fn, job_id):
        self.submitted.append((fn, job_id))
        fn(job_id)


def _base_job(checkpoint_dir, checkpoint_path=""):
    return {
        "id": "job-1",
        "name": "train_job",
        "status": "interrupted",
        "config": {},
        "data_path": "/tmp/corpus.txt",
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": checkpoint_dir,
    }


@pytest.fixture
def deps():
    """Wire fakes into the router module and yield the handles to assert on."""
    executor = _SyncExecutor()
    trainer_cls = MagicMock()
    trainer_inst = trainer_cls.return_value
    trainer_inst._last_checkpoint_path = None
    trainer_inst.train.return_value = {"success": True, "global_step": 7}

    with (
        patch.object(router_mod, "get_training_executor", return_value=executor),
        patch.object(router_mod, "get_training_controller", return_value=_FakeController()),
        patch.object(router_mod, "notify_training_event", new=MagicMock()),
        patch(
            "domains.training.train_pipeline.SloughGPTTrainer",
            new=trainer_cls,
        ),
    ):
        yield executor, trainer_inst


def _recover(tmp_path, job, patches=()):
    store = _FakeJobStore(job)
    ctxs = [patch.object(router_mod, "get_job_store", return_value=store)]
    ctxs.extend(patches)
    with contextlib.ExitStack() as stack:
        for ctx in ctxs:
            stack.enter_context(ctx)
        resp = client.post("/recovery/recover/job-1")
    resp._store = store
    return resp


# ── Validation ──────────────────────────────────────────────────────────────


def test_recover_404_when_job_missing(tmp_path, deps):
    resp = _recover(tmp_path, None)
    assert resp.status_code == 404
    assert resp.json()["error"] == "Job not found"
    assert deps[0].submitted == []


def test_recover_400_when_status_not_interruptible(tmp_path, deps):
    job = _base_job(str(tmp_path))
    job["status"] = "completed"
    resp = _recover(tmp_path, job)
    assert resp.status_code == 400
    assert "only 'interrupted' or 'failed'" in resp.json()["error"]
    assert deps[0].submitted == []


def test_recover_400_when_recovering_with_fresh_heartbeat(tmp_path, deps):
    from datetime import datetime

    job = _base_job(str(tmp_path))
    job["status"] = "recovering"
    job["last_heartbeat"] = datetime.now().isoformat()
    resp = _recover(tmp_path, job)
    assert resp.status_code == 400
    assert "stale heartbeat" in resp.json()["error"]
    assert deps[0].submitted == []


def test_recover_allows_stale_recovering_job(tmp_path, deps):
    from datetime import datetime, timedelta

    job = _base_job(str(tmp_path))
    job["status"] = "recovering"
    job["last_heartbeat"] = (datetime.now() - timedelta(seconds=600)).isoformat()
    resp = _recover(tmp_path, job)
    assert resp.status_code == 200
    assert ("mark_recovering", ("job-1",), {}) in resp._store.calls
    router_mod.training_jobs.pop("recovery_job-1", None)


# ── Recorded path: strict, fail-loud, no job created on failure ─────────────


def test_recover_corrupt_recorded_path_422_no_job(tmp_path, deps):
    from domains.training.train_pipeline import CheckpointManager

    bad = tmp_path / "ck" / "corrupt.soul"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"garbage")
    job = _base_job(str(tmp_path), checkpoint_path=str(bad))
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "is_resumable", return_value=True),
            patch.object(CheckpointManager, "load_from_path", side_effect=ValueError("boom")),
        ],
    )
    assert resp.status_code == 422
    assert "Cannot resume from" in resp.json()["error"]
    assert "recovery_job-1" not in router_mod.training_jobs
    assert deps[0].submitted == []


def test_recover_missing_recorded_path_422_no_job(tmp_path, deps):
    from domains.training.train_pipeline import CheckpointManager

    missing = str(tmp_path / "ck" / "nope.soul")
    job = _base_job(str(tmp_path), checkpoint_path=missing)
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "is_resumable", return_value=True),
            patch.object(CheckpointManager, "load_from_path", return_value=None),
        ],
    )
    assert resp.status_code == 422
    assert "missing or unsupported" in resp.json()["error"]
    assert "recovery_job-1" not in router_mod.training_jobs
    assert deps[0].submitted == []


def test_recover_recorded_path_missing_on_disk_422(tmp_path, deps):
    # A recorded path whose file does not exist on disk must fail loudly (422)
    # via the real is_resumable check — never silently fall back to resuming
    # from a different checkpoint. Regression: pre-fix code fell through to
    # load_latest_with_path() and started a fresh job on an empty directory.
    missing = str(tmp_path / "ck" / "gone.soul")
    job = _base_job(str(tmp_path), checkpoint_path=missing)
    resp = _recover(tmp_path, job)
    assert resp.status_code == 422
    assert "missing or unsupported" in resp.json()["error"]
    assert "recovery_job-1" not in router_mod.training_jobs
    assert deps[0].submitted == []


# ── Recorded path: valid bundle handed to train() exactly once ──────────────


def test_recover_valid_recorded_path_resumes_with_bundle(tmp_path, deps):
    executor, trainer_inst = deps
    from domains.training.train_pipeline import CheckpointManager

    ckpt = str(tmp_path / "ck" / "model_100.soul")
    bundle = {"step": 7, "epoch": 2, "model_state_dict": {}}
    job = _base_job(str(tmp_path), checkpoint_path=ckpt)
    job["global_step"] = 7
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "is_resumable", return_value=True),
            patch.object(CheckpointManager, "load_from_path", return_value=bundle),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recovered"
    assert body["checkpoint_path"] == ckpt

    rec = router_mod.training_jobs["recovery_job-1"]
    assert rec["checkpoint_path"] == ckpt
    assert rec["status"] == "completed"
    assert rec["global_step"] == 7

    call = trainer_inst.train.call_args
    assert call is not None
    assert call.kwargs["resume"] is True
    assert call.kwargs["resume_checkpoint"] is bundle
    assert call.kwargs.get("resume_path") is None
    assert call.kwargs["on_progress"] is not None


# ── Fallback: no recorded path uses load_latest_with_path ───────────────────


def test_recover_fallback_no_checkpoint_starts_fresh(tmp_path, deps):
    executor, trainer_inst = deps
    from domains.training.train_pipeline import CheckpointManager

    job = _base_job(str(tmp_path), checkpoint_path="")
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "load_latest_with_path", return_value=(None, None)),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint_path"] == ""
    assert "beginning" in body["message"]

    rec = router_mod.training_jobs["recovery_job-1"]
    assert rec["checkpoint_path"] == ""

    call = trainer_inst.train.call_args
    assert call.kwargs["resume"] is True
    assert call.kwargs["resume_checkpoint"] is None


def test_recover_checkpoint_dir_from_job_config(tmp_path, deps):
    # The store's checkpoint_dir column is NULL for interrupted jobs; the scan
    # directory must come from the job's stored request config, not the
    # hardcoded "checkpoints" default. Regression: pre-fix code scanned the
    # wrong directory for real recovered jobs.
    executor, trainer_inst = deps
    from domains.training.train_pipeline import CheckpointManager

    custom_dir = str(tmp_path / "custom")
    job = _base_job(str(tmp_path), checkpoint_path="")
    job.pop("checkpoint_dir")
    job["config"] = {"checkpoint_dir": custom_dir}
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "load_latest_with_path", return_value=(None, None)),
        ],
    )

    assert resp.status_code == 200
    rec = router_mod.training_jobs["recovery_job-1"]
    assert rec["checkpoint_dir"] == custom_dir


def test_recover_fallback_uses_latest_bundle(tmp_path, deps):
    executor, trainer_inst = deps
    from domains.training.train_pipeline import CheckpointManager

    latest = str(tmp_path / "ck" / "model_200.soul")
    bundle = {"step": 5, "epoch": 1, "model_state_dict": {}}
    job = _base_job(str(tmp_path), checkpoint_path="")
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "load_latest_with_path", return_value=(latest, bundle)),
        ],
    )

    assert resp.status_code == 200
    assert resp.json()["checkpoint_path"] == latest

    rec = router_mod.training_jobs["recovery_job-1"]
    assert rec["checkpoint_path"] == latest

    call = trainer_inst.train.call_args
    assert call.kwargs["resume_checkpoint"] is bundle


# ── Persistent store lifecycle ───────────────────────────────────────────────


def test_recover_success_records_completion_on_original_job(tmp_path, deps):
    # Terminal writes must target the original job's durable row, never the
    # ephemeral recovery id (which has no store row — those writes were silent
    # no-ops that also dropped the produced checkpoint path).
    from domains.training.train_pipeline import CheckpointManager

    ckpt = str(tmp_path / "ck" / "model_100.soul")
    job = _base_job(str(tmp_path), checkpoint_path=ckpt)
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "is_resumable", return_value=True),
            patch.object(
                CheckpointManager, "load_from_path", return_value={"model_state_dict": {}}
            ),
        ],
    )
    assert resp.status_code == 200

    store = resp._store
    assert ("mark_completed", ("job-1", ckpt), {}) in store.calls
    assert not any(
        method == "mark_completed" and args[0].startswith("recovery_")
        for method, args, kwargs in store.calls
    )


def test_recover_failure_marks_original_job_failed(tmp_path, deps):
    # A failing recovery must mark the ORIGINAL job failed — otherwise its row
    # stayed "recovering" forever and was never recoverable or visible again.
    executor, trainer_inst = deps
    from domains.training.train_pipeline import CheckpointManager

    job = _base_job(str(tmp_path), checkpoint_path="")
    trainer_inst.train.side_effect = RuntimeError("boom")
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "load_latest_with_path", return_value=(None, None)),
        ],
    )
    assert resp.status_code == 200

    store = resp._store
    assert ("mark_failed", ("job-1", "boom"), {}) in store.calls
    assert not any(
        method == "mark_failed" and args[0].startswith("recovery_")
        for method, args, kwargs in store.calls
    )
    assert router_mod.training_jobs["recovery_job-1"]["status"] == "failed"


def test_recover_cancel_restores_interrupted(tmp_path, deps):
    # A cancelled recovery leaves the job recoverable again (status restored to
    # "interrupted"), not wrongly marked "recovered".
    executor, trainer_inst = deps
    from domains.training.train_pipeline import CheckpointManager

    def _set_cancel(**kwargs):
        kwargs["cancel_event"].set()
        return {"success": True, "global_step": 0}

    job = _base_job(str(tmp_path), checkpoint_path="")
    trainer_inst.train.side_effect = _set_cancel
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "load_latest_with_path", return_value=(None, None)),
        ],
    )
    assert resp.status_code == 200

    store = resp._store
    assert ("update", ("job-1",), {"status": "interrupted"}) in store.calls
    assert not any(method == "mark_completed" for method, args, kwargs in store.calls)
    assert router_mod.training_jobs["recovery_job-1"]["status"] == "cancelled"


def test_recover_reuses_original_hyperparameters(tmp_path, deps):
    # The recovered run must continue with the ORIGINAL job's trainer
    # configuration (same builder as /training/start), not a fixed subset that
    # silently dropped LoRA/dropout/scheduler/device settings.
    from domains.training.train_pipeline import CheckpointManager

    tcls = MagicMock()
    job = _base_job(str(tmp_path), checkpoint_path="")
    job["config"] = {
        "use_lora": True,
        "dropout": 0.05,
        "lora_rank": 16,
        "scheduler": "linear",
        "learning_rate": 5e-4,
    }
    resp = _recover(
        tmp_path,
        job,
        patches=[
            patch.object(CheckpointManager, "load_latest_with_path", return_value=(None, None)),
            patch("domains.training.train_pipeline.SloughGPTTrainer", new=tcls),
        ],
    )
    assert resp.status_code == 200

    kw = tcls.call_args.kwargs
    assert kw["data_path"] == "/tmp/corpus.txt"
    assert kw["use_lora"] is True
    assert kw["dropout"] == 0.05
    assert kw["lora_rank"] == 16
    assert kw["scheduler_type"] == "linear"
    assert kw["lr"] == 5e-4
