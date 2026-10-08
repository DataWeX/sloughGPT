"""POST /training/start resumes from a checkpoint row — or refuses it.

``SloughGPTTrainer.train(resume=..., resume_path=...)`` has always existed and
``TrainingRequest.checkpoint_name`` has always claimed "Resume from checkpoint",
but the endpoint never connected them: a resume request started a FRESH run
silently. The contract now: a resolved row (name, optional path) becomes an
explicit resume on the trainer; an unknown checkpoint is refused in pre-flight
with no job record at all; no checkpoint keeps the fresh-run path untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from training import feeds
from training.jobs import training_jobs
from training.router import router

app = FastAPI()
register_app_error_handler(app)
app.include_router(router)
client = TestClient(app)


class _FakeTrainer:
    """Records train() kwargs; runs nothing (hermetic — no compute, no writes)."""

    train_kwargs: list[dict] = []

    def __init__(self, data_path, **kwargs) -> None:
        self.data_path = data_path
        self.kwargs = kwargs

    def train(self, **kwargs) -> dict:
        _FakeTrainer.train_kwargs.append(kwargs)
        return {"best_eval_loss": 0.5}

    def save(self, path: str) -> None:  # noqa: ARG002 - stub, must not touch disk
        pass


class _FakeTracker:
    def end_run(self) -> None:
        pass


async def _async_noop(*args, **kwargs) -> None:  # noqa: ARG002
    return None


@pytest.fixture(autouse=True)
def _feed_source(tmp_path, monkeypatch) -> Path:
    """A one-record corpus behind ``feed:api-conversations``.

    The feed spec short-circuits disk-dataset pre-flight, so the resume tests
    only exercise the checkpoint contract.
    """
    corpus = tmp_path / "corpus.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "first user query"},
            {"role": "assistant", "content": "first assistant reply"},
        ]
    }
    corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(feeds, "_SOURCES", {"api-conversations": str(corpus)})
    return corpus


@pytest.fixture(autouse=True)
def _ckpt_roots(tmp_path, monkeypatch) -> dict[str, Path]:
    """Send every checkpoint root to tmp_path instead of the real repo tree."""
    import domain.training._internal.checkpoints as ckpt_mod

    roots = {
        "CHECKPOINTS_DIR": tmp_path / "models" / "auto-training",
        "TURBO_DIR": tmp_path / "models" / "turbo-trained",
        "LORA_DIR": tmp_path / "data" / "user_adapters",
        "TRAINED_DIR": tmp_path / "models",
    }
    for name, p in roots.items():
        p.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(ckpt_mod, name, p)
    return roots


@pytest.fixture(autouse=True)
def _stub_trainer(monkeypatch) -> None:
    """Swap the real trainer/telemetry for recorders; clear prior kwargs."""
    from training import execution as exec_mod

    import domain.training as domain_training
    from domain.training._internal import wandb_helpers

    monkeypatch.setattr(domain_training, "SloughGPTTrainer", _FakeTrainer)
    monkeypatch.setattr(
        wandb_helpers, "create_training_tracker_for_api_job", lambda **kw: _FakeTracker()
    )
    monkeypatch.setattr(exec_mod, "notify_training_event", _async_noop)
    monkeypatch.setattr(exec_mod, "notify_push", lambda **kw: None)
    _FakeTrainer.train_kwargs.clear()


def _start(**overrides) -> dict:
    payload = {"name": "resume-run", "model": "slonet", "dataset": "feed:api-conversations"}
    payload.update(overrides)
    resp = client.post("/training/start", json=payload)
    if resp.status_code == 200:
        return resp.json()
    return {"__status": resp.status_code, **resp.json()}


def _wait_done(job_id: str, timeout: float = 5.0) -> str:
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = training_jobs.get(job_id, {}).get("status")
        if status in ("completed", "failed"):
            return status
        time.sleep(0.02)
    return status or "timeout"


class TestResumeFromCheckpoint:
    def test_resume_threads_resolved_path_into_train(self, _ckpt_roots):
        ckpt = _ckpt_roots["CHECKPOINTS_DIR"] / "resume-here.soul"
        ckpt.write_text("x" * 5000)

        out = _start(checkpoint_name="resume-here.soul", checkpoint_path=str(ckpt))
        assert out.get("status") == "started", out
        assert _wait_done(out["job_id"]) == "completed"

        assert _FakeTrainer.train_kwargs, "trainer was never invoked"
        kw = _FakeTrainer.train_kwargs[-1]
        assert kw["resume"] is True
        # The exact file the caller addressed — not whichever twin a
        # name-sweep across the roots would have hit first.
        assert kw["resume_path"] == str(ckpt.resolve())
        # Visible on the job record: fresh-vs-resumed must never be silent.
        assert training_jobs[out["job_id"]]["resume_from"] == str(ckpt.resolve())

    def test_unknown_checkpoint_is_refused_before_any_job(self):
        before = set(training_jobs)

        out = _start(checkpoint_name="ghost.soul")

        assert out["__status"] == 400, out
        assert set(training_jobs) == before, "a refused request must not register a job"
        assert _FakeTrainer.train_kwargs == [], "nothing may train after a refusal"

    def test_bad_path_is_refused_not_name_swept(self, _ckpt_roots):
        # The twin exists under the NAME, but the caller addressed a path
        # outside the roots: address beats name, and a bad address refuses.
        ckpt = _ckpt_roots["CHECKPOINTS_DIR"] / "twin.soul"
        ckpt.write_text("x" * 5000)
        outside = _ckpt_roots["CHECKPOINTS_DIR"].parent.parent / "elsewhere.soul"
        before = set(training_jobs)

        out = _start(checkpoint_name="twin.soul", checkpoint_path=str(outside))

        assert out["__status"] == 400, out
        assert set(training_jobs) == before
        assert _FakeTrainer.train_kwargs == []

    def test_fresh_run_is_unchanged_without_a_checkpoint(self):
        out = _start()
        assert out.get("status") == "started", out
        assert _wait_done(out["job_id"]) == "completed"

        kw = _FakeTrainer.train_kwargs[-1]
        assert not kw.get("resume")
        assert kw.get("resume_path") is None
        assert training_jobs[out["job_id"]]["resume_from"] is None
