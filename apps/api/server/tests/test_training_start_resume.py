"""``POST /training/start`` honors ``checkpoint_name`` (card 20261120_124701).

The contract was declared but dead on the tracked-job path: ``TrainingRequest.
checkpoint_name`` is documented "Resume from checkpoint" (schemas.py) while
``start_training`` never read it, so every start trained from scratch — the
same defect class as a route that is registered in the descriptor but never
served.

Resolution happens in the request handler BEFORE any job exists: a bad name is
a 4xx with no job row and no worker spin-up, and the pre-loaded bundle is
handed to ``train(resume=True, resume_checkpoint=bundle)`` so the worker
thread never re-reads the file — the contract
``test_training_recovery_router.py`` already pins for the recovery path.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from training import feeds
from training.execution import router
from training.jobs import training_jobs

app = FastAPI()
register_app_error_handler(app)
app.include_router(router)
client = TestClient(app)

FEED_BODY = {
    "name": "resume journey",
    "model": "slonet",
    "dataset": "feed:api-conversations",
}


@pytest.fixture(autouse=True)
def _feed_sources(tmp_path, monkeypatch) -> Iterator[Path]:
    """Point the feed at a temp corpus so pre-flight never touches the repo."""
    corpus = tmp_path / "corpus.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "first user query"},
            {"role": "assistant", "content": "first assistant reply"},
        ],
        "meta": {"model": "test", "captured_at": "2026-11-20T00:00:00+00:00"},
    }
    corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        feeds,
        "_SOURCES",
        {"api-conversations": str(corpus), "feedback": str(tmp_path / "missing.jsonl")},
    )
    yield corpus


class _FakeTrainer:
    """Records construction + every ``train()`` kwargs dict; runs nothing."""

    trains: list[dict] = []

    def __init__(self, data_path, **kwargs) -> None:  # noqa: ARG002 - stub
        pass

    def train(self, **kwargs) -> dict:
        _FakeTrainer.trains.append(kwargs)
        return {"best_eval_loss": 0.5}

    def save(self, path: str) -> None:  # noqa: ARG002 - must not touch disk
        pass


class _FakeTracker:
    def end_run(self) -> None:
        pass


async def _async_noop(*args, **kwargs) -> None:  # noqa: ARG002
    return None


@pytest.fixture(autouse=True)
def _fakes(monkeypatch) -> Iterator[None]:
    from training import execution as exec_mod

    import domain.training as domain_training
    from domain.training._internal import wandb_helpers

    monkeypatch.setattr(domain_training, "SloughGPTTrainer", _FakeTrainer)
    monkeypatch.setattr(
        wandb_helpers, "create_training_tracker_for_api_job", lambda **kw: _FakeTracker()
    )
    monkeypatch.setattr(exec_mod, "notify_training_event", _async_noop)
    monkeypatch.setattr(exec_mod, "notify_push", lambda **kw: None)
    _FakeTrainer.trains.clear()
    yield
    _FakeTrainer.trains.clear()


def _start(**overrides) -> dict:
    resp = client.post("/training/start", json={**FEED_BODY, **overrides})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    deadline = time.time() + 5.0
    status = None
    while time.time() < deadline:
        status = training_jobs.get(job_id, {}).get("status")
        if status in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert status == "completed", f"job did not complete: {status}"
    assert _FakeTrainer.trains, "trainer.train() was never called"
    return _FakeTrainer.trains[-1]


def test_start_without_checkpoint_name_trains_fresh():
    """No checkpoint_name -> no resume: the default path must not regress."""
    kwargs = _start()
    assert kwargs.get("resume") is False
    assert kwargs.get("resume_checkpoint") is None


def test_unknown_checkpoint_is_404_before_any_job_exists():
    """A name that resolves to nothing fails loudly, with zero side effects."""
    before = set(training_jobs)
    resp = client.post(
        "/training/start", json={**FEED_BODY, "checkpoint_name": "does-not-exist-xyz"}
    )
    assert resp.status_code == 404
    assert "Checkpoint not found" in resp.text
    assert set(training_jobs) == before
    assert _FakeTrainer.trains == []


@pytest.mark.parametrize("name", ["../../etc/passwd", "/etc/passwd"])
def test_checkpoint_name_cannot_escape_the_checkpoint_roots(name):
    """Path traversal is rejected as not-found — the roots are the boundary."""
    before = set(training_jobs)
    resp = client.post("/training/start", json={**FEED_BODY, "checkpoint_name": name})
    assert resp.status_code == 404
    assert set(training_jobs) == before
    assert _FakeTrainer.trains == []


def test_checkpoint_name_resumes_with_the_preloaded_bundle(monkeypatch):
    """The named checkpoint is resolved once, in the handler, and handed over.

    Identity (``is``) rather than equality: the worker must consume the very
    bundle the handler loaded — no second disk read, no silent substitution.
    """
    from training import execution as exec_mod

    from domain.training._internal.state import CHECKPOINTS_DIR
    from domain.training.engine import get_training_engine

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS_DIR / f"zz_resume_test_{uuid.uuid4().hex}.soul"
    ckpt.write_bytes(b"placeholder")
    bundle = {"model_state_dict": {}, "step": 7}
    seen: dict[str, str] = {}
    try:

        def _resolve(path: str, directory: str) -> tuple[str, dict]:
            seen["path"], seen["dir"] = path, directory
            return path, bundle

        # Instance patch: the engine is a singleton and an instance attribute
        # shadows the bound method without re-binding ``self``.
        monkeypatch.setattr(get_training_engine(), "resolve_resume_checkpoint", _resolve)
        monkeypatch.setattr(exec_mod, "_find_start_checkpoint", lambda name: ckpt)

        kwargs = _start(checkpoint_name=ckpt.name)
    finally:
        ckpt.unlink(missing_ok=True)

    assert kwargs.get("resume") is True
    assert kwargs.get("resume_checkpoint") is bundle
    assert seen["path"] == str(ckpt)
