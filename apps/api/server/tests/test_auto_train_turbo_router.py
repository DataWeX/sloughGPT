"""
Tests for the /auto-train/start-turbo + /auto-train/turbo/status endpoints.

The turbo trainer runs SloughGPTTrainer on a background daemon thread and
publishes live telemetry (global_step, total_steps, steps_per_sec, eta_s,
elapsed_s) into module-level ``_turbo_state``. Only the auto_train router is
registered; the real trainer is replaced with a fake whose ``train()`` invokes
the ``on_progress`` callback so the progress plumbing is exercised end to end.
"""

import threading
import time

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.auto_train as mod
from routers.auto_train import router as auto_train_router

app = FastAPI()
app.include_router(auto_train_router)
client = TestClient(app)


_IDLE = {
    "status": "idle",
    "job_id": None,
    "global_step": 0,
    "total_steps": 0,
    "progress": 0.0,
    "loss": None,
    "learning_rate": None,
    "steps_per_sec": None,
    "eta_s": None,
    "elapsed_s": None,
    "result": None,
    "error": None,
}


@pytest.fixture(autouse=True)
def _reset_turbo(tmp_path):
    """Reset the module-level turbo state/events and redirect the output dir."""
    mod._turbo_state.update(_IDLE)
    mod._turbo_cancel_event = threading.Event()
    with patch.object(mod._auto_train_instance, "REPO_ROOT", tmp_path):
        yield


def _wait_for_status(desired, timeout=5.0):
    """Poll GET /auto-train/turbo/status until it reaches ``desired``."""
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get("/auto-train/turbo/status").json()
        if body["status"] == desired:
            return body
        time.sleep(0.05)
    raise AssertionError(f"status never became {desired!r}; last={body!r}")


def _fake_train_result(blocker=None, error=None, cancel=False):
    def _train(**kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress({
                "global_step": 42,
                "total_steps": 500,
                "progress_percent": 8.4,
                "train_loss": 2.31,
                "learning_rate": 0.0003,
                "steps_per_sec": 4.25,
                "eta_s": 98,
                "elapsed_s": 20,
            })
        if cancel:
            kwargs.get("cancel_event", threading.Event()).set()
        if blocker is not None:
            blocker.wait(timeout=5)
        if error is not None:
            raise error
        return {"status": "ok", "final_loss": 0.5, "total_steps": 500,
                "model_path": "/tmp/fake.soul"}
    return _train


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_status_idle(_):
    body = client.get("/auto-train/turbo/status").json()
    assert body["status"] == "idle"
    assert body["job_id"] is None
    assert body["global_step"] == 0


def test_start_turbo_missing_data():
    resp = client.post("/auto-train/start-turbo", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert "data_path" in body["message"]


def test_start_turbo_rejects_when_running():
    mod._turbo_state["status"] = "running"
    mod._turbo_state["job_id"] = "turbo_123"
    resp = client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    body = resp.json()
    assert body["status"] == "error"
    assert "already running" in body["message"]


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_start_turbo_validates_epochs(MockTrainer):
    resp = client.post("/auto-train/start-turbo", json={"data_path": "x.txt", "epochs": 0})
    assert resp.status_code == 422
    MockTrainer.assert_not_called()


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_progress_and_complete(MockTrainer):
    blocker = threading.Event()
    MockTrainer.return_value.train.side_effect = _fake_train_result(blocker=blocker)

    resp = client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["job_id"]

    running = _wait_for_status("running")
    assert running["job_id"] == body["job_id"]
    assert running["global_step"] == 42
    assert running["total_steps"] == 500
    assert running["progress"] == 8.4
    assert running["loss"] == 2.31
    assert running["learning_rate"] == 0.0003
    assert running["steps_per_sec"] == 4.25
    assert running["eta_s"] == 98
    assert running["elapsed_s"] == 20

    blocker.set()
    complete = _wait_for_status("complete")
    assert complete["progress"] == 100.0
    assert complete["result"]["final_loss"] == 0.5
    assert complete["result"]["total_steps"] == 500


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_cancelled(MockTrainer):
    MockTrainer.return_value.train.side_effect = _fake_train_result(cancel=True)
    client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    body = _wait_for_status("error")
    assert body["error"] == "Training cancelled"


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_trainer_error(MockTrainer):
    MockTrainer.return_value.train.side_effect = _fake_train_result(error=RuntimeError("boom"))
    client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    body = _wait_for_status("error")
    assert body["error"] == "boom"
