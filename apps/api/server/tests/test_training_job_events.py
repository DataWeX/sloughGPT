"""Subscribe-only job-events stream — descriptor + envelope contract.

Pins the vocabulary the web client aggregates against (card
``20261120_110356``): stream name ``training``, the closed phase set, the
status mapping, the ``init`` snapshot shape, and the fact that the store's
mutations reach the wire without any poll. Changing any of these breaks the
client contract — update both sides together.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from training.job_store import JobStore
from training.runtime import TrainingRuntime


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture
def runtime(store):
    return TrainingRuntime(store=store, grace_timeout_s=0.2)


def _job(jid: str, status: str = "running", **overrides) -> dict:
    base = {
        "id": jid,
        "name": "run",
        "model": "sloughgpt",
        "dataset": "test",
        "status": status,
        "progress": 0.0,
        "epochs": 3,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "eval_loss": None,
        "checkpoint": None,
        "error": None,
    }
    base.update(overrides)
    return base


# ── Descriptor projection ───────────────────────────────────────────────


def test_descriptor_registered():
    import training.job_events  # noqa: F401 — import registers the descriptor
    from infrastructure.contract import REGISTRY

    entry = next((e for e in REGISTRY.routes() if e["name"] == "training.job_events"), None)
    assert entry is not None, "descriptor 'training.job_events' not registered"
    assert entry["path"] == "/training/job-events/stream"
    assert entry["method"] == "GET"
    assert entry["idempotent"] is True
    assert entry["auth_scope"] == "authenticated"
    assert entry["params"]["properties"] == {}
    assert entry["params"]["required"] == []


# ── Envelope contract ───────────────────────────────────────────────────


def test_envelope_shape_and_status_mapping():
    from training.job_events import _envelope, _init_frame

    frame = _envelope({"phase": "progress", "job": {"id": "j1", "loss": 1.0}})
    assert set(frame) == {"stream", "phase", "status", "data", "meta", "message"}
    assert frame["stream"] == "training"
    assert frame["phase"] == "progress"
    assert frame["status"] == "working"
    assert frame["data"]["loss"] == 1.0

    assert _envelope({"phase": "completed", "job": {}})["status"] == "complete"
    assert _envelope({"phase": "recovered", "job": {}})["status"] == "success"
    for phase in ("failed", "cancelled", "interrupted", "abandoned"):
        assert _envelope({"phase": phase, "job": {}})["status"] == "error"

    init = _init_frame([{"id": "j1"}, {"id": "j2"}])
    assert init["stream"] == "training"
    assert init["phase"] == "init"
    assert init["status"] == "working"
    assert init["data"] == {"jobs": [{"id": "j1"}, {"id": "j2"}]}


def test_store_emits_only_contracted_phases(runtime):
    """Every phase the store can publish is in the pinned vocabulary."""
    from training.job_events import PHASES

    events: list[dict] = []
    runtime.subscribe(events.append)
    runtime.register("j1", _job("j1"), None)
    rec = runtime.get("j1")
    rec.update({"progress": 50, "global_step": 2, "loss": 2.0})
    runtime.sync("j1")
    for status in ("completed", "failed", "cancelled", "interrupted"):
        rec["status"] = status
        runtime.sync("j1")

    phases = {e["phase"] for e in events}
    assert phases == {"started", "progress", "completed", "failed", "cancelled", "interrupted"}
    assert phases <= PHASES


def test_progress_event_carries_poll_only_detail_fields(runtime):
    """loss/global_step/eta/steps-per-sec reach subscribers without a poll."""
    runtime.register("j1", _job("j1"), None)
    events: list[dict] = []
    runtime.subscribe(events.append)
    runtime.get("j1").update(
        {"progress": 42, "global_step": 7, "loss": 1.5, "eta_s": 12.5, "steps_per_sec": 30.0}
    )
    runtime.sync("j1")

    assert [e["phase"] for e in events] == ["progress"]
    job = events[0]["job"]
    for key in ("progress", "global_step", "loss", "eta_s", "steps_per_sec"):
        assert key in job, f"{key} missing from published job view"
    assert job["loss"] == 1.5
    assert job["eta_s"] == 12.5


# ── Wire: init first, then store-pushed progress ────────────────────────


def _get_client(runtime, monkeypatch):
    import training.job_events as je

    monkeypatch.setattr(je, "get_training_runtime", lambda: runtime)
    app = FastAPI()
    app.include_router(je.router)
    return TestClient(app)


def _frames(resp) -> list[dict]:
    """Parse ``data:`` frames out of a completed SSE response body."""
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def test_endpoint_serves_init_snapshot_first(runtime, monkeypatch):
    """Connect → ``init`` frame with every tracked job, no poll involved.

    TestClient runs the app to completion, so the generator's own
    disconnect check is driven directly (the same fake-request trick the
    ``live_stream`` tests use): first check after the init frame ends it.
    """
    from unittest.mock import patch

    runtime.register("j1", _job("j1", name="alpha"), None)
    client = _get_client(runtime, monkeypatch)

    async def _disconnect_after_init(_self):
        return True

    with patch("starlette.requests.Request.is_disconnected", _disconnect_after_init):
        resp = client.get("/training/job-events/stream")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _frames(resp)
    assert [f["phase"] for f in frames] == ["init"]
    assert frames[0]["stream"] == "training"
    assert frames[0]["data"]["jobs"][0]["id"] == "j1"
    assert frames[0]["data"]["jobs"][0]["name"] == "alpha"
    # Disconnect tears the subscription down.
    assert runtime._subscribers == {}, "subscription leaked after client disconnect"


def test_endpoint_pushes_progress_to_a_connected_client(runtime, monkeypatch):
    """A store mutation reaches an already-connected client with no request.

    A helper thread stands in for the training worker: it mutates the store
    while the (blocking) TestClient call is in flight, and the frame must
    come back down the same open stream.
    """
    import threading
    from unittest.mock import patch

    runtime.register("j1", _job("j1"), None)
    client = _get_client(runtime, monkeypatch)

    def _mutate_store():
        time.sleep(0.15)  # let the request subscribe first
        runtime.get("j1").update({"progress": 10, "global_step": 3, "loss": 2.5})
        runtime.sync("j1")

    checks = {"n": 0}

    async def _disconnect_after_progress(_self):
        checks["n"] += 1
        return checks["n"] >= 2  # 1st check: keep reading, 2nd: done

    worker = threading.Thread(target=_mutate_store, daemon=True)
    with (
        patch("starlette.requests.Request.is_disconnected", _disconnect_after_progress),
        patch("training.job_events.HEARTBEAT_INTERVAL_S", 0.5),
    ):
        worker.start()
        resp = client.get("/training/job-events/stream")

    phases = [f["phase"] for f in _frames(resp)]
    assert phases[0] == "init"
    assert "progress" in phases, f"progress frame never arrived: {phases}"
    progress = next(f for f in _frames(resp) if f["phase"] == "progress")
    assert progress["stream"] == "training"
    assert progress["data"]["loss"] == 2.5
    assert progress["data"]["global_step"] == 3
    assert runtime._subscribers == {}, "subscription leaked after client disconnect"
