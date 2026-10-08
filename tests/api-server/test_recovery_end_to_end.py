"""
End-to-end recovery journey: real FastAPI app, real JobStore, no fakes.

``test_training_recovery_router.py`` drives the handler with a fake store and a
sync executor. This file pins the wiring the UI actually depends on:

- a row that cannot be resumed is not offered by ``GET /recovery/recoverable``;
- ``POST /recovery/recover/{id}`` refuses it with a plain-language 422 *before*
  it writes any recovery state — the original bug spawned
  ``SloughGPTTrainer(data_path="")`` instead, which died in the data loader
  after the training phase had already flipped to ``error``;
- the error body is ``{"error": ...}``, the key ``apps/web/lib/http-client.ts``
  reads (``j.detail ?? j.message ?? j.error``) and hands to ``formatToastError``.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from training.job_store import JobStore

# ``from training import router`` yields the APIRouter object (the package
# re-exports ``.router.router``); the module that holds ``get_job_store`` needs
# to be reached through the module cache.
router_module = importlib.import_module("training.router")

app = FastAPI()
register_app_error_handler(app)
app.include_router(router_module.router)
client = TestClient(app)


def _seed(store, job_id, *, data_path, status="interrupted", config=None, dataset="corpus"):
    store.create(job_id, f"job {job_id}", config or {"model": "sloughgpt"}, dataset)
    store.update(job_id, data_path=data_path, status=status)
    return store.get(job_id)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A real JobStore on a throw-away path, wired where the router looks."""
    job_store = JobStore(str(tmp_path / "training_jobs.db"))
    monkeypatch.setattr(router_module, "get_job_store", lambda: job_store)
    return job_store


def test_row_without_a_dataset_is_not_offered(store):
    store.create("stale", "distill", {}, "")
    store.update("stale", data_path="", status="interrupted")

    resp = client.get("/recovery/recoverable")

    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "jobs": []}


def test_offers_only_rows_whose_dataset_still_exists(store, tmp_path):
    live = tmp_path / "corpus.txt"
    live.write_text("hello world\n" * 10, encoding="utf-8")
    _seed(store, "ok", data_path=str(live))
    _seed(store, "stale", data_path="")
    _seed(store, "gone", data_path=str(tmp_path / "deleted-corpus.txt"))

    resp = client.get("/recovery/recoverable")

    assert [job["id"] for job in resp.json()["jobs"]] == ["ok"]


def test_impossible_recovery_422s_before_any_state_is_written(store):
    store.create("stale", "distill", {}, "")
    store.update("stale", data_path="", status="interrupted")
    rows_before = len(store.list())

    resp = client.post("/recovery/recover/stale")

    assert resp.status_code == 422
    assert "No dataset recorded" in resp.json()["error"]
    # Nothing was created: no recovery row, no mark_recovering/mark_failed.
    assert len(store.list()) == rows_before
    assert store.get("recovery_stale") is None


def test_running_job_is_rejected_by_the_status_check(store, tmp_path):
    live = tmp_path / "corpus.txt"
    live.write_text("hello world\n" * 10, encoding="utf-8")
    _seed(store, "running", data_path=str(live), status="running")

    resp = client.post("/recovery/recover/running")

    assert resp.status_code == 400


def test_unknown_job_is_a_404(store):
    resp = client.post("/recovery/recover/does-not-exist")

    assert resp.status_code == 404


def test_recoverable_agrees_with_the_summary_flag(store, tmp_path):
    """The count the page shows and the list it renders must be the same set."""
    from training.jobs_api import _job_summary

    live = tmp_path / "corpus.txt"
    live.write_text("hello world\n" * 10, encoding="utf-8")
    _seed(store, "ok", data_path=str(live))
    _seed(store, "stale", data_path="")

    offered = {job["id"] for job in client.get("/recovery/recoverable").json()["jobs"]}
    summary_flags = {
        job["id"]: _job_summary(job)["recoverable"] for job in store.list() if job["id"] in offered
    }

    assert offered == {"ok"}
    assert summary_flags == {"ok": True}


def test_abandon_marks_the_row_and_takes_it_off_the_list(store, tmp_path):
    live = tmp_path / "corpus.txt"
    live.write_text("hello world\n" * 10, encoding="utf-8")
    _seed(store, "ok", data_path=str(live))
    assert client.get("/recovery/recoverable").json()["count"] == 1

    resp = client.delete("/recovery/abandon/ok")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "abandoned",
        "job_id": "ok",
        "message": "Job marked as abandoned",
    }
    assert store.get("ok")["status"] == "abandoned"
    assert client.get("/recovery/recoverable").json()["count"] == 0


def test_abandon_unknown_job_404s(store):
    resp = client.delete("/recovery/abandon/does-not-exist")

    assert resp.status_code == 404


def _iso(delta_seconds: float) -> str:
    """Heartbeat string the store writes, ``delta_seconds`` in the past."""
    return (datetime.now(UTC) - timedelta(seconds=delta_seconds)).isoformat().replace("+00:00", "Z")


def test_stats_count_matches_the_list_it_describes(store, tmp_path):
    """/recovery/stats feeds the KPI and /recovery/recoverable feeds the list."""
    live = tmp_path / "corpus.txt"
    live.write_text("hello world\n" * 10, encoding="utf-8")
    _seed(store, "ok", data_path=str(live))
    _seed(store, "stale", data_path="")
    _seed(store, "gone", data_path=str(tmp_path / "deleted.txt"))

    stats = client.get("/recovery/stats").json()
    listed = client.get("/recovery/recoverable").json()

    assert stats["recoverable_jobs"] == listed["count"] == 1
    assert [job["id"] for job in listed["jobs"]] == ["ok"]


def test_check_reports_only_stale_active_jobs(store):
    store.create("stale", "run", {}, "corpus")
    store.update("stale", status="running", last_heartbeat=_iso(600))
    store.create("fresh", "run", {}, "corpus")
    store.update("fresh", status="running", last_heartbeat=_iso(1))

    resp = client.get("/recovery/check?timeout_seconds=300")

    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_crashes"] == 1
    assert [job["id"] for job in body["jobs"]] == ["stale"]
    assert body["message"] == "Found 1 potentially crashed job(s)"
