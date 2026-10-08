"""Tests for GET /training/jobs — the live-dict auto-purge age check.

The purge bounds process-local memory only; the store keeps history. Its age
key must fall back to ``created_at``: recovery-synced originals (and any record
never started) have no ``updated_at``/``started_at`` — timestamp 0 made them
look an hour+ stale and purged them the moment they went terminal, dropping
the fresh live state that wins over the store in ``values()``.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler

from domain.shared import to_iso

jobs_api_mod = importlib.import_module("training.jobs_api")

app = FastAPI()
register_app_error_handler(app)
app.include_router(jobs_api_mod.router)
client = TestClient(app)


def _list_ids() -> set[str]:
    resp = client.get("/training/jobs")
    assert resp.status_code == 200
    return {j["id"] for j in resp.json()}


def test_fresh_terminal_job_without_updated_at_survives_the_purge():
    from training.jobs import training_jobs

    training_jobs.set_live(
        "fresh-terminal", {"id": "fresh-terminal", "status": "completed", "progress": 100}
    )
    try:
        assert "fresh-terminal" in _list_ids()
    finally:
        training_jobs.discard_live("fresh-terminal")


def test_terminal_job_past_the_window_is_still_purged():
    # The fix adds created_at to the chain; it must not widen the window.
    from training.jobs import training_jobs

    stale = to_iso(datetime.now(UTC) - timedelta(hours=2))
    training_jobs.set_live(
        "stale-terminal",
        {"id": "stale-terminal", "status": "failed", "created_at": stale},
    )
    try:
        # The endpoint snapshots values() BEFORE the purge loop, so a record
        # purged on this request only disappears from the next response.
        _list_ids()
        assert "stale-terminal" not in _list_ids()
    finally:
        training_jobs.discard_live("stale-terminal")
