"""Dispatch re-validation — the server is the authority on action dispatch.

Card 20261102_014254. The notification dock renders an action list that
was projected from some earlier process state; by the time a user clicks,
the process may have moved on. The UI must never be trusted: the backend
re-validates the requested action against the process's CURRENT state and
fails closed, so a stale click degrades to a clean refusal — never a wrong
action against a live job (the same seam discipline as the approval gate).
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient
from training.job_store import JobStore
from training.router import router

from tests.conftest import build_test_app

# training/__init__ re-exports `router` (the APIRouter) as a package
# attribute, shadowing the submodule — resolve the module object itself
# so the patch lands on get_job_store, not on the router.
_router_module = importlib.import_module("training.router")


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = JobStore(db_path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(_router_module, "get_job_store", lambda: s)
    return s


@pytest.fixture
def client(store):
    return TestClient(build_test_app(router), raise_server_exceptions=False)


def _row(store: JobStore, jid: str, status: str, **fields) -> None:
    store.create(jid, f"job-{jid}", {"model": "sloughgpt"}, "data/input.txt")
    store.update(jid, status=status)
    for k, v in fields.items():
        store.update(jid, **{k: v})


class TestRecoverRevalidation:
    def test_unknown_job_is_404(self, client):
        r = client.post("/recovery/recover/nope")
        assert r.status_code == 404

    def test_live_running_job_refuses_recover(self, client, store):
        """A notification's Recover button clicked after the job already
        resumed (state moved to running) must be refused fail-closed."""
        _row(store, "j-live", "running")
        r = client.post("/recovery/recover/j-live")
        assert r.status_code == 400
        assert "only 'interrupted' or 'failed'" in r.json()["error"]

    def test_completed_job_refuses_recover(self, client, store):
        _row(store, "j-done", "completed")
        r = client.post("/recovery/recover/j-done")
        assert r.status_code == 400

    def test_interrupted_job_passes_validation_gate(self, client, store):
        """The validation layer itself must accept an interrupted job —
        the refusal above is state-aware, not blanket."""
        _row(store, "j-int", "interrupted")
        # Passes the status gate; then fails at checkpoint resolution
        # (no engine in test env) — anything but the 400 status refusal.
        r = client.post("/recovery/recover/j-int")
        assert r.status_code != 400, r.text
        assert r.status_code in (422, 200), r.text

    def test_stale_recovery_row_with_fresh_heartbeat_refuses(self, client, store):
        """A 'recovering' row whose recovery run is actively alive is not
        recoverable — only a stale-heartbeat one is."""
        _row(store, "j-fresh", "recovering")
        # is_stale_heartbeat uses last_heartbeat set at create() = now
        r = client.post("/recovery/recover/j-fresh")
        assert r.status_code == 400

    def test_refusal_never_mutates_the_store(self, client, store):
        _row(store, "j-live", "running")
        client.post("/recovery/recover/j-live")
        assert store.get("j-live")["status"] == "running"


class TestActionSurfaceFollowsState:
    """The projected actions must track state — the dock's contract test
    at the API seam: recoverable rows carry the recover action."""

    def test_recoverable_snapshot_excludes_live_jobs(self, client, store):
        _row(store, "j-live", "running")
        _row(store, "j-int", "interrupted")
        r = client.get("/recovery/recoverable")
        assert r.status_code == 200
        jobs = r.json()["jobs"]
        assert [j["id"] for j in jobs] == ["j-int"]
