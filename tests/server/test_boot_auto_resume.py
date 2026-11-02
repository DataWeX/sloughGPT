"""Boot-time auto-resume execution + the REST events snapshot endpoint.

Card 20261102_014254. The policy planner is pure (test_auto_resume.py);
this file pins the *execution seam*: the startup hook fetches durable
rows, probes checkpoints, plans, and executes through the SAME recovery
path the endpoint uses — so the server-authority re-validation that
guards the endpoint guards boot-time dispatch too (a race where the job
moved on since restore() must refuse, not double-start).

The events endpoint is the dock's boot source: SSE cannot deliver events
that fired while no client was connected, so the frontend's initial
notification state is this REST snapshot of durable state.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient
from training.auto_resume import AutoResumeOutcome, run_auto_resume
from training.job_store import JobStore
from training.router import router

from tests.conftest import build_test_app

_router_module = importlib.import_module("training.router")


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = JobStore(db_path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(_router_module, "get_job_store", lambda: s)
    return s


@pytest.fixture
def client(store):
    return TestClient(build_test_app(router), raise_server_exceptions=False)


def _row(store: JobStore, jid: str, status: str = "interrupted", **fields) -> dict:
    store.create(jid, f"job-{jid}", {"model": "sloughgpt"}, "data/input.txt")
    store.update(jid, status=status)
    for k, v in fields.items():
        store.update(jid, **{k: v})
    return store.get(jid)


def _loadable(job: dict) -> bool:
    return bool(job.get("checkpoint_path"))


class TestRunAutoResume:
    def test_disabled_flag_never_touches_the_recovery_path(self, store):
        _row(store, "j1", checkpoint_path="models/j1/ckpt")
        calls: list[str] = []

        result = run_auto_resume(
            store,
            enabled=False,
            checkpoint_loadable=_loadable,
            recover=lambda jid: calls.append(jid) or {"status": "recovered"},
        )

        assert result.outcome is AutoResumeOutcome.DISABLED
        assert calls == []
        assert result.events == []

    def test_enabled_resumes_through_the_injected_path(self, store):
        _row(store, "j1", checkpoint_path="models/j1/ckpt")
        calls: list[str] = []

        result = run_auto_resume(
            store,
            enabled=True,
            checkpoint_loadable=_loadable,
            recover=lambda jid: calls.append(jid) or {"status": "recovered"},
        )

        assert result.outcome is AutoResumeOutcome.RESUMED
        assert calls == ["j1"]
        assert result.events[0]["event"] == "auto_train_started"

    def test_cap_holds_at_one_concurrent_resume(self, store):
        for jid in ("j1", "j2", "j3"):
            _row(store, jid, checkpoint_path=f"models/{jid}/ckpt")
        calls: list[str] = []

        run_auto_resume(
            store,
            enabled=True,
            checkpoint_loadable=_loadable,
            recover=lambda jid: calls.append(jid) or {"status": "recovered"},
        )

        assert len(calls) == 1

    def test_no_eligible_jobs_plans_nothing(self, store):
        _row(store, "j1", checkpoint_path=None)
        calls: list[str] = []

        result = run_auto_resume(
            store,
            enabled=True,
            checkpoint_loadable=_loadable,
            recover=lambda jid: calls.append(jid) or {"status": "recovered"},
        )

        assert result.outcome is AutoResumeOutcome.SKIPPED
        assert calls == []
        assert result.events[0]["event"] == "auto_resume_skipped"

    def test_recovery_refusal_is_swallowed_not_fatal(self, store):
        """Server-authority refusal (job moved on since restore) must not
        crash the boot path — the plan records it and boot continues."""
        _row(store, "j1", checkpoint_path="models/j1/ckpt")

        def refuse(_jid):
            raise RuntimeError("Job status is 'running', only ... can be recovered")

        result = run_auto_resume(
            store, enabled=True, checkpoint_loadable=_loadable, recover=refuse
        )

        assert result.outcome is AutoResumeOutcome.SKIPPED
        assert result.events[0]["event"] == "auto_resume_skipped"

    def test_store_failure_is_swallowed_not_fatal(self):
        class Broken:
            def get_recoverable_jobs(self):
                raise RuntimeError("db down")

        result = run_auto_resume(
            Broken(),
            enabled=True,
            checkpoint_loadable=_loadable,
            recover=lambda jid: {"status": "recovered"},
        )
        assert result.outcome is AutoResumeOutcome.SKIPPED
        assert result.to_resume == []


class TestAutoResumeConfigFlag:
    """The policy is flag-gated (decision 20261102_001425) — default OFF,
    so an infra reboot never surprises anyone with GPU burn."""

    def test_default_is_off(self, monkeypatch):
        from config import ServerConfig

        monkeypatch.delenv("SLO_AUTO_RESUME", raising=False)
        assert ServerConfig.from_env().auto_resume is False

    def test_env_enables_it(self, monkeypatch):
        from config import ServerConfig

        monkeypatch.setenv("SLO_AUTO_RESUME", "true")
        assert ServerConfig.from_env().auto_resume is True
        monkeypatch.setenv("SLO_AUTO_RESUME", "0")
        assert ServerConfig.from_env().auto_resume is False


class TestEventsEndpoint:
    def test_interrupted_job_projects_contract_event(self, client, store):
        _row(store, "j1", checkpoint_path="models/j1/ckpt")
        r = client.get("/recovery/events")
        assert r.status_code == 200
        body = r.json()
        assert body["contract_version"] == 1
        assert len(body["events"]) == 1
        ev = body["events"][0]
        for key in ("event_id", "kind", "process_id", "event", "state", "actions"):
            assert key in ev
        assert ev["kind"] == "training"
        assert ev["process_id"] == "j1"
        assert any(a["id"] == "recover_now" for a in ev["actions"])

    def test_settled_jobs_are_not_events(self, client, store):
        _row(store, "j1", checkpoint_path="models/j1/ckpt")
        store.update("j1", status="completed")
        r = client.get("/recovery/events")
        assert r.json()["events"] == []

    def test_snapshot_dedupes_identical_events(self, client, store):
        _row(store, "j1", checkpoint_path="models/j1/ckpt")
        _row(store, "j1b", checkpoint_path="models/j1b/ckpt")
        r = client.get("/recovery/events")
        ids = [e["event_id"] for e in r.json()["events"]]
        assert len(ids) == len(set(ids))
