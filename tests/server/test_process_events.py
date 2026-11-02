"""Process-notification contract — ProcessEvent shape + projector.

Card 20261102_014254. The notification tier is a projection of process
events: the action list is the process's REAL command surface, resolved
from the current state (never frozen at creation), and the event_id is
the dedupe key. Boot-time events cannot ride SSE (no client is connected
yet) — they derive from a REST snapshot of the durable JobStore.
"""

from __future__ import annotations

import pytest
from training.job_store import JobStore
from training.process_events import (
    CONTRACT_VERSION,
    PROJECTOR_KIND,
    action_ids,
    boot_snapshot,
    project_job_event,
)

REQUIRED_KEYS = {
    "event_id",
    "kind",
    "process_id",
    "event",
    "state",
    "actions",
    "process_link",
    "title",
}


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=str(tmp_path / "jobs.db"))


def _interrupted_row(store: JobStore, jid: str, **overrides) -> dict:
    store.create(jid, f"job-{jid}", {"model": "sloughgpt"}, "data/input.txt")
    store.update(jid, status="interrupted")
    for k, v in overrides.items():
        store.update(jid, **{k: v})
    return store.get(jid)


class TestProcessEventShape:
    def test_interrupted_job_projects_full_contract(self):
        row = {
            "id": "j1",
            "name": "distill",
            "status": "interrupted",
            "checkpoint_path": "models/j1/ckpt-500",
        }
        ev = project_job_event(row)
        assert ev is not None
        for key in REQUIRED_KEYS:
            assert key in ev, f"missing contract key: {key}"
        assert ev["kind"] == PROJECTOR_KIND == "training"
        assert ev["process_id"] == "j1"
        assert ev["state"] == "interrupted"
        assert ev["event"] == "interrupted"
        assert ev["event_id"].startswith("training:j1:")
        assert ev["process_link"]

    def test_event_id_is_deterministic_for_same_row(self):
        """The event_id is the dedupe key — projecting the same durable row
        twice (REST snapshot + live SSE transition) must yield the same id."""
        row = {"id": "j1", "name": "d", "status": "interrupted"}
        first = project_job_event(row)["event_id"]
        second = project_job_event(dict(row))["event_id"]
        assert first == second

    def test_event_id_differs_across_jobs_and_states(self):
        a = project_job_event({"id": "j1", "status": "interrupted"})["event_id"]
        b = project_job_event({"id": "j2", "status": "interrupted"})["event_id"]
        assert a != b

    def test_settled_states_project_nothing(self):
        """Completed/purged jobs are not notifications — the projector
        returns None and the dock never sees them."""
        for status in ("completed", "pending"):
            assert project_job_event({"id": "j1", "status": status}) is None


class TestActionSurface:
    def test_interrupted_offers_recover(self):
        ev = project_job_event({"id": "j1", "status": "interrupted"})
        assert "recover_now" in action_ids(ev)

    def test_failed_offers_recover(self):
        ev = project_job_event({"id": "j1", "status": "failed"})
        assert "recover_now" in action_ids(ev)

    def test_running_offers_cancel_not_recover(self):
        """Actions are a function of CURRENT state — a live job's surface
        carries cancel and must not offer recover (the stale-closure bug
        class this contract exists to prevent)."""
        ev = project_job_event({"id": "j1", "status": "running"})
        ids = action_ids(ev)
        assert "cancel" in ids
        assert "recover_now" not in ids

    def test_every_action_is_structured(self):
        ev = project_job_event({"id": "j1", "status": "interrupted"})
        for action in ev["actions"]:
            assert set(action) == {"id", "label", "intent"}
            assert action["label"]
            assert action["intent"] in ("recover", "cancel", "navigate", "dismiss")


class TestBootSnapshot:
    def test_snapshot_lists_recoverable_rows_only(self, store):
        _interrupted_row(store, "j1")
        _interrupted_row(store, "j2")
        store.create("j3", "done", {}, "")
        store.update("j3", status="completed")

        events = boot_snapshot(store)
        assert {e["process_id"] for e in events} == {"j1", "j2"}

    def test_snapshot_event_ids_unique(self, store):
        _interrupted_row(store, "j1")
        _interrupted_row(store, "j2")
        events = boot_snapshot(store)
        ids = [e["event_id"] for e in events]
        assert len(ids) == len(set(ids))

    def test_snapshot_is_tolerant_of_store_failure(self):
        class Broken:
            def get_recoverable_jobs(self):
                raise RuntimeError("store down")

        assert boot_snapshot(Broken()) == []

    def test_contract_version_stable(self):
        assert CONTRACT_VERSION == 1
