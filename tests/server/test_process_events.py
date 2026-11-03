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
    ACTION_INTENTS,
    ACTION_SCHEMA,
    CONTRACT_VERSION,
    EVENT_SCHEMA,
    EVENTS,
    PROJECTOR_KIND,
    STATES,
    _ACTIONS_BY_STATE,
    _TITLES,
    action_ids,
    boot_snapshot,
    make_event,
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
        required = {f for f, (_t, req) in ACTION_SCHEMA.items() if req}
        for action in ev["actions"]:
            assert set(action) == required
            assert action["label"]
            assert action["intent"] in ACTION_INTENTS


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


class TestContractSchema:
    """One EVENT_SCHEMA source (card 20261102_110421): the schema, the
    vocabularies and the projector must not drift apart. The TS half of
    the projection (``contracts.gen.ts`` ProcessEvent) is generated from
    this same declaration by scripts/gen_ts_contracts.py — two hand-
    written copies of one envelope is the drift this replaces.
    """

    @staticmethod
    def _required() -> set:
        return {f for f, (_t, req) in EVENT_SCHEMA.items() if req}

    @staticmethod
    def _optional() -> set:
        return {f for f, (_t, req) in EVENT_SCHEMA.items() if not req}

    def test_required_fields_are_exactly_the_bare_make_event_output(self):
        ev = make_event("j1", "interrupted", "interrupted")
        assert set(ev) == self._required()

    def test_optional_fields_are_exactly_the_declared_extras(self):
        ev = make_event(
            "j1",
            "auto_resume_skipped",
            "interrupted",
            surface="skipped",
            reason="no loadable checkpoint",
            name="distill",
        )
        assert set(ev) == self._required() | self._optional()
        assert self._optional() == {"reason", "name"}

    def test_undeclared_extra_field_is_rejected(self):
        """A producer inventing a field (e.g. the frontend-only ``resolved``
        that never existed on the wire) must fail at construction, not
        drift silently past the schema."""
        with pytest.raises(ValueError, match="undeclared"):
            make_event("j1", "interrupted", "interrupted", resolved=True)

    def test_unknown_state_event_and_surface_are_rejected(self):
        with pytest.raises(ValueError, match="state"):
            make_event("j1", "interrupted", "completed")
        with pytest.raises(ValueError, match="event"):
            make_event("j1", "not_a_title", "interrupted")
        with pytest.raises(ValueError, match="surface"):
            make_event("j1", "interrupted", "interrupted", surface="bogus")

    def test_every_table_intent_is_declared(self):
        intents = {
            a["intent"]
            for surface in _ACTIONS_BY_STATE.values()
            for a in surface
        }
        assert intents <= set(ACTION_INTENTS)

    def test_every_table_action_matches_the_action_schema(self):
        """The action sub-object has ONE declaration too (ACTION_SCHEMA) —
        the generated TS ProcessAction is rendered from it."""
        required = {f for f, (_t, req) in ACTION_SCHEMA.items() if req}
        assert required == {"id", "label", "intent"}
        for surface in _ACTIONS_BY_STATE.values():
            for action in surface:
                assert set(action) == required

    def test_vocabularies_agree_with_the_tables(self):
        assert set(EVENTS) == set(_TITLES)
        assert set(STATES) <= set(_ACTIONS_BY_STATE)
        assert set(STATES) <= set(EVENTS)

    def test_every_projected_event_stays_within_schema_and_vocab(self, store):
        _interrupted_row(store, "j1")
        required = self._required()
        events = boot_snapshot(store)
        assert events, "fixture should project at least one event"
        for ev in events:
            assert required <= set(ev)
            assert set(ev) <= required | self._optional()
            assert ev["state"] in STATES
            assert ev["event"] in EVENTS
            for action in ev["actions"]:
                assert action["intent"] in ACTION_INTENTS
