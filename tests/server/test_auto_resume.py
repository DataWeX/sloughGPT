"""Boot-time auto-resume policy (decision card 20261102_001425, option 2).

The ruling: auto-resume WITH policy — flag-gated, at most one concurrent
resume, skip when no loadable checkpoint. The recovery machinery itself
already exists end-to-end; this is only the boot-time automation, and it
must be legible: every decision (resumed / skipped / declined) is emitted
as a ProcessEvent so the notification dock can show what happened and
offer the real action surface.
"""

from __future__ import annotations

import pytest
from training.auto_resume import AutoResumeOutcome, plan_auto_resume
from training.job_store import JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=str(tmp_path / "jobs.db"))


def _row(store: JobStore, jid: str, status: str = "interrupted", **fields) -> dict:
    store.create(jid, f"job-{jid}", {"model": "sloughgpt"}, "data/input.txt")
    store.update(jid, status=status)
    for k, v in fields.items():
        store.update(jid, **{k: v})
    return store.get(jid)


def _loadable(job: dict) -> bool:
    """Stand-in for engine.resolve_resume_checkpoint success."""
    return bool(job.get("checkpoint_path"))


class TestFlagGate:
    def test_disabled_plan_resumes_nothing(self, store):
        row = _row(store, "j1", checkpoint_path="models/j1/ckpt")
        plan = plan_auto_resume([row], enabled=False, checkpoint_loadable=_loadable)
        assert plan.outcome is AutoResumeOutcome.DISABLED
        assert plan.to_resume == []
        assert plan.events == []

    def test_enabled_plan_resumes_eligible_job(self, store):
        row = _row(store, "j1", checkpoint_path="models/j1/ckpt")
        plan = plan_auto_resume([row], enabled=True, checkpoint_loadable=_loadable)
        assert plan.outcome is AutoResumeOutcome.RESUMED
        assert [j["id"] for j in plan.to_resume] == ["j1"]
        assert len(plan.events) == 1
        assert plan.events[0]["event"] == "auto_train_started"
        assert plan.events[0]["process_id"] == "j1"


class TestConcurrencyCap:
    def test_only_one_job_resumes_even_When_many_eligible(self, store):
        rows = [
            _row(store, jid, checkpoint_path=f"models/{jid}/ckpt") for jid in ("j1", "j2", "j3")
        ]
        plan = plan_auto_resume(rows, enabled=True, checkpoint_loadable=_loadable)
        assert len(plan.to_resume) == 1

    def test_cap_uses_most_recent_eligible_first(self, store):
        """get_recoverable_jobs sorts created_at desc — the plan must not
        reverse that order, so the freshest run resumes, not the oldest."""
        rows = [
            _row(store, "j1", checkpoint_path="models/j1/ckpt"),
            _row(store, "j2", checkpoint_path="models/j2/ckpt"),
        ]
        plan = plan_auto_resume(rows, enabled=True, checkpoint_loadable=_loadable)
        assert [j["id"] for j in plan.to_resume] == ["j1"]


class TestCheckpointGate:
    def test_job_without_loadable_checkpoint_is_skipped(self, store):
        row = _row(store, "j1", checkpoint_path=None)
        plan = plan_auto_resume([row], enabled=True, checkpoint_loadable=_loadable)
        assert plan.to_resume == []
        assert plan.outcome is AutoResumeOutcome.SKIPPED
        assert plan.events[0]["event"] == "auto_resume_skipped"
        assert "checkpoint" in plan.events[0].get("reason", "")

    def test_corrupt_checkpoint_is_skipped_not_fatal(self, store):
        row = _row(store, "j1", checkpoint_path="models/j1/ckpt")

        def corrupt(_job):
            raise ValueError("checkpoint unreadable")

        plan = plan_auto_resume([row], enabled=True, checkpoint_loadable=corrupt)
        assert plan.to_resume == []
        assert plan.events[0]["event"] == "auto_resume_skipped"

    def test_one_bad_one_good_resumes_the_good(self, store):
        bad = _row(store, "j1", checkpoint_path=None)
        good = _row(store, "j2", checkpoint_path="models/j2/ckpt")
        plan = plan_auto_resume([bad, good], enabled=True, checkpoint_loadable=_loadable)
        assert [j["id"] for j in plan.to_resume] == ["j2"]


class TestDecisionLegibility:
    def test_every_skipped_job_emits_its_own_event(self, store):
        rows = [
            _row(store, "j1", checkpoint_path=None),
            _row(store, "j2", checkpoint_path="models/j2/ckpt"),
            _row(store, "j3", checkpoint_path="models/j3/ckpt"),
        ]
        plan = plan_auto_resume(rows, enabled=True, checkpoint_loadable=_loadable)
        # j1 skipped (no checkpoint), j2 resumed, j3 capped out — all legible.
        assert {e["event"] for e in plan.events} == {
            "auto_resume_skipped",
            "auto_train_started",
        }
        skipped_ids = {e["process_id"] for e in plan.events if e["event"] == "auto_resume_skipped"}
        assert "j1" in skipped_ids

    def test_capped_job_gets_no_event(self, store):
        """A job left untouched by the cap is not notified about — no event
        means no dock entry, no phantom 'skipped' claim."""
        rows = [_row(store, jid, checkpoint_path=f"models/{jid}/ckpt") for jid in ("j1", "j2")]
        plan = plan_auto_resume(rows, enabled=True, checkpoint_loadable=_loadable)
        assert {e["process_id"] for e in plan.events} == {"j1"}

    def test_events_carry_the_contract_shape(self, store):
        row = _row(store, "j1", checkpoint_path="models/j1/ckpt")
        plan = plan_auto_resume([row], enabled=True, checkpoint_loadable=_loadable)
        ev = plan.events[0]
        for key in (
            "event_id",
            "kind",
            "process_id",
            "event",
            "state",
            "actions",
            "process_link",
        ):
            assert key in ev, f"missing contract key: {key}"
        assert ev["kind"] == "training"
