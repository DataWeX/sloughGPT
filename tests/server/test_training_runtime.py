"""Tests for the persistent TrainingRuntime (shutdown + restore)."""

from __future__ import annotations

import threading
import time

import pytest
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
        "name": "distill",
        "model": "sloughgpt",
        "dataset": "test",
        "data_path": "data/input.txt",
        "data_source": "distill",
        "status": status,
        "progress": 0.0,
        "epochs": 10,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "checkpoint": None,
        "checkpoint_dir": "models/auto-training",
        "error": None,
    }
    base.update(overrides)
    return base


class TestRegisterAndSync:
    def test_register_creates_store_row(self, runtime, store):
        runtime.register("j1", _job("j1"), threading.Event(), {"epochs": 10})
        row = store.get("j1")
        assert row is not None
        assert row["name"] == "distill"
        assert row["status"] in ("pending", "running")

    def test_register_backfills_config_on_existing_row(self, runtime, store):
        """job_1438982c: a row persisted without config (the in-memory job dict
        has no config key) must still receive register()'s request snapshot —
        recovery reads the config column to rebuild trainer kwargs, and an
        empty column silently falls back to defaults."""
        store.create("j1", "distill", {}, "data/input.txt")
        snapshot = {"block_size": 512, "batch_size": 8, "epochs": 3}
        runtime.register("j1", _job("j1"), threading.Event(), snapshot)
        assert store.get("j1")["config"] == snapshot

    def test_register_does_not_clobber_non_empty_config(self, runtime, store):
        """First registration wins: a later register() with a different
        snapshot must not overwrite config the row already carries."""
        runtime.register("j1", _job("j1"), threading.Event(), {"epochs": 10})
        runtime.register("j1", _job("j1"), threading.Event(), {"epochs": 99})
        assert store.get("j1")["config"] == {"epochs": 10}

    def test_sync_flushes_progress(self, runtime, store):
        runtime.register("j1", _job("j1"), threading.Event())
        job = runtime.get("j1")
        job["status"] = "running"
        job["progress"] = 0.42
        job["current_epoch"] = 3
        job["train_loss"] = 2.5
        runtime.sync("j1")
        row = store.get("j1")
        assert row["progress"] == 0.42
        assert row["current_epoch"] == 3
        assert row["train_loss"] == 2.5

    def test_sync_terminal_completed_sets_completed_at(self, runtime, store):
        runtime.register("j1", _job("j1"), threading.Event())
        job = runtime.get("j1")
        job["status"] = "completed"
        job["checkpoint"] = "models/auto-training/j1.soul"
        runtime.sync("j1")
        row = store.get("j1")
        assert row["status"] == "completed"
        assert row["completed_at"]

    def test_sync_missing_job_is_noop(self, runtime):
        runtime.sync("nope")  # must not raise


class TestShutdownCooperative:
    def test_sets_cancel_event(self, runtime):
        ev = threading.Event()
        runtime.register("j1", _job("j1"), ev)
        runtime.shutdown()
        assert ev.is_set()

    def test_waits_for_cooperative_job_to_finish(self, store):
        runtime = TrainingRuntime(store=store, grace_timeout_s=2.0)
        ev = threading.Event()
        job = _job("j1")
        runtime.register("j1", job, ev)

        def _worker():
            ev.wait()
            job["status"] = "completed"
            job["checkpoint"] = "models/auto-training/j1.soul"
            runtime.sync("j1")

        t = threading.Thread(target=_worker)
        t.start()
        start = time.monotonic()
        runtime.shutdown()
        elapsed = time.monotonic() - start
        t.join(timeout=5)
        assert elapsed < 2.0
        assert store.get("j1")["status"] == "completed"

    def test_marks_stuck_cooperative_job_interrupted(self, store):
        runtime = TrainingRuntime(store=store, grace_timeout_s=0.2)
        ev = threading.Event()
        runtime.register("j1", _job("j1"), ev)
        runtime.shutdown()
        row = store.get("j1")
        assert row["status"] == "interrupted"
        assert row["crashed"] == 1


class TestShutdownNonCooperative:
    def test_marks_non_cooperative_job_interrupted_immediately(self, store):
        runtime = TrainingRuntime(store=store, grace_timeout_s=5.0)
        runtime.register("j1", _job("j1"), None)
        start = time.monotonic()
        runtime.shutdown()
        elapsed = time.monotonic() - start
        # No cooperative event to wait on — must not burn the full grace.
        assert elapsed < 3.0
        row = store.get("j1")
        assert row["status"] == "interrupted"
        assert row["crashed"] == 1

    def test_terminal_non_cooperative_job_untouched(self, store):
        runtime = TrainingRuntime(store=store, grace_timeout_s=0.2)
        runtime.register("j1", _job("j1", status="completed"), None)
        runtime.shutdown()
        assert store.get("j1")["status"] == "completed"


class TestRestore:
    def test_stale_running_row_marked_interrupted(self, store):
        store.create("stale", "distill", {})
        store.mark_started("stale")
        runtime = TrainingRuntime(store=store, grace_timeout_s=0.2)
        runtime.restore()
        assert store.get("stale")["status"] == "interrupted"

    def test_surviving_rows_seeded_into_job_registry(self, store, monkeypatch):
        store.create("done1", "distill", {"model": "sloughgpt", "epochs": 5}, "test")
        store.mark_completed("done1", "models/auto-training/x.soul")

        from training import jobs as jobs_mod

        monkeypatch.setattr(jobs_mod, "training_jobs", {})
        runtime = TrainingRuntime(store=store, grace_timeout_s=0.2)
        runtime.restore()
        seeded = jobs_mod.training_jobs
        assert "done1" in seeded
        assert seeded["done1"]["name"] == "distill"
        assert seeded["done1"]["status"] == "completed"

    def test_restore_with_empty_store_is_noop(self, runtime):
        runtime.restore()  # must not raise


class TestGet:
    def test_get_returns_mutable_dict(self, runtime):
        job = _job("j1")
        runtime.register("j1", job, threading.Event())
        assert runtime.get("j1") is job

    def test_get_missing_returns_none(self, runtime):
        assert runtime.get("nope") is None


class TestPublisher:
    """The store is the single publisher of job-state changes (card 20261120_110356)."""

    def test_subscribe_returns_init_snapshot_and_publishes_nothing(self, runtime):
        runtime.register("j1", _job("j1"), None)
        events: list[dict] = []
        _token, jobs = runtime.subscribe(events.append)
        assert events == []  # subscribe itself is not an event
        assert len(jobs) == 1
        assert jobs[0]["id"] == "j1"
        assert jobs[0]["status"] == "running"

    def test_register_publishes_started(self, runtime):
        events: list[dict] = []
        runtime.subscribe(events.append)
        runtime.register("j2", _job("j2"), None)
        assert [e["phase"] for e in events] == ["started"]
        assert events[0]["job"]["id"] == "j2"

    def test_sync_publishes_progress_with_detail_fields(self, runtime):
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
            assert key in job
        assert job["loss"] == 1.5

    def test_terminal_sync_publishes_terminal_phase(self, runtime):
        runtime.register("j1", _job("j1"), None)
        events: list[dict] = []
        runtime.subscribe(events.append)
        runtime.get("j1")["status"] = "completed"
        runtime.sync("j1")
        assert [e["phase"] for e in events] == ["completed"]

    def test_fanout_to_multiple_subscribers_and_unsubscribe(self, runtime):
        a: list[dict] = []
        b: list[dict] = []
        token_a, _ = runtime.subscribe(a.append)
        runtime.subscribe(b.append)
        runtime.register("j1", _job("j1"), None)
        runtime.unsubscribe(token_a)
        runtime.register("j2", _job("j2"), None)

        assert [e["job"]["id"] for e in a] == ["j1"]  # gone after unsubscribe
        assert [e["job"]["id"] for e in b] == ["j1", "j2"]

    def test_raising_subscriber_is_dropped_without_wedging(self, runtime):
        def boom(_event):
            raise RuntimeError("dead client")

        runtime.subscribe(boom)
        alive: list[dict] = []
        runtime.subscribe(alive.append)
        runtime.register("j1", _job("j1"), None)  # must not raise
        runtime.register("j2", _job("j2"), None)  # boom already dropped

        assert [e["job"]["id"] for e in alive] == ["j1", "j2"]
        assert len(runtime._subscribers) == 1

    def test_no_subscribers_is_a_cheap_noop(self, runtime):
        runtime.register("j1", _job("j1"), None)
        runtime.get("j1")["progress"] = 10
        runtime.sync("j1")  # must not raise with zero subscribers
        assert runtime._subscribers == {}
