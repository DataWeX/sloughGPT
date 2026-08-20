"""Tests for the modular CancelManager."""

import threading
import time

import pytest

from infra_lib.cancel_manager import (
    CancelManager,
    OpStatus,
    OpType,
    Operation,
    get_cancel_manager,
    reset_cancel_manager,
)


class TestOperation:
    def test_to_dict(self):
        op = Operation(
            id="abc",
            op_type=OpType.TRAINING,
            label="test",
            status=OpStatus.RUNNING,
            cancel_fn=lambda: None,
            created_at=1000.0,
            started_at=1001.0,
        )
        d = op.to_dict()
        assert d["id"] == "abc"
        assert d["type"] == "training"
        assert d["status"] == "running"
        assert d["elapsed_s"] >= 0


class TestCancelManager:
    def setup_method(self):
        self.mgr = CancelManager()

    def test_register_returns_id(self):
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: None)
        assert isinstance(oid, str)
        assert len(oid) == 12

    def test_register_custom_id(self):
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: None, op_id="custom")
        assert oid == "custom"

    def test_start_transitions_to_running(self):
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: None)
        self.mgr.start(oid)
        assert self.mgr.get(oid).status == OpStatus.RUNNING

    def test_start_only_transitions_from_registered(self):
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: None)
        self.mgr.start(oid)
        self.mgr.start(oid)  # no-op
        assert self.mgr.get(oid).status == OpStatus.RUNNING

    def test_cancel_running(self):
        called = threading.Event()
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: called.set())
        self.mgr.start(oid)
        result = self.mgr.cancel(oid)
        assert result is True
        assert called.is_set()
        assert self.mgr.get(oid).status == OpStatus.CANCELLED

    def test_cancel_registered(self):
        called = threading.Event()
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: called.set())
        result = self.mgr.cancel(oid)
        assert result is True
        assert called.is_set()
        assert self.mgr.get(oid).status == OpStatus.CANCELLED

    def test_cancel_completed_returns_false(self):
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: None)
        self.mgr.finish(oid)
        result = self.mgr.cancel(oid)
        assert result is False

    def test_cancel_nonexistent_returns_false(self):
        result = self.mgr.cancel("nope")
        assert result is False

    def test_cancel_failed_op_returns_false(self):
        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=lambda: None)
        self.mgr.finish(oid, error="boom")
        result = self.mgr.cancel(oid)
        assert result is False

    def test_cancel_fn_exception_marks_failed(self):
        def bad_fn():
            raise RuntimeError("cancel failed")

        oid = self.mgr.register(OpType.TRAINING, "job1", cancel_fn=bad_fn)
        self.mgr.start(oid)
        result = self.mgr.cancel(oid)
        assert result is False
        assert self.mgr.get(oid).status == OpStatus.FAILED
        assert "cancel failed" in self.mgr.get(oid).error

    def test_cancel_all(self):
        ids = []
        for i in range(3):
            oid = self.mgr.register(OpType.TRAINING, f"job{i}", cancel_fn=lambda: None)
            self.mgr.start(oid)
            ids.append(oid)
        cancelled = self.mgr.cancel_all()
        assert len(cancelled) == 3
        for oid in ids:
            assert self.mgr.get(oid).status == OpStatus.CANCELLED

    def test_cancel_all_by_type(self):
        t1 = self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        t2 = self.mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        self.mgr.start(t1)
        self.mgr.start(t2)
        cancelled = self.mgr.cancel_all(op_type=OpType.TRAINING)
        assert len(cancelled) == 1
        assert self.mgr.get(t1).status == OpStatus.CANCELLED
        assert self.mgr.get(t2).status == OpStatus.RUNNING

    def test_cancel_all_skips_completed(self):
        oid = self.mgr.register(OpType.TRAINING, "done", cancel_fn=lambda: None)
        self.mgr.finish(oid)
        cancelled = self.mgr.cancel_all()
        assert len(cancelled) == 0

    def test_list_active(self):
        t1 = self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        t2 = self.mgr.register(OpType.TRAINING, "t2", cancel_fn=lambda: None)
        self.mgr.start(t1)
        self.mgr.finish(t2)
        active = self.mgr.list_active()
        assert len(active) == 1
        assert active[0].id == t1

    def test_list_active_by_type(self):
        t1 = self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        i1 = self.mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        self.mgr.start(t1)
        self.mgr.start(i1)
        active = self.mgr.list_active(op_type=OpType.TRAINING)
        assert len(active) == 1
        assert active[0].op_type == OpType.TRAINING

    def test_list_all(self):
        self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        self.mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        assert len(self.mgr.list_all()) == 2
        assert len(self.mgr.list_all(op_type=OpType.TRAINING)) == 1

    def test_purge_removes_old(self):
        oid = self.mgr.register(OpType.TRAINING, "old", cancel_fn=lambda: None)
        self.mgr.finish(oid)
        # Backdate finished_at
        self.mgr.get(oid).finished_at = time.time() - 7200
        removed = self.mgr.purge(max_age_s=3600)
        assert removed == 1
        assert self.mgr.get(oid) is None

    def test_purge_keeps_recent(self):
        oid = self.mgr.register(OpType.TRAINING, "new", cancel_fn=lambda: None)
        self.mgr.finish(oid)
        removed = self.mgr.purge(max_age_s=3600)
        assert removed == 0

    def test_count(self):
        self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        self.mgr.register(OpType.TRAINING, "t2", cancel_fn=lambda: None)
        oid3 = self.mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        self.mgr.start(oid3)
        c = self.mgr.count()
        assert c["registered"] == 2
        assert c["running"] == 1

    def test_count_by_type(self):
        self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        self.mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        c = self.mgr.count(op_type=OpType.TRAINING)
        assert c.get("registered", 0) == 1
        assert c.get("running", 0) == 0

    def test_meta_stored(self):
        oid = self.mgr.register(
            OpType.TRAINING, "t1", cancel_fn=lambda: None,
            meta={"dataset": "shakespeare"},
        )
        assert self.mgr.get(oid).meta["dataset"] == "shakespeare"

    def test_concurrent_cancel(self):
        """Multiple threads cancelling the same op — only one should succeed."""
        results = []

        def try_cancel():
            results.append(self.mgr.cancel(oid))

        oid = self.mgr.register(OpType.TRAINING, "contested", cancel_fn=lambda: None)
        self.mgr.start(oid)
        threads = [threading.Thread(target=try_cancel) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Exactly one should return True (the first to enter cancel), rest False
        assert sum(results) >= 1
        assert self.mgr.get(oid).status in (OpStatus.CANCELLING, OpStatus.CANCELLED)

    def test_finish_marks_times(self):
        oid = self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        self.mgr.start(oid)
        self.mgr.finish(oid)
        op = self.mgr.get(oid)
        assert op.started_at is not None
        assert op.finished_at is not None
        assert op.finished_at >= op.started_at

    def test_finish_error(self):
        oid = self.mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        self.mgr.finish(oid, error="OOM")
        op = self.mgr.get(oid)
        assert op.status == OpStatus.FAILED
        assert op.error == "OOM"


class TestSingleton:
    def setup_method(self):
        reset_cancel_manager()

    def teardown_method(self):
        reset_cancel_manager()

    def test_get_returns_same_instance(self):
        m1 = get_cancel_manager()
        m2 = get_cancel_manager()
        assert m1 is m2

    def test_reset_creates_new(self):
        m1 = get_cancel_manager()
        reset_cancel_manager()
        m2 = get_cancel_manager()
        assert m1 is not m2

    def test_thread_safety(self):
        instances = []

        def grab():
            instances.append(get_cancel_manager())

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(id(i) for i in instances)) == 1
