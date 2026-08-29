"""Meaningful tests for CancelManager — state transitions, cancel flow, error handling, purge, count."""

import threading
import time
from domains.infrastructure.cancel_manager import (
    CancelManager, OpType, OpStatus, Operation, get_cancel_manager, reset_cancel_manager,
)


class TestOperationToDict:
    def test_to_dict_completed(self):
        op = Operation(
            id="x", op_type=OpType.TRAINING, label="test", status=OpStatus.COMPLETED,
            cancel_fn=lambda: None, created_at=100.0, started_at=101.0, finished_at=105.0,
        )
        d = op.to_dict()
        assert d["id"] == "x"
        assert d["elapsed_s"] == 4.0

    def test_to_dict_no_start(self):
        op = Operation(
            id="x", op_type=OpType.INFERENCE, label="test", status=OpStatus.REGISTERED,
            cancel_fn=lambda: None, created_at=100.0,
        )
        d = op.to_dict()
        # elapsed from created_at to now should be >= 0
        assert d["elapsed_s"] >= 0


class TestCancelManagerRegister:
    def test_register_returns_id(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        assert isinstance(oid, str)
        assert len(oid) == 12

    def test_register_custom_id(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None, op_id="my_id")
        assert oid == "my_id"
        assert mgr.get("my_id") is not None

    def test_register_stored_as_registered(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        op = mgr.get(oid)
        assert op.status == OpStatus.REGISTERED


class TestCancelManagerStart:
    def test_start_transitions_to_running(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        mgr.start(oid)
        assert mgr.get(oid).status == OpStatus.RUNNING

    def test_start_sets_started_at(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        before = time.time()
        mgr.start(oid)
        after = time.time()
        op = mgr.get(oid)
        assert before <= op.started_at <= after

    def test_start_only_works_from_registered(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        mgr.start(oid)
        mgr.start(oid)  # Already running — should be no-op
        assert mgr.get(oid).status == OpStatus.RUNNING

    def test_start_nonexistent(self):
        mgr = CancelManager()
        mgr.start("nonexistent")  # Should not raise


class TestCancelManagerFinish:
    def test_finish_completed(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        mgr.finish(oid)
        op = mgr.get(oid)
        assert op.status == OpStatus.COMPLETED
        assert op.error is None
        assert op.finished_at is not None

    def test_finish_with_error(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        mgr.finish(oid, error="OOM")
        op = mgr.get(oid)
        assert op.status == OpStatus.FAILED
        assert op.error == "OOM"

    def test_finish_nonexistent(self):
        mgr = CancelManager()
        mgr.finish("nonexistent")  # Should not raise


class TestCancelManagerCancel:
    def test_cancel_registered(self):
        cancelled = []
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: cancelled.append(True))
        result = mgr.cancel(oid)
        assert result is True
        assert cancelled == [True]
        assert mgr.get(oid).status == OpStatus.CANCELLED

    def test_cancel_running(self):
        cancelled = []
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: cancelled.append(True))
        mgr.start(oid)
        result = mgr.cancel(oid)
        assert result is True
        assert mgr.get(oid).status == OpStatus.CANCELLED

    def test_cancel_nonexistent(self):
        mgr = CancelManager()
        assert mgr.cancel("nope") is False

    def test_cancel_already_completed(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        mgr.finish(oid)
        assert mgr.cancel(oid) is False

    def test_cancel_already_cancelled(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", lambda: None)
        mgr.cancel(oid)
        assert mgr.cancel(oid) is False

    def test_cancel_fn_raises_marks_failed(self):
        def bad_cancel():
            raise RuntimeError("boom")

        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "job1", bad_cancel)
        result = mgr.cancel(oid)
        assert result is False
        op = mgr.get(oid)
        assert op.status == OpStatus.FAILED
        assert "boom" in op.error

    def test_cancel_multiple(self):
        mgr = CancelManager()
        ids = []
        for i in range(5):
            ids.append(mgr.register(OpType.TRAINING, f"job{i}", lambda: None))
        cancelled = mgr.cancel_all()
        assert len(cancelled) == 5
        for oid in ids:
            assert mgr.get(oid).status == OpStatus.CANCELLED


class TestCancelAll:
    def test_cancel_all_filters_by_type(self):
        mgr = CancelManager()
        t1 = mgr.register(OpType.TRAINING, "t1", lambda: None)
        t2 = mgr.register(OpType.TRAINING, "t2", lambda: None)
        i1 = mgr.register(OpType.INFERENCE, "i1", lambda: None)
        cancelled = mgr.cancel_all(op_type=OpType.TRAINING)
        assert len(cancelled) == 2
        assert mgr.get(i1).status == OpStatus.REGISTERED

    def test_cancel_all_returns_empty(self):
        mgr = CancelManager()
        cancelled = mgr.cancel_all()
        assert cancelled == []


class TestCancelManagerPurge:
    def test_purge_removes_old(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "old_job", lambda: None)
        mgr.finish(oid)
        # Fake old timestamp
        mgr.get(oid).finished_at = time.time() - 7200
        removed = mgr.purge(max_age_s=3600)
        assert removed == 1
        assert mgr.get(oid) is None

    def test_purge_keeps_recent(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "new_job", lambda: None)
        mgr.finish(oid)
        removed = mgr.purge(max_age_s=3600)
        assert removed == 0
        assert mgr.get(oid) is not None


class TestCancelManagerCount:
    def test_count(self):
        mgr = CancelManager()
        mgr.register(OpType.TRAINING, "t1", lambda: None)
        mgr.register(OpType.TRAINING, "t2", lambda: None)
        mgr.register(OpType.INFERENCE, "i1", lambda: None)
        c = mgr.count()
        assert c["registered"] == 3

    def test_count_filtered(self):
        mgr = CancelManager()
        mgr.register(OpType.TRAINING, "t1", lambda: None)
        mgr.register(OpType.INFERENCE, "i1", lambda: None)
        c = mgr.count(op_type=OpType.TRAINING)
        assert c["registered"] == 1


class TestCancelManagerListActive:
    def test_list_active(self):
        mgr = CancelManager()
        oid1 = mgr.register(OpType.TRAINING, "t1", lambda: None)
        mgr.start(oid1)
        oid2 = mgr.register(OpType.TRAINING, "t2", lambda: None)
        mgr.finish(oid2)
        active = mgr.list_active()
        assert len(active) == 1
        assert active[0].id == oid1

    def test_list_active_cancelling(self):
        mgr = CancelManager()
        oid = mgr.register(OpType.TRAINING, "t1", lambda: None)
        # Manually set to cancelling
        mgr.get(oid).status = OpStatus.CANCELLING
        active = mgr.list_active()
        assert len(active) == 1


class TestSingleton:
    def test_singleton(self):
        reset_cancel_manager()
        m1 = get_cancel_manager()
        m2 = get_cancel_manager()
        assert m1 is m2
        reset_cancel_manager()

    def test_reset(self):
        reset_cancel_manager()
        m1 = get_cancel_manager()
        m1.register(OpType.TRAINING, "test", lambda: None)
        reset_cancel_manager()
        m2 = get_cancel_manager()
        assert m2 is not m1
        assert len(m2.list_all()) == 0


class TestCancelManagerConcurrency:
    def test_concurrent_register(self):
        mgr = CancelManager()
        ids = []

        def register_one():
            oid = mgr.register(OpType.TRAINING, "concurrent", lambda: None)
            ids.append(oid)

        threads = [threading.Thread(target=register_one) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(ids) == 50
        assert len(set(ids)) == 50
