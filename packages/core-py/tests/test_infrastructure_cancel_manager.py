"""Tests for CancelManager — operation registration, lifecycle, cancellation."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from domains.infrastructure.cancel_manager import CancelManager, OpStatus, OpType


@pytest.fixture()
def mgr() -> CancelManager:
    """Fresh CancelManager per test — no cross-test pollution."""
    return CancelManager()


class TestRegister:
    def test_register_returns_id(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "test-job", cancel_fn=lambda: None)
        assert isinstance(op_id, str)
        assert len(op_id) > 0

    def test_register_stores_operation(self, mgr: CancelManager):
        op_id = mgr.register(OpType.INFERENCE, "infer-1", cancel_fn=lambda: None)
        op = mgr.get(op_id)
        assert op is not None
        assert op.label == "infer-1"
        assert op.status == OpStatus.REGISTERED

    def test_register_with_meta(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None, meta={"lr": 0.01})
        op = mgr.get(op_id)
        assert op is not None
        assert op.meta == {"lr": 0.01}


class TestLifecycle:
    def test_start_sets_running(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None)
        mgr.start(op_id)
        op = mgr.get(op_id)
        assert op is not None
        assert op.status == OpStatus.RUNNING
        assert op.started_at is not None

    def test_finish_sets_completed(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None)
        mgr.start(op_id)
        mgr.finish(op_id)
        op = mgr.get(op_id)
        assert op is not None
        assert op.status == OpStatus.COMPLETED
        assert op.finished_at is not None

    def test_finish_with_error_sets_failed(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None)
        mgr.start(op_id)
        mgr.finish(op_id, error="OOM")
        op = mgr.get(op_id)
        assert op is not None
        assert op.status == OpStatus.FAILED
        assert op.error == "OOM"


class TestCancel:
    def test_cancel_calls_cancel_fn(self, mgr: CancelManager):
        cancel_fn = MagicMock()
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=cancel_fn)
        mgr.start(op_id)
        result = mgr.cancel(op_id)
        assert result is True
        cancel_fn.assert_called_once()

    def test_cancel_sets_cancelling_then_cancelled(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None)
        mgr.start(op_id)
        mgr.cancel(op_id)
        op = mgr.get(op_id)
        assert op is not None
        assert op.status == OpStatus.CANCELLED

    def test_cancel_nonexistent_returns_false(self, mgr: CancelManager):
        assert mgr.cancel("nonexistent") is False

    def test_cancel_already_completed_returns_false(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None)
        mgr.start(op_id)
        mgr.finish(op_id)
        assert mgr.cancel(op_id) is False


class TestCancelAll:
    def test_cancel_all_by_type(self, mgr: CancelManager):
        cancel1 = MagicMock()
        cancel2 = MagicMock()
        mgr.register(OpType.TRAINING, "t1", cancel_fn=cancel1)
        mgr.register(OpType.TRAINING, "t2", cancel_fn=cancel2)
        mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        cancelled = mgr.cancel_all(op_type=OpType.TRAINING)
        assert len(cancelled) == 2
        assert cancel1.call_count == 1
        assert cancel2.call_count == 1

    def test_cancel_all_without_type(self, mgr: CancelManager):
        mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        cancelled = mgr.cancel_all()
        assert len(cancelled) == 2


class TestListAndCount:
    def test_list_active(self, mgr: CancelManager):
        mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        mgr.register(OpType.TRAINING, "t2", cancel_fn=lambda: None)
        active = mgr.list_active()
        assert len(active) == 2

    def test_list_active_excludes_completed(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        mgr.start(op_id)
        mgr.finish(op_id)
        assert len(mgr.list_active()) == 0

    def test_count(self, mgr: CancelManager):
        mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        counts = mgr.count()
        assert counts["registered"] == 2

    def test_count_by_type(self, mgr: CancelManager):
        mgr.register(OpType.TRAINING, "t1", cancel_fn=lambda: None)
        mgr.register(OpType.INFERENCE, "i1", cancel_fn=lambda: None)
        counts = mgr.count(op_type=OpType.TRAINING)
        assert counts["registered"] == 1

    def test_list_all_with_limit(self, mgr: CancelManager):
        for i in range(5):
            mgr.register(OpType.OTHER, f"job-{i}", cancel_fn=lambda: None)
        result = mgr.list_all(limit=3)
        assert len(result) == 3


class TestPurge:
    def test_purge_removes_old_ops(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "old", cancel_fn=lambda: None)
        mgr.start(op_id)
        mgr.finish(op_id)
        # Simulate old timestamp
        op = mgr.get(op_id)
        op.created_at = time.time() - 7200
        op.finished_at = time.time() - 7200
        purged = mgr.purge(max_age_s=3600)
        assert purged >= 1
        assert mgr.get(op_id) is None

    def test_purge_keeps_recent_ops(self, mgr: CancelManager):
        mgr.register(OpType.TRAINING, "new", cancel_fn=lambda: None)
        purged = mgr.purge(max_age_s=3600)
        assert purged == 0


class TestOperationToDict:
    def test_to_dict(self, mgr: CancelManager):
        op_id = mgr.register(OpType.TRAINING, "job", cancel_fn=lambda: None)
        op = mgr.get(op_id)
        d = op.to_dict()
        assert d["id"] == op_id
        assert d["type"] == "training"
        assert d["label"] == "job"
        assert "elapsed_s" in d
