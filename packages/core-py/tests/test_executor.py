"""Comprehensive tests for domains.training.executor."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from domains.training.executor import (
    JobInfo,
    JobStatus,
    TrainingExecutor,
    get_training_executor,
)


def _noop(job_id, *args, **kwargs):
    return {"ok": True}


def _sleep_fn(job_id, duration=0.5, *args, **kwargs):
    time.sleep(duration)
    return {"slept": duration}


def _fail_fn(job_id, *args, **kwargs):
    raise ValueError("intentional error")


def _rm_factory():
    rm = MagicMock()
    rm.mode = "balanced"
    rm.recompute = MagicMock(return_value=MagicMock())
    return rm


RM_PATCH = "domains.infrastructure.resource_manager.get_resource_manager"
PC_PATCH = "domains.infrastructure.pugqeep.PointCompressor"


@pytest.fixture(autouse=True)
def _reset_singleton():
    import domains.training.executor as exec_mod
    old = exec_mod._instance
    exec_mod._instance = None
    yield
    current = exec_mod._instance
    if current is not None:
        try:
            current.shutdown(wait=False)
        except Exception:
            pass
    exec_mod._instance = None
    if old is not None:
        exec_mod._instance = old


@pytest.fixture()
def executor():
    ex = TrainingExecutor(max_workers=2)
    yield ex
    try:
        ex.shutdown(wait=False)
    except Exception:
        pass


# ── JobInfo ──────────────────────────────────────────────────────────────

class TestJobInfo:

    def test_construction_defaults(self):
        info = JobInfo(job_id="j1")
        assert info.job_id == "j1"
        assert info.tree_id is None
        assert info.status == JobStatus.QUEUED
        assert info.future is None
        assert isinstance(info.submitted_at, float)
        assert info.started_at is None
        assert info.completed_at is None
        assert info.error is None
        assert info.cancel_requested is False
        assert info.result is None

    def test_elapsed_while_running(self):
        info = JobInfo(job_id="j1", submitted_at=time.time() - 1.0)
        elapsed = info.elapsed()
        assert elapsed is not None
        assert elapsed >= 0.9

    def test_elapsed_after_completion(self):
        info = JobInfo(job_id="j1", submitted_at=100.0, completed_at=105.0)
        assert info.elapsed() == 5.0

    def test_to_dict_fields(self):
        info = JobInfo(job_id="j2", tree_id="t1", status=JobStatus.RUNNING)
        d = info.to_dict()
        assert d["job_id"] == "j2"
        assert d["tree_id"] == "t1"
        assert d["status"] == "running"
        assert "submitted_at" in d
        assert "elapsed_s" in d
        assert d["error"] is None
        assert d["cancel_requested"] is False

    def test_to_dict_with_dict_result(self):
        arr = np.zeros((4, 4), dtype=np.float32)
        info = JobInfo(job_id="j3", status=JobStatus.COMPLETED, result={"weight": arr})
        d = info.to_dict()
        assert "result_keys" in d
        assert d["result_keys"] == ["weight"]
        assert d["result_size_bytes"] == arr.nbytes

    def test_to_dict_with_non_dict_result(self):
        info = JobInfo(job_id="j4", status=JobStatus.COMPLETED, result="simple string")
        d = info.to_dict()
        assert d["result_type"] == "str"
        assert "result_keys" not in d

    def test_to_dict_with_none_result(self):
        info = JobInfo(job_id="j5", status=JobStatus.COMPLETED, result=None)
        d = info.to_dict()
        assert "result_keys" not in d
        assert "result_type" not in d


# ── TrainingExecutor basic ──────────────────────────────────────────────

class TestTrainingExecutorBasic:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_submit_runs_function(self, _mock_rm, executor):
        job_id = executor.submit(_noop, "job_a")
        time.sleep(0.2)
        s = executor.status(job_id)
        assert s is not None
        assert s["status"] == JobStatus.COMPLETED.value

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_job_tracks_queued_to_completed(self, _mock_rm, executor):
        barrier = threading.Barrier(2, timeout=3)

        def slow(job_id):
            barrier.wait()
            return 42

        job_id = executor.submit(slow, "job_b")
        deadline = time.time() + 2
        while time.time() < deadline:
            s = executor.status(job_id)
            if s and s["status"] == JobStatus.RUNNING.value:
                break
            time.sleep(0.02)
        s = executor.status(job_id)
        assert s["status"] == JobStatus.RUNNING.value
        barrier.wait()
        time.sleep(0.2)
        s = executor.status(job_id)
        assert s["status"] == JobStatus.COMPLETED.value

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_status_returns_dict(self, _mock_rm, executor):
        job_id = executor.submit(_noop, "j1")
        time.sleep(0.15)
        s = executor.status(job_id)
        assert isinstance(s, dict)
        assert s["job_id"] == "j1"

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_list_jobs_returns_all(self, _mock_rm, executor):
        executor.submit(_noop, "a1")
        executor.submit(_noop, "a2")
        time.sleep(0.15)
        jobs = executor.list_jobs()
        ids = {j["job_id"] for j in jobs}
        assert ids == {"a1", "a2"}

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_list_jobs_sorted_newest_first(self, _mock_rm, executor):
        executor.submit(_sleep_fn, "old", duration=0.0)
        time.sleep(0.05)
        executor.submit(_sleep_fn, "new", duration=0.0)
        time.sleep(0.15)
        jobs = executor.list_jobs()
        assert jobs[0]["job_id"] == "new"
        assert jobs[1]["job_id"] == "old"

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_active_count_tracks_running(self, _mock_rm, executor):
        barrier = threading.Barrier(2, timeout=3)

        def blocker(job_id):
            barrier.wait()
            return None

        executor.submit(blocker, "r1")
        deadline = time.time() + 2
        while time.time() < deadline:
            if executor.active_count() >= 1:
                break
            time.sleep(0.02)
        assert executor.active_count() >= 1
        barrier.wait()
        time.sleep(0.2)
        assert executor.active_count() == 0


# ── TrainingExecutor error ──────────────────────────────────────────────

class TestTrainingExecutorError:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_failed_job_status(self, _mock_rm, executor):
        job_id = executor.submit(_fail_fn, "err1")
        time.sleep(0.2)
        s = executor.status(job_id)
        assert s["status"] == JobStatus.FAILED.value

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_error_string_captured(self, _mock_rm, executor):
        job_id = executor.submit(_fail_fn, "err2")
        time.sleep(0.2)
        s = executor.status(job_id)
        assert "intentional error" in s["error"]

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_exception_re_raised_from_future(self, _mock_rm, executor):
        job_id = executor.submit(_fail_fn, "err3")
        time.sleep(0.2)
        info = executor._jobs[job_id]
        with pytest.raises(ValueError, match="intentional"):
            info.future.result()


# ── TrainingExecutor cancel ─────────────────────────────────────────────

class TestTrainingExecutorCancel:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_cancel_queued_job_returns_true(self, _mock_rm, executor):
        started = threading.Event()

        def blocker(job_id):
            started.wait(timeout=3)
            return None

        executor.submit(blocker, "b1")
        executor.submit(blocker, "b2")
        # Third job must be queued (both slots occupied)
        job_id = executor.submit(blocker, "queued")
        result = executor.cancel(job_id)
        assert result is True
        # Release blockers
        started.set()
        time.sleep(0.2)

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_cancel_running_job_sets_flag(self, _mock_rm, executor):
        barrier = threading.Barrier(2, timeout=3)

        def blocker(job_id):
            barrier.wait()
            return None

        job_id = executor.submit(blocker, "run1")
        deadline = time.time() + 2
        while time.time() < deadline:
            s = executor.status(job_id)
            if s and s["status"] == JobStatus.RUNNING.value:
                break
            time.sleep(0.02)
        result = executor.cancel(job_id)
        assert result is True
        assert executor.is_cancelled(job_id) is True
        barrier.wait()
        time.sleep(0.1)

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_is_cancelled_returns_true(self, _mock_rm, executor):
        barrier = threading.Barrier(2, timeout=3)

        def blocker(job_id):
            barrier.wait()
            return None

        job_id = executor.submit(blocker, "run2")
        deadline = time.time() + 2
        while time.time() < deadline:
            s = executor.status(job_id)
            if s and s["status"] == JobStatus.RUNNING.value:
                break
            time.sleep(0.02)
        executor.cancel(job_id)
        assert executor.is_cancelled(job_id) is True
        barrier.wait()
        time.sleep(0.1)

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_cancel_unknown_job_returns_false(self, _mock_rm, executor):
        assert executor.cancel("nonexistent") is False


# ── TrainingExecutor purge ──────────────────────────────────────────────

class TestTrainingExecutorPurge:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_purge_removes_old_completed(self, _mock_rm, executor):
        job_id = executor.submit(_noop, "p1")
        time.sleep(0.15)
        info = executor._jobs[job_id]
        info.completed_at = time.time() - 7200
        purged = executor.purge_completed(max_age_s=3600)
        assert purged == 1
        assert executor.status(job_id) is None

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_purge_keeps_recent(self, _mock_rm, executor):
        job_id = executor.submit(_noop, "p2")
        time.sleep(0.15)
        purged = executor.purge_completed(max_age_s=3600)
        assert purged == 0
        assert executor.status(job_id) is not None

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_purge_returns_count(self, _mock_rm, executor):
        j1 = executor.submit(_noop, "pa")
        j2 = executor.submit(_noop, "pb")
        time.sleep(0.15)
        for jid in (j1, j2):
            executor._jobs[jid].completed_at = time.time() - 7200
        purged = executor.purge_completed(max_age_s=3600)
        assert purged == 2


# ── TrainingExecutor shutdown ───────────────────────────────────────────

class TestTrainingExecutorShutdown:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_shutdown_waits(self, _mock_rm, executor):
        results = []

        def slow(job_id):
            time.sleep(0.1)
            results.append("done")

        executor.submit(slow, "s1")
        executor.shutdown(wait=True)
        assert results == ["done"]

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_shutdown_no_wait(self, _mock_rm, executor):
        executor.submit(_sleep_fn, "s2", duration=5.0)
        executor.shutdown(wait=False)


# ── submit_training ─────────────────────────────────────────────────────

class TestSubmitTraining:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_passes_tree_id_and_is_cancelled(self, _mock_rm, executor):
        received = {}

        def capture(jid, tree_id, point_library, is_cancelled, *a, **kw):
            received["jid"] = jid
            received["tree_id"] = tree_id
            received["pl"] = point_library
            received["is_cancelled"] = is_cancelled
            return {"captured": True}

        executor.submit_training(capture, "st1", "tree_abc", point_library=None)
        time.sleep(0.2)
        assert received["jid"] == "st1"
        assert received["tree_id"] == "tree_abc"
        assert received["pl"] is None
        assert callable(received["is_cancelled"])
        assert received["is_cancelled"]() is False

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_auto_stores_dict_result_as_points(self, _mock_rm, executor):
        mock_lib = MagicMock()
        arr = np.ones((4, 4), dtype=np.float32)

        def train_fn(jid, tree_id, point_library, is_cancelled, *a, **kw):
            return {"layer1": arr}

        mock_compressor = MagicMock()
        mock_point = MagicMock()
        mock_point.nbytes.return_value = 64
        mock_compressor.compress_cluster.return_value = mock_point

        with patch(PC_PATCH, return_value=mock_compressor):
            job_id = executor.submit_training(
                train_fn, "st2", "tree_xyz", point_library=mock_lib
            )
            time.sleep(0.3)
            s = executor.status(job_id)
            assert s["status"] == JobStatus.COMPLETED.value
            mock_compressor.compress_cluster.assert_called_once()
            mock_lib.add.assert_called_once_with(mock_point)

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_handles_point_library_store_failure(self, _mock_rm, executor):
        mock_lib = MagicMock()
        arr = np.ones((4, 4), dtype=np.float32)

        def train_fn(jid, tree_id, point_library, is_cancelled, *a, **kw):
            return {"layer1": arr}

        with patch(PC_PATCH, side_effect=RuntimeError("compressor boom")):
            job_id = executor.submit_training(
                train_fn, "st3", "tree_fail", point_library=mock_lib
            )
            time.sleep(0.3)
            s = executor.status(job_id)
            assert s["status"] == JobStatus.COMPLETED.value
            info = executor._jobs[job_id]
            assert info.result == {"layer1": arr}


# ── result_summary ──────────────────────────────────────────────────────

class TestResultSummary:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_returns_none_for_unknown_job(self, _mock_rm, executor):
        assert executor.result_summary("nope") is None

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_returns_none_for_non_completed(self, _mock_rm, executor):
        barrier = threading.Barrier(2, timeout=3)

        def blocker(job_id):
            barrier.wait()
            return {"w": np.zeros(4)}

        job_id = executor.submit(blocker, "rs1")
        deadline = time.time() + 2
        while time.time() < deadline:
            s = executor.status(job_id)
            if s and s["status"] == JobStatus.RUNNING.value:
                break
            time.sleep(0.02)
        assert executor.result_summary(job_id) is None
        barrier.wait()
        time.sleep(0.1)

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_returns_shape_dtype_for_completed_dict(self, _mock_rm, executor):
        arr = np.zeros((3, 5), dtype=np.float32)

        def train(job_id):
            return {"w1": arr}

        job_id = executor.submit(train, "rs2")
        time.sleep(0.2)
        summary = executor.result_summary(job_id)
        assert summary is not None
        assert summary["job_id"] == "rs2"
        assert "w1" in summary["weights"]
        w = summary["weights"]["w1"]
        assert w["shape"] == [3, 5]
        assert "float32" in w["dtype"]
        assert w["nbytes"] == arr.nbytes
        assert summary["total_bytes"] == arr.nbytes


# ── get_training_executor singleton ─────────────────────────────────────

class TestGetTrainingExecutor:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_returns_same_instance(self, _mock_rm):
        import domains.training.executor as exec_mod
        exec_mod._instance = None
        a = get_training_executor()
        b = get_training_executor()
        assert a is b
        a.shutdown(wait=False)
        exec_mod._instance = None

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_thread_safe(self, _mock_rm):
        import domains.training.executor as exec_mod
        exec_mod._instance = None
        instances = []

        def grab():
            instances.append(get_training_executor())

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(i is instances[0] for i in instances)
        instances[0].shutdown(wait=False)
        exec_mod._instance = None


# ── _running() internal ─────────────────────────────────────────────────

class TestRunningInternal:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_running_counts_correctly(self, _mock_rm, executor):
        barrier = threading.Barrier(3, timeout=3)

        def blocker(job_id):
            barrier.wait()
            return None

        executor.submit(blocker, "r1")
        executor.submit(blocker, "r2")
        deadline = time.time() + 2
        while time.time() < deadline:
            if executor._running() >= 2:
                break
            time.sleep(0.02)
        assert executor._running() >= 2
        barrier.wait()
        time.sleep(0.3)
        assert executor._running() == 0


# ── JobStatus enum ──────────────────────────────────────────────────────

class TestJobStatus:

    def test_all_statuses_exist(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_is_str_enum(self):
        assert isinstance(JobStatus.QUEUED, str)
        assert JobStatus.QUEUED == "queued"


# ── Submit kwargs forwarding ────────────────────────────────────────────

class TestSubmitKwargs:

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_call_args_merged(self, _mock_rm, executor):
        received = {}

        def fn(job_id, lr=0.001, epochs=10):
            received["lr"] = lr
            received["epochs"] = epochs
            return True

        executor.submit(fn, "kw1", _call_args={"lr": 0.01}, epochs=5)
        time.sleep(0.15)
        assert received["lr"] == 0.01
        assert received["epochs"] == 5

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_tree_id_stored(self, _mock_rm, executor):
        job_id = executor.submit(_noop, "kw2", tree_id="my_tree")
        time.sleep(0.15)
        s = executor.status(job_id)
        assert s["tree_id"] == "my_tree"

    @patch(RM_PATCH, side_effect=_rm_factory)
    def test_positional_args_forwarded(self, _mock_rm, executor):
        received = {}

        def fn(job_id, a, b, c):
            received["a"] = a
            received["b"] = b
            received["c"] = c
            return True

        executor.submit(fn, "kw3", 1, 2, 3)
        time.sleep(0.15)
        assert received == {"a": 1, "b": 2, "c": 3}
