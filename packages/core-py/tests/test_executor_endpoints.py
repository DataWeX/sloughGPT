"""Tests for domains.training.executor — TrainingExecutor, JobInfo, JobStatus, get_training_executor."""

import sys
import time
import threading
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pytest

from domains.training.executor import (
    TrainingExecutor,
    JobInfo,
    JobStatus,
    get_training_executor,
)


@pytest.fixture(autouse=True)
def reset_executor():
    """Reset TrainingExecutor singleton before each test."""
    import domains.training.executor as exec_mod
    old = exec_mod._instance
    exec_mod._instance = None
    yield
    if old is not None:
        old.shutdown(wait=False)
    exec_mod._instance = None


# ── JobStatus ────────────────────────────────────────────────────────────────

class TestJobStatus:
    def test_queued_value(self):
        assert JobStatus.QUEUED.value == "queued"

    def test_running_value(self):
        assert JobStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert JobStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert JobStatus.FAILED.value == "failed"

    def test_cancelled_value(self):
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_is_string_enum(self):
        assert isinstance(JobStatus.QUEUED, str)


# ── JobInfo ──────────────────────────────────────────────────────────────────

class TestJobInfo:
    def test_defaults(self):
        info = JobInfo(job_id="j1")
        assert info.job_id == "j1"
        assert info.tree_id is None
        assert info.status == JobStatus.QUEUED
        assert info.submitted_at > 0
        assert info.started_at is None
        assert info.completed_at is None
        assert info.error is None
        assert info.cancel_requested is False
        assert info.result is None

    def test_elapsed_running(self):
        info = JobInfo(job_id="j1")
        time.sleep(0.01)
        elapsed = info.elapsed()
        assert elapsed > 0

    def test_elapsed_completed(self):
        info = JobInfo(job_id="j1")
        info.completed_at = info.submitted_at + 5.0
        assert info.elapsed() == 5.0

    def test_to_dict_basic(self):
        info = JobInfo(job_id="j1")
        d = info.to_dict()
        assert d["job_id"] == "j1"
        assert d["status"] == "queued"
        assert "elapsed_s" in d
        assert "submitted_at" in d

    def test_to_dict_completed_with_dict_result(self):
        info = JobInfo(job_id="j1", status=JobStatus.COMPLETED)
        info.result = {"w1": np.zeros(8, dtype=np.float32)}
        d = info.to_dict()
        assert "result_keys" in d
        assert "w1" in d["result_keys"]
        assert "result_size_bytes" in d
        assert d["result_size_bytes"] > 0

    def test_to_dict_completed_with_non_dict_result(self):
        info = JobInfo(job_id="j1", status=JobStatus.COMPLETED)
        info.result = 42
        d = info.to_dict()
        assert d["result_type"] == "int"

    def test_to_dict_tree_id(self):
        info = JobInfo(job_id="j1", tree_id="tree_a")
        d = info.to_dict()
        assert d["tree_id"] == "tree_a"


# ── TrainingExecutor construction ───────────────────────────────────────────

class TestTrainingExecutorConstruction:
    def test_default_max_workers(self):
        ex = TrainingExecutor()
        assert ex._max_workers == 2
        ex.shutdown(wait=False)

    def test_custom_max_workers(self):
        ex = TrainingExecutor(max_workers=4)
        assert ex._max_workers == 4
        ex.shutdown(wait=False)

    def test_jobs_starts_empty(self):
        ex = TrainingExecutor()
        assert len(ex._jobs) == 0
        ex.shutdown(wait=False)

    def test_singleton(self):
        ex1 = get_training_executor()
        ex2 = get_training_executor()
        assert ex1 is ex2


# ── Submit ───────────────────────────────────────────────────────────────────

class TestSubmit:
    def test_submit_returns_job_id(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        result = ex.submit(noop, "job1")
        assert result == "job1"
        ex.shutdown(wait=False)

    def test_submit_creates_job_info(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "job1")
        time.sleep(0.05)
        assert "job1" in ex._jobs
        ex.shutdown(wait=False)

    def test_submit_runs_function(self):
        ex = TrainingExecutor(max_workers=2)
        results = []
        def worker(job_id):
            results.append("done")
        ex.submit(worker, "w1")
        time.sleep(0.1)
        assert "done" in results
        ex.shutdown(wait=False)

    def test_submit_with_args(self):
        ex = TrainingExecutor(max_workers=2)
        received = []
        def worker(job_id, x, y):
            received.append((x, y))
        ex.submit(worker, "args_job", 10, 20)
        time.sleep(0.1)
        assert received == [(10, 20)]
        ex.shutdown(wait=False)

    def test_submit_with_call_args(self):
        ex = TrainingExecutor(max_workers=2)
        received = {}
        def worker(job_id, alpha=1, beta=2):
            received["alpha"] = alpha
            received["beta"] = beta
        ex.submit(worker, "kw_job", _call_args={"alpha": 5, "beta": 6})
        time.sleep(0.1)
        assert received["alpha"] == 5
        assert received["beta"] == 6
        ex.shutdown(wait=False)

    def test_submit_with_tree_id(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "tree_job", tree_id="my_tree")
        time.sleep(0.05)
        assert ex._jobs["tree_job"].tree_id == "my_tree"
        ex.shutdown(wait=False)

    def test_submit_multiple(self):
        ex = TrainingExecutor(max_workers=4)
        done = []
        def worker(job_id):
            done.append(job_id)
        for i in range(5):
            ex.submit(worker, f"m{i}")
        time.sleep(0.2)
        assert len(done) == 5
        ex.shutdown(wait=False)

    def test_submit_exception_marks_failed(self):
        ex = TrainingExecutor(max_workers=2)
        def bad(job_id):
            raise ValueError("boom")
        ex.submit(bad, "fail1")
        time.sleep(0.1)
        info = ex._jobs["fail1"]
        assert info.status == JobStatus.FAILED
        assert "boom" in info.error
        ex.shutdown(wait=False)

    def test_submit_sets_started_at(self):
        ex = TrainingExecutor(max_workers=2)
        def slow(job_id):
            time.sleep(0.05)
        ex.submit(slow, "started")
        time.sleep(0.05)
        assert ex._jobs["started"].started_at is not None
        ex.shutdown(wait=False)

    def test_submit_sets_completed_at(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "comp")
        time.sleep(0.1)
        assert ex._jobs["comp"].completed_at is not None
        ex.shutdown(wait=False)


# ── Status ───────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_unknown_returns_none(self):
        ex = TrainingExecutor(max_workers=2)
        assert ex.status("unknown") is None
        ex.shutdown(wait=False)

    def test_status_returns_dict(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "s1")
        time.sleep(0.1)
        s = ex.status("s1")
        assert isinstance(s, dict)
        assert s["job_id"] == "s1"
        ex.shutdown(wait=False)

    def test_status_running(self):
        ex = TrainingExecutor(max_workers=2)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        ex.submit(blocker, "run_s")
        time.sleep(0.05)
        s = ex.status("run_s")
        assert s["status"] == "running"
        evt.set()
        ex.shutdown(wait=False)


# ── Result Summary ───────────────────────────────────────────────────────────

class TestResultSummary:
    def test_result_summary_unknown(self):
        ex = TrainingExecutor(max_workers=2)
        assert ex.result_summary("unknown") is None
        ex.shutdown(wait=False)

    def test_result_summary_not_completed(self):
        ex = TrainingExecutor(max_workers=2)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        ex.submit(blocker, "running_res")
        time.sleep(0.05)
        assert ex.result_summary("running_res") is None
        evt.set()
        ex.shutdown(wait=False)

    def test_result_summary_completed_with_weights(self):
        ex = TrainingExecutor(max_workers=2)
        def train(job_id):
            return {"w1": np.zeros(16, dtype=np.float32), "w2": np.ones(8, dtype=np.float64)}
        ex.submit(train, "res_w")
        time.sleep(0.1)
        summary = ex.result_summary("res_w")
        assert summary is not None
        assert summary["job_id"] == "res_w"
        assert "w1" in summary["weights"]
        assert summary["weights"]["w1"]["shape"] == [16]
        assert summary["weights"]["w1"]["dtype"] == "float32"
        assert summary["weights"]["w2"]["shape"] == [8]
        assert summary["total_bytes"] > 0
        ex.shutdown(wait=False)

    def test_result_summary_non_dict_result(self):
        ex = TrainingExecutor(max_workers=2)
        def train(job_id):
            return "not a dict"
        ex.submit(train, "res_nd")
        time.sleep(0.1)
        assert ex.result_summary("res_nd") is None
        ex.shutdown(wait=False)

    def test_result_summary_none_result(self):
        ex = TrainingExecutor(max_workers=2)
        def train(job_id):
            return None
        ex.submit(train, "res_none")
        time.sleep(0.1)
        assert ex.result_summary("res_none") is None
        ex.shutdown(wait=False)

    def test_result_summary_with_tree_id(self):
        ex = TrainingExecutor(max_workers=2)
        def train(job_id):
            return {"w": np.zeros(4)}
        ex.submit(train, "res_tree", tree_id="my_tree")
        time.sleep(0.1)
        summary = ex.result_summary("res_tree")
        assert summary["tree_id"] == "my_tree"
        ex.shutdown(wait=False)


# ── List Jobs ────────────────────────────────────────────────────────────────

class TestListJobs:
    def test_list_empty(self):
        ex = TrainingExecutor(max_workers=2)
        assert ex.list_jobs() == []
        ex.shutdown(wait=False)

    def test_list_returns_dicts(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "l1")
        ex.submit(noop, "l2")
        time.sleep(0.1)
        jobs = ex.list_jobs()
        assert len(jobs) == 2
        assert all(isinstance(j, dict) for j in jobs)
        ex.shutdown(wait=False)

    def test_list_newest_first(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "first")
        time.sleep(0.01)
        ex.submit(noop, "second")
        time.sleep(0.1)
        jobs = ex.list_jobs()
        assert jobs[0]["job_id"] == "second"
        assert jobs[1]["job_id"] == "first"
        ex.shutdown(wait=False)


# ── Cancel ───────────────────────────────────────────────────────────────────

class TestCancel:
    def test_cancel_unknown_returns_false(self):
        ex = TrainingExecutor(max_workers=2)
        assert ex.cancel("unknown") is False
        ex.shutdown(wait=False)

    def test_cancel_queued_job(self):
        ex = TrainingExecutor(max_workers=1)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        ex.submit(blocker, "blocker")
        time.sleep(0.05)
        job_id = ex.submit(blocker, "to_cancel")
        time.sleep(0.01)
        result = ex.cancel(job_id)
        assert result is True
        evt.set()
        ex.shutdown(wait=False)

    def test_cancel_running_sets_flag(self):
        ex = TrainingExecutor(max_workers=2)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        job_id = ex.submit(blocker, "run_cancel")
        time.sleep(0.05)
        result = ex.cancel(job_id)
        assert result is True
        assert ex._jobs[job_id].cancel_requested is True
        evt.set()
        ex.shutdown(wait=False)

    def test_is_cancelled(self):
        ex = TrainingExecutor(max_workers=2)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        job_id = ex.submit(blocker, "chk")
        time.sleep(0.05)
        assert ex.is_cancelled(job_id) is False
        ex.cancel(job_id)
        assert ex.is_cancelled(job_id) is True
        evt.set()
        ex.shutdown(wait=False)

    def test_is_cancelled_unknown_returns_false(self):
        ex = TrainingExecutor(max_workers=2)
        assert ex.is_cancelled("nope") is False
        ex.shutdown(wait=False)


# ── Purge ────────────────────────────────────────────────────────────────────

class TestPurge:
    def test_purge_removes_old_completed(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "old")
        time.sleep(0.1)
        ex._jobs["old"].completed_at = time.time() - 7200
        purged = ex.purge_completed(max_age_s=3600)
        assert purged == 1
        assert ex.status("old") is None
        ex.shutdown(wait=False)

    def test_purge_keeps_recent(self):
        ex = TrainingExecutor(max_workers=2)
        def noop(job_id):
            pass
        ex.submit(noop, "new")
        time.sleep(0.1)
        purged = ex.purge_completed(max_age_s=3600)
        assert purged == 0
        assert ex.status("new") is not None
        ex.shutdown(wait=False)

    def test_purge_empty(self):
        ex = TrainingExecutor(max_workers=2)
        purged = ex.purge_completed(max_age_s=1)
        assert purged == 0
        ex.shutdown(wait=False)

    def test_purge_keeps_failed(self):
        ex = TrainingExecutor(max_workers=2)
        def bad(job_id):
            raise RuntimeError("fail")
        ex.submit(bad, "fail_purge")
        time.sleep(0.1)
        ex._jobs["fail_purge"].completed_at = time.time() - 7200
        purged = ex.purge_completed(max_age_s=3600)
        assert purged == 1
        ex.shutdown(wait=False)

    def test_purge_keeps_running(self):
        ex = TrainingExecutor(max_workers=2)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        ex.submit(blocker, "running")
        time.sleep(0.05)
        purged = ex.purge_completed(max_age_s=0)
        assert purged == 0
        evt.set()
        ex.shutdown(wait=False)


# ── Active Count ─────────────────────────────────────────────────────────────

class TestActiveCount:
    def test_active_count_zero(self):
        ex = TrainingExecutor(max_workers=2)
        assert ex.active_count() == 0
        ex.shutdown(wait=False)

    def test_active_count_running(self):
        ex = TrainingExecutor(max_workers=2)
        evt = threading.Event()
        def blocker(job_id):
            evt.wait(timeout=2)
        ex.submit(blocker, "a1")
        time.sleep(0.05)
        assert ex.active_count() >= 1
        evt.set()
        ex.shutdown(wait=False)


# ── Shutdown ─────────────────────────────────────────────────────────────────

class TestShutdown:
    def test_shutdown_clears_singleton(self):
        ex = get_training_executor()
        assert ex is not None
        ex.shutdown(wait=False)
        import domains.training.executor as exec_mod
        assert exec_mod._instance is None
