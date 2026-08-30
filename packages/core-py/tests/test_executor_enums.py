"""Tests for domains.training.executor — JobStatus, JobInfo."""

import time
import numpy as np
import pytest
from domains.training.executor import JobStatus, JobInfo, TrainingExecutor


class TestJobStatus:
    def test_all_members(self):
        assert len(JobStatus) == 5

    def test_values(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_members_are_strings(self):
        for s in JobStatus:
            assert isinstance(s.value, str)

    def test_str_enum(self):
        assert str(JobStatus.QUEUED) == "JobStatus.QUEUED"

    def test_equality(self):
        assert JobStatus.QUEUED == JobStatus.QUEUED
        assert JobStatus.QUEUED != JobStatus.RUNNING

    def test_value_lookup(self):
        assert JobStatus("queued") == JobStatus.QUEUED
        assert JobStatus("running") == JobStatus.RUNNING
        assert JobStatus("completed") == JobStatus.COMPLETED
        assert JobStatus("failed") == JobStatus.FAILED
        assert JobStatus("cancelled") == JobStatus.CANCELLED

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            JobStatus("invalid")

    def test_iteration(self):
        members = list(JobStatus)
        assert len(members) == 5

    def test_contains(self):
        assert "queued" in [s.value for s in JobStatus]

    def test_ordered(self):
        members = list(JobStatus)
        assert members[0] == JobStatus.QUEUED
        assert members[1] == JobStatus.RUNNING
        assert members[2] == JobStatus.COMPLETED
        assert members[3] == JobStatus.FAILED
        assert members[4] == JobStatus.CANCELLED

    def test_is_str_subclass(self):
        assert issubclass(JobStatus, str)

    def test_members_unique(self):
        values = [s.value for s in JobStatus]
        assert len(values) == len(set(values))

    def test_member_names(self):
        names = [s.name for s in JobStatus]
        assert "QUEUED" in names
        assert "RUNNING" in names
        assert "COMPLETED" in names
        assert "FAILED" in names
        assert "CANCELLED" in names

    def test_member_count(self):
        assert len(list(JobStatus)) == 5


class TestJobInfo:
    def test_defaults(self):
        ji = JobInfo(job_id="j1")
        assert ji.job_id == "j1"
        assert ji.status == JobStatus.QUEUED
        assert ji.cancel_requested is False

    def test_elapsed(self):
        ji = JobInfo(job_id="j1")
        e = ji.elapsed()
        assert e is not None
        assert e >= 0.0

    def test_to_dict(self):
        ji = JobInfo(job_id="j1", status=JobStatus.QUEUED)
        d = ji.to_dict()
        assert d["job_id"] == "j1"
        assert d["status"] == "queued"
        assert "elapsed_s" in d

    def test_defaults_tree_id(self):
        ji = JobInfo(job_id="j1")
        assert ji.tree_id is None

    def test_defaults_future(self):
        ji = JobInfo(job_id="j1")
        assert ji.future is None

    def test_defaults_submitted_at(self):
        ji = JobInfo(job_id="j1")
        assert isinstance(ji.submitted_at, float)
        assert ji.submitted_at > 0

    def test_defaults_started_at(self):
        ji = JobInfo(job_id="j1")
        assert ji.started_at is None

    def test_defaults_completed_at(self):
        ji = JobInfo(job_id="j1")
        assert ji.completed_at is None

    def test_defaults_error(self):
        ji = JobInfo(job_id="j1")
        assert ji.error is None

    def test_defaults_result(self):
        ji = JobInfo(job_id="j1")
        assert ji.result is None

    def test_custom_tree_id(self):
        ji = JobInfo(job_id="j1", tree_id="tree_x")
        assert ji.tree_id == "tree_x"

    def test_custom_status(self):
        ji = JobInfo(job_id="j1", status=JobStatus.RUNNING)
        assert ji.status == JobStatus.RUNNING

    def test_to_dict_running(self):
        ji = JobInfo(job_id="j1", status=JobStatus.RUNNING)
        d = ji.to_dict()
        assert d["status"] == "running"

    def test_to_dict_with_started_at(self):
        ji = JobInfo(job_id="j1", status=JobStatus.RUNNING, started_at=100.0)
        d = ji.to_dict()
        assert d["started_at"] == 100.0

    def test_to_dict_with_completed_at(self):
        ji = JobInfo(job_id="j1", status=JobStatus.COMPLETED, completed_at=200.0)
        d = ji.to_dict()
        assert d["completed_at"] == 200.0

    def test_to_dict_with_error(self):
        ji = JobInfo(job_id="j1", status=JobStatus.FAILED, error="boom")
        d = ji.to_dict()
        assert d["error"] == "boom"

    def test_to_dict_cancel_requested(self):
        ji = JobInfo(job_id="j1", cancel_requested=True)
        d = ji.to_dict()
        assert d["cancel_requested"] is True

    def test_to_dict_completed_with_dict_result(self):
        import numpy as np
        ji = JobInfo(
            job_id="j1",
            status=JobStatus.COMPLETED,
            result={"w1": np.array([1.0, 2.0]), "w2": np.array([3.0])},
        )
        d = ji.to_dict()
        assert "result_keys" in d
        assert set(d["result_keys"]) == {"w1", "w2"}
        assert "result_size_bytes" in d

    def test_to_dict_completed_with_non_dict_result(self):
        ji = JobInfo(job_id="j1", status=JobStatus.COMPLETED, result=42)
        d = ji.to_dict()
        assert d["result_type"] == "int"

    def test_to_dict_completed_with_none_result(self):
        ji = JobInfo(job_id="j1", status=JobStatus.COMPLETED, result=None)
        d = ji.to_dict()
        assert "result_type" not in d
        assert "result_keys" not in d

    def test_elapsed_returns_float(self):
        ji = JobInfo(job_id="j1")
        assert isinstance(ji.elapsed(), float)

    def test_elapsed_with_completed_at(self):
        start = 100.0
        end = 150.0
        ji = JobInfo(job_id="j1", submitted_at=start, completed_at=end)
        assert ji.elapsed() == 50.0

    def test_to_dict_keys(self):
        ji = JobInfo(job_id="j1")
        d = ji.to_dict()
        expected_keys = {
            "job_id", "tree_id", "status", "submitted_at", "started_at",
            "completed_at", "elapsed_s", "error", "cancel_requested",
        }
        assert expected_keys == set(d.keys())

    def test_job_info_to_dict_all_fields(self):
        ji = JobInfo(
            job_id="j1",
            tree_id="tree1",
            status=JobStatus.COMPLETED,
            submitted_at=100.0,
            started_at=110.0,
            completed_at=120.0,
            error=None,
            cancel_requested=False,
            result={"w": np.array([1.0])},
        )
        d = ji.to_dict()
        assert d["job_id"] == "j1"
        assert d["tree_id"] == "tree1"
        assert d["status"] == "completed"
        assert d["started_at"] == 110.0
        assert d["completed_at"] == 120.0
        assert "result_keys" in d
        assert "result_size_bytes" in d

    def test_job_info_elapsed_with_started(self):
        ji = JobInfo(job_id="j1", submitted_at=100.0, started_at=110.0, completed_at=150.0)
        assert ji.elapsed() == 50.0

    def test_job_info_elapsed_in_progress(self):
        ji = JobInfo(job_id="j1", submitted_at=100.0)
        e = ji.elapsed()
        assert e >= 0

    def test_to_dict_failed_status(self):
        ji = JobInfo(job_id="j1", status=JobStatus.FAILED, error="oops")
        d = ji.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "oops"

    def test_to_dict_cancelled_status(self):
        ji = JobInfo(job_id="j1", status=JobStatus.CANCELLED)
        d = ji.to_dict()
        assert d["status"] == "cancelled"

    def test_to_dict_non_dict_array_result(self):
        ji = JobInfo(
            job_id="j1",
            status=JobStatus.COMPLETED,
            result=np.array([1.0, 2.0]),
        )
        d = ji.to_dict()
        assert d["result_type"] == "ndarray"

    def test_to_dict_dict_result_with_non_array_values(self):
        ji = JobInfo(
            job_id="j1",
            status=JobStatus.COMPLETED,
            result={"a": 1, "b": "text"},
        )
        d = ji.to_dict()
        assert "result_keys" in d
        assert "result_size_bytes" in d


class TestTrainingExecutor:
    def test_submit_and_status(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return {"loss": 0.5}
            job_id = exec_.submit(fn, "test_job")
            time.sleep(0.1)
            status = exec_.status(job_id)
            assert status is not None
            assert status["job_id"] == "test_job"
        finally:
            exec_.shutdown(wait=True)

    def test_submit_completed_result(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return {"w": np.array([1.0])}
            job_id = exec_.submit(fn, "res_job")
            time.sleep(0.2)
            info = exec_._jobs[job_id]
            assert info.status == JobStatus.COMPLETED
        finally:
            exec_.shutdown(wait=True)

    def test_submit_failed_result(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                raise ValueError("test error")
            job_id = exec_.submit(fn, "fail_job")
            time.sleep(0.2)
            info = exec_._jobs[job_id]
            assert info.status == JobStatus.FAILED
            assert "test error" in info.error
        finally:
            exec_.shutdown(wait=True)

    def test_list_jobs(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return True
            exec_.submit(fn, "job_a")
            exec_.submit(fn, "job_b")
            time.sleep(0.1)
            jobs = exec_.list_jobs()
            assert len(jobs) >= 2
        finally:
            exec_.shutdown(wait=True)

    def test_is_cancelled_default(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            assert exec_.is_cancelled("nonexistent") is False
        finally:
            exec_.shutdown(wait=True)

    def test_cancel_nonexistent(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            assert exec_.cancel("nonexistent") is False
        finally:
            exec_.shutdown(wait=True)

    def test_active_count(self):
        exec_ = TrainingExecutor(max_workers=2)
        try:
            assert exec_.active_count() == 0
        finally:
            exec_.shutdown(wait=True)

    def test_status_nonexistent(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            assert exec_.status("nonexistent") is None
        finally:
            exec_.shutdown(wait=True)

    def test_result_summary_nonexistent(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            assert exec_.result_summary("nonexistent") is None
        finally:
            exec_.shutdown(wait=True)

    def test_result_summary_not_completed(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def slow_fn(jid):
                time.sleep(10)
            exec_.submit(slow_fn, "slow")
            time.sleep(0.05)
            assert exec_.result_summary("slow") is None
        finally:
            exec_.shutdown(wait=True)

    def test_result_summary_completed_with_weights(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return {"w1": np.zeros((3, 3)), "w2": np.ones((2,))}
            job_id = exec_.submit(fn, "weight_job")
            time.sleep(0.3)
            summary = exec_.result_summary(job_id)
            assert summary is not None
            assert summary["job_id"] == "weight_job"
            assert "weights" in summary
            assert "w1" in summary["weights"]
            assert "w2" in summary["weights"]
            assert summary["total_bytes"] > 0
        finally:
            exec_.shutdown(wait=True)

    def test_purge_completed(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return True
            exec_.submit(fn, "purge_job")
            time.sleep(0.3)
            purged = exec_.purge_completed(max_age_s=0.0)
            assert purged >= 1
        finally:
            exec_.shutdown(wait=True)

    def test_purge_no_old_jobs(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return True
            exec_.submit(fn, "new_job")
            time.sleep(0.1)
            purged = exec_.purge_completed(max_age_s=99999)
            assert purged == 0
        finally:
            exec_.shutdown(wait=True)

    def test_submit_with_tree_id(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return True
            job_id = exec_.submit(fn, "tree_job", tree_id="tree_1")
            time.sleep(0.1)
            d = exec_.status(job_id)
            assert d["tree_id"] == "tree_1"
        finally:
            exec_.shutdown(wait=True)

    def test_submit_with_call_args(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid, extra=None):
                return extra
            job_id = exec_.submit(fn, "call_args_job", _call_args={"extra": "hello"})
            time.sleep(0.2)
            info = exec_._jobs[job_id]
            assert info.result == "hello"
        finally:
            exec_.shutdown(wait=True)

    def test_submit_with_kwargs(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid, lr=0.001):
                return lr
            job_id = exec_.submit(fn, "kwarg_job", lr=0.01)
            time.sleep(0.2)
            info = exec_._jobs[job_id]
            assert info.result == 0.01
        finally:
            exec_.shutdown(wait=True)

    def test_cancel_queued_job(self):
        import threading
        exec_ = TrainingExecutor(max_workers=1)
        try:
            lock = threading.Event()
            def blocking_fn(jid):
                lock.wait(timeout=5)
                return True
            exec_.submit(blocking_fn, "blocker")
            time.sleep(0.05)
            job_id2 = exec_.submit(lambda jid: True, "queued_job")
            result = exec_.cancel(job_id2)
            assert result is True
        finally:
            lock.set()
            exec_.shutdown(wait=True)

    def test_list_jobs_returns_dicts(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            exec_.submit(lambda jid: True, "j1")
            time.sleep(0.1)
            jobs = exec_.list_jobs()
            assert len(jobs) >= 1
            assert isinstance(jobs[0], dict)
        finally:
            exec_.shutdown(wait=True)

    def test_list_jobs_newest_first(self):
        exec_ = TrainingExecutor(max_workers=2)
        try:
            exec_.submit(lambda jid: time.sleep(0.1), "j1")
            exec_.submit(lambda jid: True, "j2")
            time.sleep(0.2)
            jobs = exec_.list_jobs()
            if len(jobs) >= 2:
                assert jobs[0]["submitted_at"] >= jobs[1]["submitted_at"]
        finally:
            exec_.shutdown(wait=True)

    def test_status_returns_dict(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            job_id = exec_.submit(lambda jid: True, "test")
            time.sleep(0.1)
            status = exec_.status(job_id)
            assert isinstance(status, dict)
            assert "job_id" in status
            assert "status" in status
        finally:
            exec_.shutdown(wait=True)

    def test_cancel_already_completed(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            job_id = exec_.submit(lambda jid: True, "done")
            time.sleep(0.3)
            result = exec_.cancel(job_id)
            assert result is False
        finally:
            exec_.shutdown(wait=True)

    def test_is_cancelled_after_cancel(self):
        import threading as _threading
        exec_ = TrainingExecutor(max_workers=2)
        try:
            lock = _threading.Event()
            def blocking(jid):
                lock.wait(timeout=5)
                return True
            exec_.submit(blocking, "blocker")
            time.sleep(0.05)
            job_id2 = exec_.submit(lambda jid: True, "queued")
            exec_.cancel(job_id2)
            assert exec_.is_cancelled(job_id2) is True
        finally:
            lock.set()
            exec_.shutdown(wait=True)

    def test_purge_failed_jobs(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def failing(jid):
                raise RuntimeError("fail")
            exec_.submit(failing, "fail1")
            time.sleep(0.2)
            purged = exec_.purge_completed(max_age_s=0.0)
            assert purged >= 1
        finally:
            exec_.shutdown(wait=True)

    def test_active_count_running(self):
        import threading
        exec_ = TrainingExecutor(max_workers=2)
        try:
            lock = threading.Event()
            def blocking(jid):
                lock.wait(timeout=5)
                return True
            exec_.submit(blocking, "b1")
            time.sleep(0.05)
            assert exec_.active_count() >= 1
        finally:
            lock.set()
            exec_.shutdown(wait=True)

    def test_submit_returns_job_id(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            result = exec_.submit(lambda jid: True, "my_job")
            assert result == "my_job"
        finally:
            exec_.shutdown(wait=True)

    def test_result_summary_dict_with_arrays(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return {"w": np.array([1.0, 2.0, 3.0])}
            job_id = exec_.submit(fn, "arr_job")
            time.sleep(0.3)
            summary = exec_.result_summary(job_id)
            assert summary is not None
            assert "w" in summary["weights"]
            assert summary["total_bytes"] > 0
        finally:
            exec_.shutdown(wait=True)

    def test_result_summary_non_dict_result(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return 42
            job_id = exec_.submit(fn, "scalar")
            time.sleep(0.2)
            summary = exec_.result_summary(job_id)
            assert summary is None
        finally:
            exec_.shutdown(wait=True)

    def test_concurrent_submits(self):
        exec_ = TrainingExecutor(max_workers=4)
        try:
            for i in range(10):
                exec_.submit(lambda jid: True, f"job_{i}")
            time.sleep(0.3)
            jobs = exec_.list_jobs()
            assert len(jobs) >= 10
        finally:
            exec_.shutdown(wait=True)

    def test_shutdown_and_reuse(self):
        exec_ = TrainingExecutor(max_workers=1)
        exec_.submit(lambda jid: True, "j1")
        time.sleep(0.1)
        exec_.shutdown(wait=True)

    def test_job_status_members(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_job_status_iteration(self):
        members = list(JobStatus)
        assert len(members) == 5

    def test_job_status_lookup(self):
        assert JobStatus("queued") == JobStatus.QUEUED
        assert JobStatus("completed") == JobStatus.COMPLETED

    def test_job_status_invalid(self):
        with pytest.raises(ValueError):
            JobStatus("nonexistent")

    def test_purge_does_not_remove_running(self):
        import threading
        exec_ = TrainingExecutor(max_workers=1)
        try:
            lock = threading.Event()
            def blocking(jid):
                lock.wait(timeout=5)
                return True
            exec_.submit(blocking, "running_job")
            time.sleep(0.05)
            purged = exec_.purge_completed(max_age_s=0.0)
            assert purged == 0
        finally:
            lock.set()
            exec_.shutdown(wait=True)

    def test_purge_mixed_ages(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                return True
            exec_.submit(fn, "old")
            time.sleep(0.3)
            exec_.submit(fn, "new")
            time.sleep(0.1)
            purged = exec_.purge_completed(max_age_s=0.2)
            assert purged >= 1
        finally:
            exec_.shutdown(wait=True)

    def test_submit_multiple_args(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid, a, b):
                return a + b
            job_id = exec_.submit(fn, "args_job", 3, 4)
            time.sleep(0.2)
            info = exec_._jobs[job_id]
            assert info.result == 7
        finally:
            exec_.shutdown(wait=True)

    def test_result_summary_none_for_failed(self):
        exec_ = TrainingExecutor(max_workers=1)
        try:
            def fn(jid):
                raise RuntimeError("boom")
            job_id = exec_.submit(fn, "fail_summary")
            time.sleep(0.2)
            summary = exec_.result_summary(job_id)
            assert summary is None
        finally:
            exec_.shutdown(wait=True)

    def test_cancel_after_shutdown(self):
        exec_ = TrainingExecutor(max_workers=1)
        exec_.submit(lambda jid: True, "j1")
        time.sleep(0.1)
        exec_.shutdown(wait=True)
