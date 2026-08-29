"""Tests for domains.training.executor — JobStatus, JobInfo."""

from domains.training.executor import JobStatus, JobInfo


class TestJobStatus:
    def test_all_members(self):
        assert len(JobStatus) == 5
    def test_values(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


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
