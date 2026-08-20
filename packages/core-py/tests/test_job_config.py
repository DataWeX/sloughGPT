"""Tests for domains.collections.scheduler — JobConfig."""

from domains.collections.scheduler import JobConfig


class TestJobConfig:
    def test_fields(self):
        jc = JobConfig(name="test_job")
        assert jc.name == "test_job"
        assert jc.interval == 60.0
        assert jc.enabled is True
        assert jc.max_runs is None
        assert jc.timeout is None

    def test_custom(self):
        jc = JobConfig(name="fast_job", interval=10.0, enabled=False, max_runs=5)
        assert jc.interval == 10.0
        assert jc.enabled is False
        assert jc.max_runs == 5
