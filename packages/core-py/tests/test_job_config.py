"""Tests for domains.collections.scheduler — JobConfig and JobScheduler."""

from __future__ import annotations

import time
import threading
from dataclasses import FrozenInstanceError

import pytest

from domains.collections.scheduler import JobConfig, JobScheduler


# ---------------------------------------------------------------------------
# JobConfig — construction & defaults
# ---------------------------------------------------------------------------

class TestJobConfig:
    def test_defaults(self):
        jc = JobConfig(name="test_job")
        assert jc.name == "test_job"
        assert jc.interval == 60.0
        assert jc.enabled is True
        assert jc.max_runs is None
        assert jc.timeout is None
        assert jc.on_complete is None
        assert jc.on_error is None

    def test_custom_fields(self):
        jc = JobConfig(name="fast", interval=10.0, enabled=False, max_runs=5, timeout=30.0)
        assert jc.interval == 10.0
        assert jc.enabled is False
        assert jc.max_runs == 5
        assert jc.timeout == 30.0

    def test_callback_fields(self):
        calls = []
        jc = JobConfig(
            name="cb",
            on_complete=lambda name, count: calls.append(("ok", name, count)),
            on_error=lambda name, exc: calls.append(("err", name, exc)),
        )
        jc.on_complete("cb", 42)
        assert calls == [("ok", "cb", 42)]

    def test_zero_interval(self):
        jc = JobConfig(name="zero", interval=0.0)
        assert jc.interval == 0.0

    def test_negative_interval(self):
        jc = JobConfig(name="neg", interval=-5.0)
        assert jc.interval == -5.0

    def test_zero_max_runs(self):
        jc = JobConfig(name="zr", max_runs=0)
        assert jc.max_runs == 0

    def test_name_required(self):
        with pytest.raises(TypeError):
            JobConfig()

    def test_equality(self):
        a = JobConfig(name="x", interval=1.0)
        b = JobConfig(name="x", interval=1.0)
        assert a == b

    def test_inequality(self):
        a = JobConfig(name="x")
        b = JobConfig(name="y")
        assert a != b

    def test_timeout_default_none(self):
        jc = JobConfig(name="t")
        assert jc.timeout is None

    def test_timeout_custom(self):
        jc = JobConfig(name="t", timeout=99.5)
        assert jc.timeout == 99.5

    def test_enabled_toggle(self):
        a = JobConfig(name="a", enabled=True)
        b = JobConfig(name="a", enabled=False)
        assert a.enabled is not b.enabled

    def test_large_interval(self):
        jc = JobConfig(name="big", interval=999999.0)
        assert jc.interval == 999999.0

    def test_very_small_interval(self):
        jc = JobConfig(name="tiny", interval=0.001)
        assert jc.interval == 0.001

    def test_max_runs_one(self):
        jc = JobConfig(name="once", max_runs=1)
        assert jc.max_runs == 1

    def test_large_max_runs(self):
        jc = JobConfig(name="many", max_runs=10**6)
        assert jc.max_runs == 10**6

    def test_on_complete_is_none_by_default(self):
        jc = JobConfig(name="nc")
        assert jc.on_complete is None

    def test_on_error_is_none_by_default(self):
        jc = JobConfig(name="ne")
        assert jc.on_error is None

    def test_on_error_called(self):
        errors = []
        jc = JobConfig(
            name="err",
            on_error=lambda n, e: errors.append((n, str(e))),
        )
        jc.on_error("err", RuntimeError("boom"))
        assert errors == [("err", "boom")]

    def test_name_with_unicode(self):
        jc = JobConfig(name="日本語ジョブ")
        assert jc.name == "日本語ジョブ"

    def test_name_empty_string(self):
        jc = JobConfig(name="")
        assert jc.name == ""

    def test_name_with_spaces(self):
        jc = JobConfig(name="my job name")
        assert jc.name == "my job name"


# ---------------------------------------------------------------------------
# JobScheduler — add / remove / list
# ---------------------------------------------------------------------------

class TestJobSchedulerAddRemove:
    def test_add_job(self):
        sched = JobScheduler()
        jc = JobConfig(name="j1")
        sched.add_job(jc, _noop_collector())
        assert "j1" in sched.list_jobs()

    def test_add_returns_self(self):
        sched = JobScheduler()
        result = sched.add_job(JobConfig(name="j"), _noop_collector())
        assert result is sched

    def test_add_multiple(self):
        sched = JobScheduler()
        for i in range(5):
            sched.add_job(JobConfig(name=f"j{i}"), _noop_collector())
        assert sched.list_jobs() == [f"j{i}" for i in range(5)]

    def test_remove_existing(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        assert sched.remove_job("j1") is True
        assert sched.list_jobs() == []

    def test_remove_nonexistent(self):
        sched = JobScheduler()
        assert sched.remove_job("ghost") is False

    def test_remove_cleans_stats(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        sched.remove_job("j1")
        assert sched.job_stats("j1") is None

    def test_remove_cleans_collector(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        sched.remove_job("j1")
        assert sched.get_collector("j1") is None

    def test_get_collector(self):
        c = _noop_collector()
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), c)
        assert sched.get_collector("j1") is c

    def test_get_collector_missing(self):
        sched = JobScheduler()
        assert sched.get_collector("missing") is None

    def test_add_replaces_existing(self):
        sched = JobScheduler()
        c1 = _noop_collector()
        c2 = _CountingCollector()
        sched.add_job(JobConfig(name="j1"), c1)
        sched.add_job(JobConfig(name="j1"), c2)
        assert sched.get_collector("j1") is c2

    def test_remove_does_not_affect_others(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), _noop_collector())
        sched.add_job(JobConfig(name="b"), _noop_collector())
        sched.remove_job("a")
        assert "b" in sched.list_jobs()

    def test_list_jobs_empty(self):
        sched = JobScheduler()
        assert sched.list_jobs() == []

    def test_list_jobs_returns_copy(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        jobs = sched.list_jobs()
        jobs.append("extra")
        assert "extra" not in sched.list_jobs()

    def test_add_twice_same_name(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="dup"), _noop_collector())
        sched.add_job(JobConfig(name="dup"), _CountingCollector())
        assert sched.list_jobs().count("dup") == 1

    def test_remove_then_add(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        sched.remove_job("j1")
        sched.add_job(JobConfig(name="j1"), _CountingCollector())
        assert "j1" in sched.list_jobs()
        assert sched.get_collector("j1") is not None


# ---------------------------------------------------------------------------
# JobScheduler — start / stop / is_running
# ---------------------------------------------------------------------------

class TestJobSchedulerStartStop:
    def test_start_enabled_job(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=10, max_runs=1), _noop_collector())
        assert sched.start_job("j1") is True
        time.sleep(0.1)
        sched.stop_job("j1")

    def test_start_disabled_job(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", enabled=False), _noop_collector())
        assert sched.start_job("j1") is False

    def test_start_nonexistent_job(self):
        sched = JobScheduler()
        assert sched.start_job("ghost") is False

    def test_start_already_running(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=10, max_runs=100), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.05)
        assert sched.start_job("j1") is False
        sched.stop_job("j1")

    def test_stop_nonexistent(self):
        sched = JobScheduler()
        assert sched.stop_job("ghost") is False

    def test_stop_returns_true(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=10, max_runs=100), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.05)
        assert sched.stop_job("j1") is True

    def test_is_running(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=10, max_runs=100), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.05)
        assert sched.is_running("j1") is True
        sched.stop_job("j1")
        time.sleep(0.1)
        assert sched.is_running("j1") is False

    def test_is_running_nonexistent(self):
        sched = JobScheduler()
        assert sched.is_running("ghost") is False

    def test_start_then_stop_status(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=10, max_runs=100), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.05)
        stats = sched.job_stats("j1")
        assert stats["status"] == "running"
        sched.stop_job("j1")
        time.sleep(0.1)
        stats = sched.job_stats("j1")
        assert stats["status"] == "stopped"

    def test_stop_after_max_runs(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.01, max_runs=1), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.3)
        assert sched.is_running("j1") is False
        assert sched.stop_job("j1") is True

    def test_start_after_stop(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.05, max_runs=1), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.3)
        assert sched.is_running("j1") is False

    def test_is_running_returns_false_when_idle(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        assert sched.is_running("j1") is False


# ---------------------------------------------------------------------------
# JobScheduler — start_all / stop_all
# ---------------------------------------------------------------------------

class TestJobSchedulerBatch:
    def test_start_all(self):
        sched = JobScheduler()
        for i in range(3):
            sched.add_job(JobConfig(name=f"j{i}", interval=10, max_runs=100), _noop_collector())
        count = sched.start_all()
        assert count == 3
        sched.stop_all()

    def test_start_all_skips_disabled(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10, max_runs=100, enabled=True), _noop_collector())
        sched.add_job(JobConfig(name="b", interval=10, enabled=False), _noop_collector())
        sched.add_job(JobConfig(name="c", interval=10, max_runs=100, enabled=True), _noop_collector())
        count = sched.start_all()
        assert count == 2
        sched.stop_all()

    def test_stop_all(self):
        sched = JobScheduler()
        for i in range(3):
            sched.add_job(JobConfig(name=f"j{i}", interval=10, max_runs=100), _noop_collector())
        sched.start_all()
        time.sleep(0.05)
        count = sched.stop_all()
        assert count >= 1

    def test_stop_all_empty(self):
        sched = JobScheduler()
        assert sched.stop_all() == 0

    def test_start_all_empty(self):
        sched = JobScheduler()
        assert sched.start_all() == 0

    def test_start_all_returns_count(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10, max_runs=100), _noop_collector())
        sched.add_job(JobConfig(name="b", interval=10, max_runs=100), _noop_collector())
        assert sched.start_all() == 2
        sched.stop_all()

    def test_stop_all_after_start_all(self):
        sched = JobScheduler()
        for i in range(3):
            sched.add_job(JobConfig(name=f"j{i}", interval=10, max_runs=100), _noop_collector())
        sched.start_all()
        time.sleep(0.05)
        stopped = sched.stop_all()
        assert stopped == 3


# ---------------------------------------------------------------------------
# JobScheduler — stats
# ---------------------------------------------------------------------------

class TestJobSchedulerStats:
    def test_initial_stats(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        stats = sched.job_stats("j1")
        assert stats["runs"] == 0
        assert stats["total_collected"] == 0
        assert stats["errors"] == 0
        assert stats["last_run"] is None
        assert stats["last_duration"] is None
        assert stats["status"] == "idle"

    def test_stats_returns_copy(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        s1 = sched.stats()
        s2 = sched.stats()
        assert s1 is not s2
        assert s1 == s2

    def test_job_stats_nonexistent(self):
        sched = JobScheduler()
        assert sched.job_stats("missing") is None

    def test_stats_reflects_run(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.1, max_runs=2), _CountingCollector())
        sched.start_job("j1")
        time.sleep(0.5)
        sched.stop_job("j1")
        stats = sched.job_stats("j1")
        assert stats["runs"] >= 1
        assert stats["total_collected"] >= 1
        assert stats["last_run"] is not None
        assert stats["last_duration"] is not None

    def test_stats_dict_keys(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        stats = sched.job_stats("j1")
        expected_keys = {"runs", "total_collected", "errors", "last_run", "last_duration", "status"}
        assert set(stats.keys()) == expected_keys

    def test_stats_initial_status_idle(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        assert sched.job_stats("j1")["status"] == "idle"

    def test_stats_multiple_jobs(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), _noop_collector())
        sched.add_job(JobConfig(name="b"), _noop_collector())
        all_stats = sched.stats()
        assert "a" in all_stats
        assert "b" in all_stats

    def test_stats_returns_dict(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), _noop_collector())
        result = sched.stats()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# JobScheduler — max_runs
# ---------------------------------------------------------------------------

class TestJobSchedulerMaxRuns:
    def test_max_runs_stops_job(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.05, max_runs=3), _noop_collector())
        sched.start_job("j1")
        time.sleep(1.0)
        stats = sched.job_stats("j1")
        assert stats["runs"] == 3
        assert stats["status"] == "stopped"

    def test_max_runs_one(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.01, max_runs=1), _CountingCollector())
        sched.start_job("j1")
        time.sleep(0.3)
        stats = sched.job_stats("j1")
        assert stats["runs"] == 1

    def test_max_runs_five(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.01, max_runs=5), _CountingCollector())
        sched.start_job("j1")
        time.sleep(1.0)
        stats = sched.job_stats("j1")
        assert stats["runs"] == 5

    def test_no_max_runs_keeps_going(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.01, max_runs=None), _CountingCollector())
        sched.start_job("j1")
        time.sleep(0.3)
        stats = sched.job_stats("j1")
        sched.stop_job("j1")
        assert stats["runs"] >= 5


# ---------------------------------------------------------------------------
# JobScheduler — callbacks
# ---------------------------------------------------------------------------

class TestJobSchedulerCallbacks:
    def test_on_complete_called(self):
        completed = []
        sched = JobScheduler()
        sched.add_job(
            JobConfig(name="j1", interval=0.05, max_runs=1,
                      on_complete=lambda n, c: completed.append((n, c))),
            _CountingCollector(),
        )
        sched.start_job("j1")
        time.sleep(0.3)
        assert len(completed) >= 1
        assert completed[0][0] == "j1"

    def test_on_error_called(self):
        errors = []

        def _fail():
            raise RuntimeError("boom")

        sched = JobScheduler()
        sched.add_job(
            JobConfig(name="j1", interval=0.05, max_runs=1,
                      on_error=lambda n, e: errors.append((n, str(e)))),
            _FailCollector(),
        )
        sched.start_job("j1")
        time.sleep(0.3)
        assert len(errors) >= 1
        assert "boom" in errors[0][1]

    def test_on_complete_receives_count(self):
        received = []
        sched = JobScheduler()
        sched.add_job(
            JobConfig(name="j1", interval=0.05, max_runs=1,
                      on_complete=lambda n, c: received.append(c)),
            _CountingCollector(),
        )
        sched.start_job("j1")
        time.sleep(0.3)
        assert len(received) >= 1
        assert received[0] >= 1

    def test_on_error_receives_exception(self):
        received = []
        sched = JobScheduler()
        sched.add_job(
            JobConfig(name="j1", interval=0.05, max_runs=1,
                      on_error=lambda n, e: received.append(type(e).__name__)),
            _FailCollector(),
        )
        sched.start_job("j1")
        time.sleep(0.3)
        assert "RuntimeError" in received

    def test_no_callbacks_no_crash(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.05, max_runs=1), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.2)
        stats = sched.job_stats("j1")
        assert stats["runs"] >= 1

    def test_error_increments_error_count(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=0.01, max_runs=3), _FailCollector())
        sched.start_job("j1")
        time.sleep(0.5)
        stats = sched.job_stats("j1")
        assert stats["errors"] >= 1


# ---------------------------------------------------------------------------
# JobScheduler — remove while running
# ---------------------------------------------------------------------------

class TestJobSchedulerRemoveWhileRunning:
    def test_remove_stops_running_job(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1", interval=10, max_runs=100), _noop_collector())
        sched.start_job("j1")
        time.sleep(0.05)
        assert sched.remove_job("j1") is True
        assert sched.list_jobs() == []

    def test_remove_nonexistent_does_not_crash(self):
        sched = JobScheduler()
        assert sched.remove_job("nope") is False

    def test_remove_multiple_running(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10, max_runs=100), _noop_collector())
        sched.add_job(JobConfig(name="b", interval=10, max_runs=100), _noop_collector())
        sched.start_all()
        time.sleep(0.05)
        assert sched.remove_job("a") is True
        assert sched.remove_job("b") is True
        assert sched.list_jobs() == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _noop_collector:
    def collect(self):
        return 0


class _CountingCollector:
    def __init__(self):
        self._count = 0

    def collect(self):
        self._count += 1
        return self._count


class _FailCollector:
    def collect(self):
        raise RuntimeError("boom")
