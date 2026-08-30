from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from domains.collections.scheduler import (
    JobConfig,
    JobScheduler,
    CollectorMonitor,
    CollectorExporter,
)
from domains.collections.collector import Collector
from domains.collections.sources import Record, Source
from domains.collections.stores import MemoryStore, FileStore
from domains.collections.validators import CollectorRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubSource:
    def __init__(self, records: list[Record] | None = None, name: str = "stub"):
        self.name = name
        self._records = records or []

    def read(self):
        yield from self._records


class CountingSource:
    """Yields incrementing records, tracks total reads."""
    def __init__(self, count_per_read: int = 1, name: str = "counting"):
        self.name = name
        self._count_per_read = count_per_read
        self.total_reads = 0

    def read(self):
        self.total_reads += 1
        for i in range(self._count_per_read):
            yield Record(content=f"r{self.total_reads}_{i}")


class FailingSource:
    name = "fail"

    def read(self):
        raise RuntimeError("boom")
        yield


def make_collector(records=None, store=None, name="stub"):
    src = StubSource(records or [], name=name)
    st = store or MemoryStore()
    return Collector(src, st)


# ===========================================================================
# JobConfig
# ===========================================================================

class TestJobConfig:
    def test_defaults(self):
        cfg = JobConfig(name="j1")
        assert cfg.interval == 60.0
        assert cfg.enabled is True
        assert cfg.max_runs is None
        assert cfg.timeout is None
        assert cfg.on_complete is None
        assert cfg.on_error is None

    def test_custom_values(self):
        cb = lambda n, c: None
        cfg = JobConfig(
            name="j2", interval=5.0, enabled=False, max_runs=10,
            timeout=30.0, on_complete=cb, on_error=cb,
        )
        assert cfg.interval == 5.0
        assert cfg.enabled is False
        assert cfg.max_runs == 10
        assert cfg.timeout == 30.0

    def test_dataclass_identity(self):
        cfg1 = JobConfig(name="x")
        cfg2 = JobConfig(name="x")
        assert cfg1 == cfg2


# ===========================================================================
# JobScheduler
# ===========================================================================

class TestJobSchedulerAddRemove:
    def test_add_job(self):
        sched = JobScheduler()
        col = make_collector()
        cfg = JobConfig(name="a")
        sched.add_job(cfg, col)
        assert "a" in sched.list_jobs()

    def test_add_job_returns_self(self):
        sched = JobScheduler()
        col = make_collector()
        result = sched.add_job(JobConfig(name="x"), col)
        assert result is sched

    def test_add_multiple_jobs(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), make_collector())
        sched.add_job(JobConfig(name="b"), make_collector())
        assert set(sched.list_jobs()) == {"a", "b"}

    def test_remove_existing_job(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), make_collector())
        assert sched.remove_job("a") is True
        assert sched.list_jobs() == []

    def test_remove_nonexistent_job(self):
        sched = JobScheduler()
        assert sched.remove_job("nope") is False

    def test_remove_stops_running_job(self):
        sched = JobScheduler()
        col = make_collector([Record(content="r")])
        sched.add_job(JobConfig(name="a", interval=10.0), col)
        sched.start_job("a")
        time.sleep(0.05)
        sched.remove_job("a")
        time.sleep(0.1)
        assert not sched.is_running("a")

    def test_get_collector(self):
        col = make_collector()
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), col)
        assert sched.get_collector("a") is col

    def test_get_collector_nonexistent(self):
        sched = JobScheduler()
        assert sched.get_collector("nope") is None


class TestJobSchedulerStartStop:
    def test_start_job(self):
        sched = JobScheduler()
        col = make_collector([Record(content="r")])
        sched.add_job(JobConfig(name="a", interval=0.1), col)
        assert sched.start_job("a") is True
        time.sleep(0.05)
        assert sched.is_running("a")
        sched.stop_all()

    def test_start_already_running(self):
        sched = JobScheduler()
        col = make_collector([Record(content="r")])
        sched.add_job(JobConfig(name="a", interval=10.0), col)
        sched.start_job("a")
        time.sleep(0.05)
        assert sched.start_job("a") is False
        sched.stop_all()

    def test_start_disabled_job(self):
        sched = JobScheduler()
        col = make_collector()
        sched.add_job(JobConfig(name="a", enabled=False), col)
        assert sched.start_job("a") is False

    def test_start_nonexistent_job(self):
        sched = JobScheduler()
        assert sched.start_job("nope") is False

    def test_stop_job(self):
        sched = JobScheduler()
        col = make_collector([Record(content="r")])
        sched.add_job(JobConfig(name="a", interval=10.0), col)
        sched.start_job("a")
        time.sleep(0.05)
        assert sched.stop_job("a") is True
        time.sleep(0.1)
        assert not sched.is_running("a")

    def test_stop_nonexistent(self):
        sched = JobScheduler()
        assert sched.stop_job("nope") is False

    def test_stop_all(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10.0), make_collector([Record(content="r")]))
        sched.add_job(JobConfig(name="b", interval=10.0), make_collector([Record(content="r")]))
        sched.start_all()
        time.sleep(0.05)
        count = sched.stop_all()
        assert count >= 2

    def test_start_all(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10.0), make_collector([Record(content="r")]))
        sched.add_job(JobConfig(name="b", interval=10.0), make_collector([Record(content="r")]))
        started = sched.start_all()
        assert started == 2
        sched.stop_all()


class TestJobSchedulerMaxRuns:
    def test_max_runs_stops(self):
        sched = JobScheduler()
        src = CountingSource(count_per_read=1)
        col = Collector(src, MemoryStore())
        sched.add_job(JobConfig(name="a", interval=0.0, max_runs=3), col)
        sched.start_job("a")
        time.sleep(0.5)
        assert not sched.is_running("a")
        stats = sched.job_stats("a")
        assert stats["runs"] == 3

    def test_no_max_runs_keeps_running(self):
        sched = JobScheduler()
        col = make_collector([Record(content="r")])
        sched.add_job(JobConfig(name="a", interval=0.01, max_runs=None), col)
        sched.start_job("a")
        time.sleep(0.15)
        assert sched.is_running("a")
        sched.stop_all()


class TestJobSchedulerCallbacks:
    def test_on_complete_called(self):
        completed = []
        sched = JobScheduler()
        col = make_collector([Record(content="data")])
        cfg = JobConfig(name="a", interval=0.0, max_runs=1,
                        on_complete=lambda n, c: completed.append((n, c)))
        sched.add_job(cfg, col)
        sched.start_job("a")
        time.sleep(0.3)
        assert len(completed) == 1
        assert completed[0][0] == "a"
        assert completed[0][1] == 1

    def test_on_error_called(self):
        errors = []
        sched = JobScheduler()
        col = make_collector()  # empty source, no issue
        # Make a source that always fails
        bad_col = Collector(FailingSource(), MemoryStore())
        cfg = JobConfig(name="fail", interval=0.0, max_runs=1,
                        on_error=lambda n, e: errors.append((n, str(e))))
        sched.add_job(cfg, bad_col)
        sched.start_job("fail")
        time.sleep(0.3)
        assert len(errors) == 1
        assert errors[0][0] == "fail"
        assert "boom" in errors[0][1]


class TestJobSchedulerStats:
    def test_stats_initial(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), make_collector())
        s = sched.stats()
        assert "a" in s
        assert s["a"]["runs"] == 0
        assert s["a"]["status"] == "idle"

    def test_job_stats(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), make_collector())
        s = sched.job_stats("a")
        assert s is not None
        assert s["runs"] == 0

    def test_job_stats_nonexistent(self):
        sched = JobScheduler()
        assert sched.job_stats("nope") is None

    def test_stats_after_collect(self):
        sched = JobScheduler()
        col = make_collector([Record(content="r")])
        sched.add_job(JobConfig(name="a", interval=0.0, max_runs=1), col)
        sched.start_job("a")
        time.sleep(0.3)
        s = sched.job_stats("a")
        assert s["runs"] == 1
        assert s["total_collected"] == 1
        assert s["last_run"] is not None
        assert s["last_duration"] is not None

    def test_stats_returns_copy(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), make_collector())
        s1 = sched.stats()
        s2 = sched.stats()
        assert s1 is not s2

    def test_stats_error_count(self):
        sched = JobScheduler()
        col = Collector(FailingSource(), MemoryStore())
        sched.add_job(JobConfig(name="fail", interval=0.0, max_runs=1), col)
        sched.start_job("fail")
        time.sleep(0.3)
        s = sched.job_stats("fail")
        assert s["errors"] == 1


class TestJobSchedulerIsRunning:
    def test_is_running_false_initially(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a"), make_collector())
        assert sched.is_running("a") is False

    def test_is_running_true_when_started(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10.0), make_collector([Record(content="r")]))
        sched.start_job("a")
        time.sleep(0.05)
        assert sched.is_running("a") is True
        sched.stop_all()

    def test_is_running_nonexistent(self):
        sched = JobScheduler()
        assert sched.is_running("nope") is False


# ===========================================================================
# CollectorMonitor
# ===========================================================================

class TestCollectorMonitorHealth:
    def test_no_health_checks(self):
        mon = CollectorMonitor()
        assert mon.check_health() == {}

    def test_health_check_pass(self):
        mon = CollectorMonitor()
        mon.add_health_check("ok", lambda: True)
        result = mon.check_health()
        assert result["ok"] is True

    def test_health_check_fail(self):
        mon = CollectorMonitor()
        mon.add_health_check("fail", lambda: False)
        result = mon.check_health()
        assert result["fail"] is False

    def test_health_check_exception(self):
        mon = CollectorMonitor()
        def bad_check():
            raise RuntimeError("oops")
        mon.add_health_check("err", bad_check)
        result = mon.check_health()
        assert result["err"] is False

    def test_add_health_check_returns_self(self):
        mon = CollectorMonitor()
        result = mon.add_health_check("x", lambda: True)
        assert result is mon

    def test_multiple_health_checks(self):
        mon = CollectorMonitor()
        mon.add_health_check("a", lambda: True)
        mon.add_health_check("b", lambda: False)
        result = mon.check_health()
        assert result["a"] is True
        assert result["b"] is False


class TestCollectorMonitorOverview:
    def test_overview_without_runner_or_scheduler(self):
        mon = CollectorMonitor()
        ov = mon.get_overview()
        assert "timestamp" in ov
        assert ov["healthy"] is True
        assert ov["components"] == {}

    def test_overview_with_scheduler(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j1"), make_collector())
        mon = CollectorMonitor(scheduler=sched)
        ov = mon.get_overview()
        assert "scheduler" in ov["components"]
        assert ov["components"]["scheduler"]["jobs"] == 1

    def test_overview_healthy_false_when_check_fails(self):
        mon = CollectorMonitor()
        mon.add_health_check("bad", lambda: False)
        ov = mon.get_overview()
        assert ov["healthy"] is False

    def test_overview_healthy_true_all_pass(self):
        mon = CollectorMonitor()
        mon.add_health_check("good", lambda: True)
        ov = mon.get_overview()
        assert ov["healthy"] is True

    def test_overview_with_runner(self):
        runner = CollectorRunner()
        runner.add("c1", make_collector([Record(content="r")]))
        runner.run("c1")
        mon = CollectorMonitor(runner=runner)
        ov = mon.get_overview()
        assert "runner" in ov["components"]
        assert ov["components"]["runner"]["collectors"] == 1

    def test_overview_scheduler_running_count(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="a", interval=10.0), make_collector([Record(content="r")]))
        sched.start_job("a")
        time.sleep(0.05)
        mon = CollectorMonitor(scheduler=sched)
        ov = mon.get_overview()
        assert ov["components"]["scheduler"]["running"] >= 1
        sched.stop_all()


class TestCollectorMonitorAlerts:
    def test_no_alerts_when_healthy(self):
        mon = CollectorMonitor()
        mon.add_health_check("ok", lambda: True)
        alerts = mon.check_alerts()
        assert alerts == []

    def test_alerts_when_unhealthy(self):
        mon = CollectorMonitor()
        mon.add_health_check("bad", lambda: False)
        alerts = mon.check_alerts()
        assert len(alerts) >= 1
        severities = [a["severity"] for a in alerts]
        assert "critical" in severities or "warning" in severities
        check_alerts = [a for a in alerts if a.get("name") == "bad"]
        assert len(check_alerts) == 1
        assert check_alerts[0]["severity"] == "warning"

    def test_overall_unhealthy_alert(self):
        mon = CollectorMonitor()
        mon.add_health_check("fail", lambda: False)
        alerts = mon.check_alerts()
        types = [a["type"] for a in alerts]
        assert "health" in types or "check" in types

    def test_stores_alerts(self):
        mon = CollectorMonitor()
        mon.add_health_check("x", lambda: False)
        mon.check_alerts()
        assert len(mon._alerts) >= 1


class TestCollectorMonitorReport:
    def test_format_report_not_empty(self):
        mon = CollectorMonitor()
        mon.add_health_check("chk", lambda: True)
        report = mon.format_report()
        assert "Collection Monitor Report" in report
        assert "Healthy: Yes" in report

    def test_format_report_unhealthy(self):
        mon = CollectorMonitor()
        mon.add_health_check("bad", lambda: False)
        report = mon.format_report()
        assert "NO" in report
        assert "FAIL" in report

    def test_format_report_with_components(self):
        sched = JobScheduler()
        sched.add_job(JobConfig(name="j"), make_collector())
        mon = CollectorMonitor(scheduler=sched)
        report = mon.format_report()
        assert "SCHEDULER" in report


# ===========================================================================
# CollectorExporter
# ===========================================================================

class TestCollectorExporterToMemory:
    def test_to_memory_empty_store(self):
        exp = CollectorExporter(MemoryStore())
        mem = exp.to_memory()
        assert mem.count() == 0

    def test_to_memory_copies_records(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        exp = CollectorExporter(store)
        mem = exp.to_memory()
        assert mem.count() == 2
        contents = [r.content for r in mem.read_all()]
        assert "a" in contents
        assert "b" in contents

    def test_to_memory_no_store(self):
        exp = CollectorExporter(None)
        mem = exp.to_memory()
        assert mem.count() == 0


class TestCollectorExporterToDicts:
    def test_to_dicts_empty(self):
        exp = CollectorExporter(MemoryStore())
        assert exp.to_dicts() == []

    def test_to_dicts_returns_list_of_dicts(self):
        store = MemoryStore()
        store.write(Record(content="x", metadata={"k": "v"}))
        exp = CollectorExporter(store)
        dicts = exp.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["content"] == "x"

    def test_to_dicts_no_store(self):
        exp = CollectorExporter(None)
        assert exp.to_dicts() == []


class TestCollectorExporterToJsonl:
    def test_to_jsonl_writes_file(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="line1"))
        store.write(Record(content="line2"))
        exp = CollectorExporter(store)
        path = str(tmp_path / "out.jsonl")
        count = exp.to_jsonl(path)
        assert count == 2
        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["content"] == "line1"

    def test_to_jsonl_no_store(self):
        exp = CollectorExporter(None)
        assert exp.to_jsonl("/tmp/nonexistent.jsonl") == 0

    def test_to_jsonl_empty_store(self, tmp_path):
        exp = CollectorExporter(MemoryStore())
        path = str(tmp_path / "empty.jsonl")
        count = exp.to_jsonl(path)
        assert count == 0


class TestCollectorExporterToJson:
    def test_to_json_writes_array(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        exp = CollectorExporter(store)
        path = str(tmp_path / "out.json")
        count = exp.to_json(path)
        assert count == 2
        data = json.loads(Path(path).read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_to_json_no_store(self):
        exp = CollectorExporter(None)
        assert exp.to_json("/tmp/nonexistent.json") == 0


class TestCollectorExporterToText:
    def test_to_text_writes_content(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="hello"))
        store.write(Record(content="world"))
        exp = CollectorExporter(store)
        path = str(tmp_path / "out.txt")
        count = exp.to_text(path)
        assert count == 2
        text = Path(path).read_text()
        assert "hello" in text
        assert "world" in text

    def test_to_text_no_store(self):
        exp = CollectorExporter(None)
        assert exp.to_text("/tmp/nonexistent.txt") == 0


class TestCollectorExporterSummary:
    def test_summary_empty_store(self):
        exp = CollectorExporter(MemoryStore())
        s = exp.summary()
        assert s["count"] == 0
        assert s["total_bytes"] == 0
        assert s["sources"] == {}

    def test_summary_no_store(self):
        exp = CollectorExporter(None)
        s = exp.summary()
        assert s["count"] == 0

    def test_summary_counts_bytes_and_sources(self):
        store = MemoryStore()
        store.write(Record(content="abc", metadata={"source": "src1"}))
        store.write(Record(content="de", metadata={"source": "src1"}))
        store.write(Record(content="f", metadata={"source": "src2"}))
        exp = CollectorExporter(store)
        s = exp.summary()
        assert s["count"] == 3
        assert s["total_bytes"] == 6  # 3+2+1
        assert s["sources"]["src1"] == 2
        assert s["sources"]["src2"] == 1

    def test_summary_unknown_source(self):
        store = MemoryStore()
        store.write(Record(content="x"))
        exp = CollectorExporter(store)
        s = exp.summary()
        assert s["sources"]["unknown"] == 1


class TestCollectorExporterSetStore:
    def test_set_store(self):
        exp = CollectorExporter()
        new_store = MemoryStore()
        result = exp.set_store(new_store)
        assert result is exp
        assert exp._store is new_store

    def test_set_store_overwrites(self):
        exp = CollectorExporter(MemoryStore())
        new_store = MemoryStore()
        exp.set_store(new_store)
        assert exp._store is new_store


# ===========================================================================
# Integration: Scheduler + Collector + Store
# ===========================================================================

class TestSchedulerIntegration:
    def test_full_cycle(self):
        store = MemoryStore()
        src = StubSource([Record(content="item1"), Record(content="item2")])
        col = Collector(src, store)
        sched = JobScheduler()
        cfg = JobConfig(name="cycle", interval=0.0, max_runs=1)
        sched.add_job(cfg, col)
        sched.start_job("cycle")
        time.sleep(0.3)
        assert store.count() == 2
        stats = sched.job_stats("cycle")
        assert stats["runs"] == 1
        assert stats["total_collected"] == 2

    def test_multiple_jobs_parallel(self):
        store_a = MemoryStore()
        store_b = MemoryStore()
        col_a = Collector(StubSource([Record(content="a")]), store_a)
        col_b = Collector(StubSource([Record(content="b")]), store_b)
        sched = JobScheduler()
        sched.add_job(JobConfig(name="job_a", interval=0.0, max_runs=1), col_a)
        sched.add_job(JobConfig(name="job_b", interval=0.0, max_runs=1), col_b)
        sched.start_all()
        time.sleep(0.5)
        assert store_a.count() == 1
        assert store_b.count() == 1
        assert sched.job_stats("job_a")["runs"] == 1
        assert sched.job_stats("job_b")["runs"] == 1

    def test_exporter_reads_from_store(self, tmp_path):
        store = MemoryStore()
        for i in range(5):
            store.write(Record(content=f"rec{i}", metadata={"source": "test"}))
        exp = CollectorExporter(store)
        path = str(tmp_path / "export.jsonl")
        count = exp.to_jsonl(path)
        assert count == 5
        reloaded = exp.to_memory()
        assert reloaded.count() == 5
