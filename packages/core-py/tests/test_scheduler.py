"""Tests for collections.scheduler — JobScheduler, CollectorMonitor, CollectorExporter."""

from __future__ import annotations

import json
import time
import tempfile
import os
from unittest.mock import MagicMock, patch

import pytest

from domains.collections.scheduler import JobConfig, JobScheduler, CollectorMonitor, CollectorExporter
from domains.collections.sources import Record
from domains.collections.stores import MemoryStore


# ── JobConfig ──────────────────────────────────────────────────────────────


class TestJobConfig:

    def test_defaults(self):
        cfg = JobConfig(name="test")
        assert cfg.name == "test"
        assert cfg.interval == 60.0
        assert cfg.enabled is True
        assert cfg.max_runs is None
        assert cfg.timeout is None
        assert cfg.on_complete is None
        assert cfg.on_error is None

    def test_custom(self):
        cfg = JobConfig(
            name="custom",
            interval=30.0,
            enabled=False,
            max_runs=5,
            timeout=10.0,
            on_complete=lambda n, c: None,
            on_error=lambda n, e: None,
        )
        assert cfg.interval == 30.0
        assert cfg.enabled is False
        assert cfg.max_runs == 5


# ── JobScheduler ───────────────────────────────────────────────────────────


class TestJobScheduler:

    def setup_method(self):
        self.scheduler = JobScheduler()

    def test_init(self):
        assert self.scheduler._jobs == {}
        assert self.scheduler._collectors == {}

    def test_add_job(self):
        collector = MagicMock()
        collector.collect.return_value = 0
        result = self.scheduler.add_job(JobConfig(name="j1"), collector)
        assert result is self.scheduler
        assert "j1" in self.scheduler.list_jobs()

    def test_add_job_chaining(self):
        c1, c2 = MagicMock(), MagicMock()
        c1.collect.return_value = 0
        c2.collect.return_value = 0
        self.scheduler.add_job(JobConfig(name="j1"), c1).add_job(JobConfig(name="j2"), c2)
        assert set(self.scheduler.list_jobs()) == {"j1", "j2"}

    def test_start_job(self):
        collector = MagicMock()
        collector.collect.return_value = 3
        self.scheduler.add_job(JobConfig(name="j1", interval=10), collector)
        assert self.scheduler.start_job("j1") is True
        time.sleep(0.1)
        self.scheduler.stop_job("j1")

    def test_start_job_not_found(self):
        assert self.scheduler.start_job("nope") is False

    def test_start_job_disabled(self):
        collector = MagicMock()
        self.scheduler.add_job(JobConfig(name="j1", enabled=False), collector)
        assert self.scheduler.start_job("j1") is False

    def test_start_job_already_running(self):
        collector = MagicMock()
        collector.collect.return_value = 0
        self.scheduler.add_job(JobConfig(name="j1", interval=1), collector)
        self.scheduler.start_job("j1")
        assert self.scheduler.start_job("j1") is False
        self.scheduler.stop_job("j1")

    def test_stop_job(self):
        collector = MagicMock()
        collector.collect.return_value = 0
        self.scheduler.add_job(JobConfig(name="j1", interval=1), collector)
        self.scheduler.start_job("j1")
        time.sleep(0.1)
        assert self.scheduler.stop_job("j1") is True

    def test_stop_job_not_found(self):
        assert self.scheduler.stop_job("nope") is False

    def test_remove_job(self):
        collector = MagicMock()
        self.scheduler.add_job(JobConfig(name="j1"), collector)
        assert self.scheduler.remove_job("j1") is True
        assert "j1" not in self.scheduler.list_jobs()

    def test_remove_job_not_found(self):
        assert self.scheduler.remove_job("nope") is False

    def test_max_runs_stops(self):
        collector = MagicMock()
        collector.collect.return_value = 1
        self.scheduler.add_job(JobConfig(name="j1", interval=0.05, max_runs=2), collector)
        self.scheduler.start_job("j1")
        time.sleep(0.5)
        stats = self.scheduler.job_stats("j1")
        assert stats["runs"] == 2
        assert stats["status"] == "stopped"

    def test_on_complete_callback(self):
        cb = MagicMock()
        collector = MagicMock()
        collector.collect.return_value = 5
        self.scheduler.add_job(JobConfig(name="j1", interval=0.05, max_runs=1, on_complete=cb), collector)
        self.scheduler.start_job("j1")
        time.sleep(0.3)
        cb.assert_called_once_with("j1", 5)

    def test_on_error_callback(self):
        cb = MagicMock()
        collector = MagicMock()
        collector.collect.side_effect = RuntimeError("boom")
        self.scheduler.add_job(JobConfig(name="j1", interval=0.05, max_runs=1, on_error=cb), collector)
        self.scheduler.start_job("j1")
        time.sleep(0.3)
        cb.assert_called_once()
        assert isinstance(cb.call_args[0][1], RuntimeError)

    def test_start_all(self):
        c1, c2 = MagicMock(), MagicMock()
        c1.collect.return_value = 0
        c2.collect.return_value = 0
        self.scheduler.add_job(JobConfig(name="j1", interval=1), c1)
        self.scheduler.add_job(JobConfig(name="j2", interval=1), c2)
        count = self.scheduler.start_all()
        assert count == 2
        self.scheduler.stop_all()

    def test_stop_all(self):
        c1, c2 = MagicMock(), MagicMock()
        c1.collect.return_value = 0
        c2.collect.return_value = 0
        self.scheduler.add_job(JobConfig(name="j1", interval=1), c1)
        self.scheduler.add_job(JobConfig(name="j2", interval=1), c2)
        self.scheduler.start_all()
        time.sleep(0.1)
        count = self.scheduler.stop_all()
        assert count == 2

    def test_stats(self):
        collector = MagicMock()
        self.scheduler.add_job(JobConfig(name="j1"), collector)
        stats = self.scheduler.stats()
        assert "j1" in stats
        assert stats["j1"]["runs"] == 0

    def test_job_stats(self):
        collector = MagicMock()
        self.scheduler.add_job(JobConfig(name="j1"), collector)
        assert self.scheduler.job_stats("j1")["status"] == "idle"
        assert self.scheduler.job_stats("nope") is None

    def test_get_collector(self):
        collector = MagicMock()
        self.scheduler.add_job(JobConfig(name="j1"), collector)
        assert self.scheduler.get_collector("j1") is collector
        assert self.scheduler.get_collector("nope") is None

    def test_is_running(self):
        collector = MagicMock()
        collector.collect.return_value = 0
        self.scheduler.add_job(JobConfig(name="j1", interval=1), collector)
        assert self.scheduler.is_running("j1") is False
        self.scheduler.start_job("j1")
        assert self.scheduler.is_running("j1") is True
        self.scheduler.stop_job("j1")


# ── CollectorMonitor ───────────────────────────────────────────────────────


class TestCollectorMonitor:

    def test_init(self):
        monitor = CollectorMonitor()
        assert monitor._runner is None
        assert monitor._scheduler is None

    def test_add_health_check(self):
        monitor = CollectorMonitor()
        result = monitor.add_health_check("db", lambda: True)
        assert result is monitor
        assert "db" in monitor._health_checks

    def test_check_health(self):
        monitor = CollectorMonitor()
        monitor.add_health_check("ok", lambda: True)
        monitor.add_health_check("fail", lambda: False)
        results = monitor.check_health()
        assert results["ok"] is True
        assert results["fail"] is False

    def test_check_health_exception(self):
        monitor = CollectorMonitor()
        monitor.add_health_check("err", lambda: 1 / 0)
        results = monitor.check_health()
        assert results["err"] is False

    def test_get_overview_no_components(self):
        monitor = CollectorMonitor()
        overview = monitor.get_overview()
        assert overview["healthy"] is True
        assert overview["components"] == {}

    def test_get_overview_with_runner(self):
        runner = MagicMock()
        runner.stats.return_value = {"c1": {"total_collected": 10, "errors": 2}}
        monitor = CollectorMonitor(runner=runner)
        overview = monitor.get_overview()
        assert "runner" in overview["components"]
        assert overview["components"]["runner"]["total_collected"] == 10

    def test_get_overview_with_scheduler(self):
        scheduler = MagicMock()
        scheduler.stats.return_value = {"j1": {"status": "running", "runs": 5}}
        monitor = CollectorMonitor(scheduler=scheduler)
        overview = monitor.get_overview()
        assert "scheduler" in overview["components"]
        assert overview["components"]["scheduler"]["running"] == 1

    def test_check_alerts_healthy(self):
        monitor = CollectorMonitor()
        alerts = monitor.check_alerts()
        assert alerts == []

    def test_check_alerts_unhealthy(self):
        monitor = CollectorMonitor()
        monitor.add_health_check("down", lambda: False)
        alerts = monitor.check_alerts()
        assert len(alerts) == 2
        assert alerts[0]["type"] == "health"
        assert alerts[1]["name"] == "down"

    def test_format_report(self):
        monitor = CollectorMonitor()
        monitor.add_health_check("db", lambda: True)
        report = monitor.format_report()
        assert "Collection Monitor Report" in report
        assert "Healthy: Yes" in report
        assert "db: OK" in report


# ── CollectorExporter ──────────────────────────────────────────────────────


class TestCollectorExporter:

    def test_init(self):
        exporter = CollectorExporter()
        assert exporter._store is None

    def test_set_store(self):
        store = MemoryStore()
        exporter = CollectorExporter()
        result = exporter.set_store(store)
        assert result is exporter
        assert exporter._store is store

    def test_to_jsonl_no_store(self):
        exporter = CollectorExporter()
        assert exporter.to_jsonl("/tmp/nonexistent.jsonl") == 0

    def test_to_jsonl_with_records(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="hello", metadata={"source": "test"}))
        store.write(Record(content="world", metadata={"source": "test"}))
        exporter = CollectorExporter(store=store)
        path = str(tmp_path / "out.jsonl")
        count = exporter.to_jsonl(path)
        assert count == 2
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "hello"

    def test_to_json_no_store(self):
        exporter = CollectorExporter()
        assert exporter.to_json("/tmp/nonexistent.json") == 0

    def test_to_json_with_records(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="a", metadata={"k": "v"}))
        exporter = CollectorExporter(store=store)
        path = str(tmp_path / "out.json")
        count = exporter.to_json(path)
        assert count == 1
        with open(path) as f:
            data = json.load(f)
        assert data[0]["content"] == "a"

    def test_to_text_no_store(self):
        exporter = CollectorExporter()
        assert exporter.to_text("/tmp/nonexistent.txt") == 0

    def test_to_text_with_records(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="line1", metadata={}))
        store.write(Record(content="line2", metadata={}))
        exporter = CollectorExporter(store=store)
        path = str(tmp_path / "out.txt")
        count = exporter.to_text(path)
        assert count == 2
        with open(path) as f:
            text = f.read()
        assert "line1\nline2\n" in text

    def test_to_memory_no_store(self):
        exporter = CollectorExporter()
        result = exporter.to_memory()
        assert isinstance(result, MemoryStore)
        assert result.count() == 0

    def test_to_memory_with_records(self):
        store = MemoryStore()
        store.write(Record(content="x", metadata={}))
        exporter = CollectorExporter(store=store)
        mem = exporter.to_memory()
        assert isinstance(mem, MemoryStore)
        assert mem.count() == 1

    def test_to_dicts_no_store(self):
        exporter = CollectorExporter()
        assert exporter.to_dicts() == []

    def test_to_dicts_with_records(self):
        store = MemoryStore()
        store.write(Record(content="d", metadata={"a": 1}))
        exporter = CollectorExporter(store=store)
        dicts = exporter.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["content"] == "d"

    def test_summary_no_store(self):
        exporter = CollectorExporter()
        s = exporter.summary()
        assert s["count"] == 0
        assert s["total_bytes"] == 0

    def test_summary_with_records(self):
        store = MemoryStore()
        store.write(Record(content="hello", metadata={"source": "web"}))
        store.write(Record(content="world", metadata={"source": "db"}))
        store.write(Record(content="!", metadata={"source": "web"}))
        exporter = CollectorExporter(store=store)
        s = exporter.summary()
        assert s["count"] == 3
        assert s["total_bytes"] == 11
        assert s["sources"]["web"] == 2
        assert s["sources"]["db"] == 1
