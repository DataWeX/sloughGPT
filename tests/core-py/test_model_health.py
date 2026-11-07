"""Meaningful tests for ModelHealthMonitor — get_trend, detect_drift, get_stats, history persistence."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from domain.feedback._internal.model_health import HealthSnapshot, ModelHealthMonitor


class TestModelHealthMonitor:
    def test_init_empty(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        assert len(monitor._history) == 0

    def test_set_model(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        model = MagicMock()
        tokenizer = MagicMock()
        monitor.set_model(model, tokenizer)
        assert monitor._model is model
        assert monitor._tokenizer is tokenizer

    def test_run_benchmark_no_model(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        assert monitor.run_benchmark() is None

    def test_get_trend_empty(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        trend = monitor.get_trend()
        assert trend["available"] is False
        assert "No benchmarks" in trend["message"]

    def test_get_trend_with_history(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monitor._history = [
            {"timestamp": 1.0, "perplexity": 10.0, "loss": 2.3, "num_sentences": 15},
            {"timestamp": 2.0, "perplexity": 8.0, "loss": 2.1, "num_sentences": 15},
            {"timestamp": 3.0, "perplexity": 12.0, "loss": 2.5, "num_sentences": 15},
        ]
        trend = monitor.get_trend()
        assert trend["available"] is True
        assert trend["current"] == 12.0
        assert trend["best"] == 8.0
        assert trend["worst"] == 12.0
        assert trend["count"] == 3

    def test_detect_drift_not_enough_data(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        result = monitor.detect_drift()
        assert result["drifted"] is False
        assert "Not enough" in result["message"]

    def test_detect_drift_stable(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monitor._history = [
            {"timestamp": i, "perplexity": 10.0, "loss": 2.3, "num_sentences": 15} for i in range(6)
        ]
        result = monitor.detect_drift()
        assert result["drifted"] is False
        assert result["change_pct"] == 0.0

    def test_detect_drift_significant(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monitor._history = [
            {"timestamp": i, "perplexity": 10.0, "loss": 2.3, "num_sentences": 15} for i in range(5)
        ]
        monitor._history.append(
            {"timestamp": 5.0, "perplexity": 20.0, "loss": 3.0, "num_sentences": 15}
        )
        result = monitor.detect_drift()
        assert result["drifted"] is True
        assert result["change_pct"] > 10.0

    def test_detect_drift_custom_threshold(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monitor._history = [
            {"timestamp": i, "perplexity": 10.0, "loss": 2.3, "num_sentences": 15} for i in range(6)
        ]
        monitor._history.append(
            {"timestamp": 6.0, "perplexity": 11.0, "loss": 2.4, "num_sentences": 15}
        )
        result = monitor.detect_drift(threshold_pct=5.0)
        assert result["drifted"] is True

    def test_get_stats(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monitor._history = [
            {"timestamp": 1.0, "perplexity": 10.0, "loss": 2.3, "num_sentences": 15},
            {"timestamp": 2.0, "perplexity": 8.0, "loss": 2.1, "num_sentences": 15},
            {"timestamp": 3.0, "perplexity": 12.0, "loss": 2.5, "num_sentences": 15},
        ]
        stats = monitor.get_stats()
        assert "trend" in stats
        assert "drift" in stats
        assert stats["last_benchmark"]["perplexity"] == 12.0

    def test_get_stats_empty(self, tmp_path):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        stats = monitor.get_stats()
        assert stats["last_benchmark"] is None

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "health.json")
        monitor = ModelHealthMonitor(db_path=path)
        monitor._history = [
            {"timestamp": 1.0, "perplexity": 10.0, "loss": 2.3, "num_sentences": 15}
        ]
        monitor._save_history()
        # Load in new instance
        monitor2 = ModelHealthMonitor(db_path=path)
        assert len(monitor2._history) == 1
        assert monitor2._history[0]["perplexity"] == 10.0

    def test_persistence_truncates_to_200(self, tmp_path):
        path = str(tmp_path / "health.json")
        monitor = ModelHealthMonitor(db_path=path)
        # Seed via the real incremental save seam — one entry per benchmark,
        # exactly as run_benchmark() does. _save_history persists only the
        # latest entry (older ones are already in MogDB); the 200-entry cap
        # is enforced on LOAD (_load_history reads limit=200, newest first),
        # so a reload after 250 benchmarks yields the newest 200.
        for i in range(250):
            monitor._history = [
                {"timestamp": i, "perplexity": float(i), "loss": 2.0, "num_sentences": 15}
            ]
            monitor._save_history()
        monitor2 = ModelHealthMonitor(db_path=path)
        assert len(monitor2._history) == 200
        assert monitor2._history[-1]["timestamp"] == 249  # newest kept

    def test_corrupted_history_file(self, tmp_path):
        path = str(tmp_path / "health.json")
        Path(path).write_text("not json")
        monitor = ModelHealthMonitor(db_path=path)
        assert len(monitor._history) == 0

    def test_health_snapshot(self):
        snap = HealthSnapshot(timestamp=1.0, perplexity=10.0, loss=2.3, num_sentences=15)
        assert snap.perplexity == 10.0
        assert snap.num_sentences == 15


class TestStartAutoMonitoringProcess:
    """process-kind migration: the monitor loop runs as a pugqeep Process.

    See docs/plans/2026-09-20-execution-consolidation.md (stage 3A).
    """

    def test_returns_process_on_shared_engine(self, tmp_path, monkeypatch):
        from domain.infrastructure._internal.pugqeep.engine import Process, ProcessStatus
        from domain.infrastructure._internal.pugqeep.singleton import get_engine

        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monkeypatch.setattr(monitor, "run_benchmark", lambda: None)
        proc = monitor.start_auto_monitoring(interval_seconds=3600)
        try:
            assert isinstance(proc, Process)
            assert proc.name == "health-monitor"
            assert proc in get_engine().list_processes()
            deadline = time.time() + 5.0
            while proc.status != ProcessStatus.RUNNING and time.time() < deadline:
                time.sleep(0.01)
            assert proc.status == ProcessStatus.RUNNING
        finally:
            proc.cancel()

    def test_cancel_releases_loop(self, tmp_path, monkeypatch):
        monitor = ModelHealthMonitor(db_path=str(tmp_path / "health.json"))
        monkeypatch.setattr(monitor, "run_benchmark", lambda: None)
        proc = monitor.start_auto_monitoring(interval_seconds=3600)
        exited = threading.Event()
        # Wait until a worker actually holds the loop: a proc cancelled
        # before dispatch is never executed (terminal state is honored),
        # so on_complete only exists for the release path this test pins.
        from domain.infrastructure._internal.pugqeep.engine import ProcessStatus

        deadline = time.time() + 5.0
        while proc.status != ProcessStatus.RUNNING and time.time() < deadline:
            time.sleep(0.01)
        assert proc.status == ProcessStatus.RUNNING
        # Tree._execute fires on_complete when the loop fn returns — i.e.
        # the cancel hook released the worker holding the forever-loop.
        proc.on_complete(lambda _p: exited.set())
        proc.cancel()
        assert exited.wait(5.0), "monitor loop did not exit after cancel"

    def test_get_engine_is_singleton_with_default_tree(self):
        from domain.infrastructure._internal.pugqeep.singleton import get_engine

        e1 = get_engine()
        assert get_engine() is e1
        assert "default" in e1.list_trees()

    def test_get_engine_recreates_stopped_engine(self):
        from domain.infrastructure._internal.pugqeep.singleton import get_engine

        e1 = get_engine()
        e1.stop()
        e2 = get_engine()
        assert e2 is not e1
        assert e2._running
