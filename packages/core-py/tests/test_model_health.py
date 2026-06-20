"""Tests for model health monitor — benchmark, trend, drift detection."""

import json
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from domains.feedback.model_health import (
    HealthSnapshot, ModelHealthMonitor, BENCHMARK_CORPUS, get_health_monitor,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def monitor(tmp_path):
    return ModelHealthMonitor(db_path=str(tmp_path / "health.json"))


class MockLSTM:
    def init_hidden(self):
        return (np.zeros((1, 1, 64)), np.zeros((1, 1, 64)))

    def forward(self, x, h):
        vocab_size = 256
        logits = np.random.randn(x.data.shape[0], x.data.shape[1], vocab_size)
        return MagicMock(data=logits), h


class MockModel:
    layers = [MockLSTM()]


class MockTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))


@pytest.fixture
def model_with_mock():
    return MockModel(), MockTokenizer()


def _make_monitor_with_mock(model, tokenizer, tmp_path):
    """Create a ModelHealthMonitor with mock model/tokenizer for benchmark tests."""
    m = ModelHealthMonitor(db_path=str(tmp_path / "bm.json"))
    m.set_model(model, tokenizer)
    return m


# ── BENCHMARK_CORPUS ──────────────────────────────────────────────────────

class TestBenchmarkCorpus:

    def test_length(self):
        assert len(BENCHMARK_CORPUS) == 15

    def test_all_strings(self):
        for s in BENCHMARK_CORPUS:
            assert isinstance(s, str)
            assert len(s) > 0


# ── HealthSnapshot ────────────────────────────────────────────────────────

class TestHealthSnapshot:

    def test_fields(self):
        snap = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=15)
        assert snap.timestamp == 1.0
        assert snap.perplexity == 5.0
        assert snap.loss == 1.5
        assert snap.num_sentences == 15


# ── ModelHealthMonitor ────────────────────────────────────────────────────

class TestModelHealthMonitor:

    def test_init_creates_dir(self, tmp_path):
        subdir = tmp_path / "sub" / "health.json"
        m = ModelHealthMonitor(db_path=str(subdir))
        assert subdir.parent.exists()

    def test_init_empty_history(self, monitor):
        assert monitor._history == []
        assert monitor._model is None
        assert monitor._tokenizer is None

    def test_set_model(self, monitor, model_with_mock):
        model, tok = model_with_mock
        monitor.set_model(model, tok)
        assert monitor._model is model
        assert monitor._tokenizer is tok

    def test_run_benchmark_no_model(self, monitor):
        result = monitor.run_benchmark()
        assert result is None

    def test_run_benchmark_with_mock(self, model_with_mock, tmp_path):
        model, tok = model_with_mock
        m = _make_monitor_with_mock(model, tok, tmp_path)
        with patch('domains.training.slonet.SloLSTM', MockLSTM):
            snap = m.run_benchmark()
        assert snap is not None
        assert isinstance(snap, HealthSnapshot)
        assert snap.perplexity > 0
        assert snap.loss > 0
        assert snap.num_sentences == 15

    def test_run_benchmark_saves_history(self, model_with_mock, tmp_path):
        model, tok = model_with_mock
        m = _make_monitor_with_mock(model, tok, tmp_path)
        with patch('domains.training.slonet.SloLSTM', MockLSTM):
            m.run_benchmark()
        assert len(m._history) == 1

    def test_run_benchmark_multiple(self, model_with_mock, tmp_path):
        model, tok = model_with_mock
        m = _make_monitor_with_mock(model, tok, tmp_path)
        with patch('domains.training.slonet.SloLSTM', MockLSTM):
            for _ in range(3):
                m.run_benchmark()
        assert len(m._history) == 3

    def test_get_trend_empty(self, monitor):
        trend = monitor.get_trend()
        assert trend["available"] is False
        assert "message" in trend

    def test_get_trend_with_history(self, monitor):
        monitor._history = [
            {"perplexity": 10.0, "loss": 2.0, "timestamp": 100.0, "num_sentences": 15},
            {"perplexity": 12.0, "loss": 2.3, "timestamp": 200.0, "num_sentences": 15},
            {"perplexity": 8.0, "loss": 1.8, "timestamp": 300.0, "num_sentences": 15},
        ]
        trend = monitor.get_trend()
        assert trend["available"] is True
        assert trend["current"] == 8.0
        assert trend["best"] == 8.0
        assert trend["worst"] == 12.0
        assert trend["average"] == pytest.approx(10.0)
        assert trend["count"] == 3
        assert len(trend["points"]) == 3

    def test_get_trend_window(self, monitor):
        monitor._history = [
            {"perplexity": float(i), "loss": 2.0, "timestamp": float(i * 100), "num_sentences": 15}
            for i in range(1, 6)
        ]
        trend = monitor.get_trend(window=2)
        assert trend["recent_count"] == 2
        assert trend["count"] == 5
        assert trend["points"][0]["ppl"] == 4.0
        assert trend["points"][1]["ppl"] == 5.0

    def test_detect_drift_insufficient_data(self, monitor):
        result = monitor.detect_drift()
        assert result["drifted"] is False
        assert "Not enough data" in result["message"]

    def test_detect_drift_stable(self, monitor):
        base_ppl = 10.0
        monitor._history = [
            {"perplexity": base_ppl, "loss": 2.0, "timestamp": time.time() - i * 60, "num_sentences": 15}
            for i in range(6, 0, -1)
        ]
        result = monitor.detect_drift()
        assert result["drifted"] is False
        assert "stable" in result["message"]

    def test_detect_drift_spike(self, monitor):
        base_ppl = 10.0
        spike_ppl = 20.0
        monitor._history = [
            {"perplexity": base_ppl, "loss": 2.0, "timestamp": time.time() - i * 60, "num_sentences": 15}
            for i in range(6, 1, -1)
        ]
        monitor._history.append({
            "perplexity": spike_ppl, "loss": 3.0,
            "timestamp": time.time(), "num_sentences": 15,
        })
        result = monitor.detect_drift(threshold_pct=10.0)
        assert result["drifted"] is True
        assert "drift detected" in result["message"]

    def test_detect_drift_change_pct(self, monitor):
        monitor._history = [
            {"perplexity": 10.0, "loss": 2.0, "timestamp": time.time() - i * 60, "num_sentences": 15}
            for i in range(6, 1, -1)
        ]
        monitor._history.append({
            "perplexity": 12.0, "loss": 2.3,
            "timestamp": time.time(), "num_sentences": 15,
        })
        result = monitor.detect_drift(threshold_pct=10.0)
        assert result["drifted"] is True
        assert result["change_pct"] == pytest.approx(20.0, abs=1.0)

    def test_detect_drift_below_threshold(self, monitor):
        monitor._history = [
            {"perplexity": 10.0, "loss": 2.0, "timestamp": time.time() - i * 60, "num_sentences": 15}
            for i in range(6, 1, -1)
        ]
        monitor._history.append({
            "perplexity": 10.5, "loss": 2.05,
            "timestamp": time.time(), "num_sentences": 15,
        })
        result = monitor.detect_drift(threshold_pct=10.0)
        assert result["drifted"] is False
        assert "stable" in result["message"]

    def test_get_stats_with_history(self, monitor):
        monitor._history = [
            {"perplexity": 10.0, "loss": 2.0, "timestamp": 100.0, "num_sentences": 15},
        ]
        stats = monitor.get_stats()
        assert "trend" in stats
        assert "drift" in stats
        assert "last_benchmark" in stats
        assert stats["last_benchmark"] is not None
        assert stats["trend"]["available"] is True

    def test_get_stats_no_history(self, monitor):
        stats = monitor.get_stats()
        assert stats["trend"]["available"] is False
        assert stats["last_benchmark"] is None

    def test_save_load_history(self, tmp_path):
        path = tmp_path / "persist.json"
        m1 = ModelHealthMonitor(db_path=str(path))
        m1._history = [
            {"perplexity": 5.0, "loss": 1.5, "timestamp": 100.0, "num_sentences": 15},
            {"perplexity": 6.0, "loss": 1.8, "timestamp": 200.0, "num_sentences": 15},
        ]
        m1._save_history()
        m2 = ModelHealthMonitor(db_path=str(path))
        assert len(m2._history) == 2
        assert m2._history[0]["perplexity"] == 5.0
        assert m2._history[1]["perplexity"] == 6.0

    def test_load_history_corrupted_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")
        m = ModelHealthMonitor(db_path=str(path))
        assert m._history == []

    def test_save_history_trims_to_200(self, monitor):
        monitor._history = [{"perplexity": i, "loss": i, "timestamp": i, "num_sentences": 15} for i in range(250)]
        monitor._save_history()
        reloaded = ModelHealthMonitor(db_path=str(monitor.db_path))
        assert len(reloaded._history) == 200


# ── Global singleton ──────────────────────────────────────────────────────

class TestGetHealthMonitor:

    def test_returns_singleton(self):
        import domains.feedback.model_health as mod
        original = mod._health_monitor
        mod._health_monitor = None
        try:
            h1 = get_health_monitor()
            h2 = get_health_monitor()
            assert h1 is h2
        finally:
            mod._health_monitor = original
