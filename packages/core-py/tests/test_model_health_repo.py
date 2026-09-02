"""Tests for domains.feedback.model_health_repo — ModelHealthRepository, HealthSnapshot."""

from __future__ import annotations

import time
import pytest
from pathlib import Path

from domains.feedback.model_health_repo import (
    HealthSnapshot,
    ModelHealthRepository,
)


def _make_snapshot(ts=None, ppl=10.0, loss=1.5, sentences=100, model="test"):
    return HealthSnapshot(
        timestamp=ts or time.time(),
        perplexity=ppl,
        loss=loss,
        num_sentences=sentences,
        model_name=model,
    )


@pytest.fixture
def repo(tmp_path):
    return ModelHealthRepository(tmp_path / "health")


# ── HealthSnapshot ────────────────────────────────────────────────────────────

class TestHealthSnapshot:
    def test_to_dict(self):
        s = _make_snapshot(ts=1000.0, ppl=5.0, loss=0.5)
        d = s.to_dict()
        assert d["timestamp"] == 1000.0
        assert d["perplexity"] == 5.0
        assert d["loss"] == 0.5

    def test_from_dict(self):
        d = {"timestamp": 1000.0, "perplexity": 5.0, "loss": 0.5, "num_sentences": 50}
        s = HealthSnapshot.from_dict(d)
        assert s.timestamp == 1000.0
        assert s.perplexity == 5.0

    def test_from_dict_extra_fields(self):
        d = {"timestamp": 1, "perplexity": 2, "loss": 3, "num_sentences": 4, "extra": "ignored"}
        s = HealthSnapshot.from_dict(d)
        assert s.timestamp == 1

    def test_defaults(self):
        s = HealthSnapshot(timestamp=0, perplexity=1.0, loss=1.0, num_sentences=0)
        assert s.model_name == ""
        assert s.quality_score == 0.0


# ── ModelHealthRepository ─────────────────────────────────────────────────────

class TestModelHealthRepository:
    def test_empty_repo(self, repo):
        assert repo.list_snapshots() == []
        assert repo.get_latest() is None
        assert repo.get_stats() == {"count": 0}

    def test_add_and_list(self, repo):
        s1 = _make_snapshot(ts=1000, ppl=10.0)
        s2 = _make_snapshot(ts=2000, ppl=8.0)
        repo.add_snapshot(s1)
        repo.add_snapshot(s2)
        snapshots = repo.list_snapshots()
        assert len(snapshots) == 2

    def test_get_latest(self, repo):
        repo.add_snapshot(_make_snapshot(ts=1000, ppl=10.0))
        repo.add_snapshot(_make_snapshot(ts=2000, ppl=8.0))
        latest = repo.get_latest()
        assert latest.perplexity == 8.0

    def test_list_with_limit(self, repo):
        for i in range(5):
            repo.add_snapshot(_make_snapshot(ts=i * 1000, ppl=float(i)))
        limited = repo.list_snapshots(limit=2)
        assert len(limited) == 2
        # Most recent first
        assert limited[0].timestamp > limited[1].timestamp

    def test_get_trend(self, repo):
        now = time.time()
        repo.add_snapshot(_make_snapshot(ts=now - 7200, ppl=10.0))  # 2 hours ago
        repo.add_snapshot(_make_snapshot(ts=now - 1800, ppl=9.0))   # 30 min ago
        repo.add_snapshot(_make_snapshot(ts=now - 3600 * 48, ppl=8.0))  # 2 days ago

        trend = repo.get_trend(hours=24)
        assert len(trend) == 2

    def test_detect_no_drift(self, repo):
        now = time.time()
        for i in range(5):
            repo.add_snapshot(_make_snapshot(ts=now - i * 100, ppl=10.0))
        drift = repo.detect_drift()
        assert drift is None

    def test_detect_drift(self, repo):
        now = time.time()
        for i in range(5):
            repo.add_snapshot(_make_snapshot(ts=now - i * 100, ppl=10.0))
        # Add a snapshot with significantly higher perplexity
        repo.add_snapshot(_make_snapshot(ts=now + 1, ppl=20.0))
        drift = repo.detect_drift(threshold=0.15)
        assert drift is not None
        assert drift["drift_detected"] is True
        assert drift["latest_ppl"] == 20.0

    def test_detect_drift_insufficient_data(self, repo):
        repo.add_snapshot(_make_snapshot(ppl=10.0))
        drift = repo.detect_drift()
        assert drift is None

    def test_get_stats(self, repo):
        repo.add_snapshot(_make_snapshot(ts=1000, ppl=10.0, loss=1.0))
        repo.add_snapshot(_make_snapshot(ts=2000, ppl=8.0, loss=0.5))
        stats = repo.get_stats()
        assert stats["count"] == 2
        assert stats["avg_perplexity"] == 9.0
        assert stats["min_perplexity"] == 8.0
        assert stats["max_perplexity"] == 10.0
        assert stats["avg_loss"] == 0.75

    def test_clear(self, repo):
        repo.add_snapshot(_make_snapshot(ts=1000))
        repo.add_snapshot(_make_snapshot(ts=2000))
        count = repo.clear()
        assert count == 2
        assert repo.list_snapshots() == []

    def test_persistence(self, tmp_path):
        s1 = ModelHealthRepository(tmp_path / "h")
        s1.add_snapshot(_make_snapshot(ts=1000, ppl=5.0))

        s2 = ModelHealthRepository(tmp_path / "h")
        snapshots = s2.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].perplexity == 5.0


# ── Legacy Migration ──────────────────────────────────────────────────────────

class TestLegacyMigration:
    def test_migrate_legacy_file(self, tmp_path):
        import json
        legacy_data = [
            {"timestamp": 1000, "perplexity": 10.0, "loss": 1.0, "num_sentences": 100},
            {"timestamp": 2000, "perplexity": 8.0, "loss": 0.5, "num_sentences": 200},
        ]
        legacy_path = tmp_path / "h" / "model_health.json"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps(legacy_data))

        repo = ModelHealthRepository(tmp_path / "h")
        snapshots = repo.list_snapshots()
        assert len(snapshots) == 2
        # Legacy file should be renamed to .bak
        assert not legacy_path.exists()
        assert (legacy_path.parent / "model_health.json.bak").exists()

    def test_no_legacy_file(self, repo):
        # Just verify it works without legacy file
        assert repo.list_snapshots() == []
