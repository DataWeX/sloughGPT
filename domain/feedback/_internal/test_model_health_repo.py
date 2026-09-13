"""Tests for ModelHealthRepository."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from domains.feedback.model_health_repo import (
    HealthSnapshot,
    ModelHealthRepository,
)


class TestHealthSnapshot:
    def test_to_dict(self):
        snapshot = HealthSnapshot(
            timestamp=1000.0,
            perplexity=42.5,
            loss=3.75,
            num_sentences=15,
            model_name="test_model",
        )
        data = snapshot.to_dict()
        assert data["timestamp"] == 1000.0
        assert data["perplexity"] == 42.5
        assert data["model_name"] == "test_model"

    def test_from_dict(self):
        data = {
            "timestamp": 1000.0,
            "perplexity": 42.5,
            "loss": 3.75,
            "num_sentences": 15,
            "model_name": "test_model",
        }
        snapshot = HealthSnapshot.from_dict(data)
        assert snapshot.perplexity == 42.5
        assert snapshot.model_name == "test_model"


class TestModelHealthRepository:
    def test_add_and_get_latest(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        repo.add_snapshot(HealthSnapshot(
            timestamp=now,
            perplexity=42.0,
            loss=3.8,
            num_sentences=15,
        ))
        latest = repo.get_latest()
        assert latest is not None
        assert latest.perplexity == 42.0

    def test_list_snapshots(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        for i in range(5):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now + i,
                perplexity=40.0 + i,
                loss=3.8 - i * 0.1,
                num_sentences=15,
            ))
        all_snapshots = repo.list_snapshots()
        assert len(all_snapshots) == 5
        # Most recent first
        assert all_snapshots[0].perplexity > all_snapshots[-1].perplexity

    def test_list_with_limit(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        for i in range(10):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now + i,
                perplexity=40.0 + i,
                loss=3.8,
                num_sentences=15,
            ))
        recent = repo.list_snapshots(limit=3)
        assert len(recent) == 3

    def test_get_trend(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        # Add old snapshot (25 hours ago)
        repo.add_snapshot(HealthSnapshot(
            timestamp=now - 25 * 3600,
            perplexity=40.0,
            loss=3.8,
            num_sentences=15,
        ))
        # Add recent snapshots
        for i in range(3):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now - i * 3600,
                perplexity=42.0 + i,
                loss=3.6,
                num_sentences=15,
            ))
        trend = repo.get_trend(hours=24)
        assert len(trend) == 3

    def test_detect_drift(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        # Add baseline snapshots with stable perplexity
        for i in range(5):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now - (5 - i) * 3600,
                perplexity=40.0,
                loss=3.8,
                num_sentences=15,
            ))
        # Add latest with high perplexity (drift)
        repo.add_snapshot(HealthSnapshot(
            timestamp=now,
            perplexity=60.0,  # 50% increase
            loss=4.5,
            num_sentences=15,
        ))
        drift = repo.detect_drift(threshold=0.15)
        assert drift is not None
        assert drift["drift_detected"] is True
        assert drift["latest_ppl"] == 60.0

    def test_detect_no_drift(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        for i in range(5):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now - (5 - i) * 3600,
                perplexity=40.0 + i * 0.1,  # Very small variation
                loss=3.8,
                num_sentences=15,
            ))
        drift = repo.detect_drift(threshold=0.15)
        assert drift is None

    def test_get_stats(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        for i in range(3):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now + i,
                perplexity=40.0 + i,
                loss=3.8 - i * 0.1,
                num_sentences=15,
            ))
        stats = repo.get_stats()
        assert stats["count"] == 3
        assert stats["avg_perplexity"] == pytest.approx(41.0, rel=0.01)
        assert stats["min_perplexity"] == 40.0
        assert stats["max_perplexity"] == 42.0

    def test_clear(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        now = time.time()
        for i in range(5):
            repo.add_snapshot(HealthSnapshot(
                timestamp=now + i,
                perplexity=40.0,
                loss=3.8,
                num_sentences=15,
            ))
        count = repo.clear()
        assert count == 5
        assert repo.list_snapshots() == []

    def test_persistence(self, tmp_path: Path):
        repo1 = ModelHealthRepository(tmp_path / "health")
        repo1.add_snapshot(HealthSnapshot(
            timestamp=time.time(),
            perplexity=42.0,
            loss=3.8,
            num_sentences=15,
        ))
        repo2 = ModelHealthRepository(tmp_path / "health")
        latest = repo2.get_latest()
        assert latest is not None
        assert latest.perplexity == 42.0

    def test_empty_repo(self, tmp_path: Path):
        repo = ModelHealthRepository(tmp_path / "health")
        assert repo.get_latest() is None
        assert repo.list_snapshots() == []
        assert repo.get_stats() == {"count": 0}
        assert repo.detect_drift() is None

    def test_legacy_migration(self, tmp_path: Path):
        # Create legacy file
        legacy_path = tmp_path / "health" / "model_health.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_data = [
            {"timestamp": 1000.0, "perplexity": 40.0, "loss": 3.8, "num_sentences": 15},
            {"timestamp": 2000.0, "perplexity": 42.0, "loss": 3.6, "num_sentences": 15},
        ]
        legacy_path.write_text(json.dumps(legacy_data))

        # Initialize repo - should migrate
        repo = ModelHealthRepository(tmp_path / "health")
        snapshots = repo.list_snapshots()
        assert len(snapshots) == 2

        # Legacy file should be backed up
        assert not legacy_path.exists()
        assert (legacy_path.with_suffix(".json.bak")).exists()
