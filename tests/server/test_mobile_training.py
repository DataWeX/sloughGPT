"""Tests for mobile training data CRUD endpoints."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.mobile import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def mock_store():
    """Mock MobileTrainingStore for all tests."""
    with patch("domains.training.mobile_training_store.get_training_store") as mock:
        store = MagicMock()
        store.add_pair.return_value = "pair_001"
        store.add_batch.return_value = ["pair_001", "pair_002"]
        store.get_pair.return_value = {
            "_id": "pair_001",
            "user_msg": "hello",
            "assistant_msg": "hi there",
            "session_id": "s1",
            "quality": 0.0,
            "synced": False,
            "used_for_training": False,
            "timestamp": 1000.0,
        }
        store.get_pending_pairs.return_value = [
            {"_id": f"pair_{i}", "user_msg": f"u{i}", "assistant_msg": f"a{i}",
             "session_id": "s1", "quality": 0.0, "synced": False, "timestamp": 1000.0 + i}
            for i in range(5)
        ]
        store.get_pairs_by_session.return_value = [
            {"_id": "pair_001", "user_msg": "hello", "assistant_msg": "hi",
             "session_id": "s1", "quality": 0.5, "timestamp": 1000.0},
        ]
        store.stats.return_value = {
            "total": 100, "pending": 30, "synced": 70, "used": 65,
        }
        store.delete_pair.return_value = True
        store.delete_synced.return_value = 10
        store.update_quality.return_value = True
        store.compact.return_value = 90
        store.count.return_value = 100
        store.quality_breakdown.return_value = {"0": 1, "1": 1, "-1": 1}
        mock.return_value = store
        yield store


class TestTrainStats:
    def test_stats_returns_fields(self):
        resp = client.get("/mobile/train/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 100
        assert data["pending"] == 30
        assert data["synced"] == 70
        assert data["used"] == 65
        assert "by_quality" in data


class TestTrainPending:
    def test_pending_returns_pairs(self):
        resp = client.get("/mobile/train/pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5
        assert len(data["pairs"]) == 5
        assert data["pairs"][0]["id"] == "pair_0"

    def test_pending_respects_limit(self):
        resp = client.get("/mobile/train/pending?limit=2")
        assert resp.status_code == 200


class TestTrainSession:
    def test_session_returns_pairs(self):
        resp = client.get("/mobile/train/session/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "s1"
        assert data["count"] == 1


class TestUpdateQuality:
    def test_update_quality(self):
        resp = client.patch("/mobile/train/pair/pair_001", json={"quality": 1.0})
        assert resp.status_code == 200
        assert resp.json()["quality"] == 1.0

    def test_update_quality_not_found(self, mock_store):
        mock_store.update_quality.return_value = False
        resp = client.patch("/mobile/train/pair/nonexistent", json={"quality": 1.0})
        assert resp.status_code == 404


class TestDeletePair:
    def test_delete_pair(self):
        resp = client.delete("/mobile/train/pair/pair_001")
        assert resp.status_code == 200
        assert resp.json()["pair_id"] == "pair_001"

    def test_delete_pair_not_found(self, mock_store):
        mock_store.delete_pair.return_value = False
        resp = client.delete("/mobile/train/pair/nonexistent")
        assert resp.status_code == 404


class TestDeleteSynced:
    def test_delete_synced(self):
        resp = client.delete("/mobile/train/synced")
        assert resp.status_code == 200
        assert resp.json()["count"] == 10


class TestCompactStore:
    def test_compact(self):
        resp = client.post("/mobile/train/compact")
        assert resp.status_code == 200
        assert resp.json()["count"] == 90


class TestTrainFromSessions:
    def test_insufficient_pairs(self):
        """Returns 400 when server logs have < 5 pairs."""
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[]), \
             patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[]):
            resp = client.post("/mobile/train/from-sessions", json={"limit": 10})
            assert resp.status_code == 400
            assert "5 training pairs" in resp.json()["detail"]

    def test_calls_extractors_in_order(self):
        """Tries sessions first, falls back to logs."""
        sessions_pairs = [
            {"user_msg": f"Question {i}", "assistant_msg": f"Answer {i} with enough text"}
            for i in range(5)
        ]
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=sessions_pairs) as mock_s, \
             patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[]) as mock_l, \
             patch("domains.training.pair_extractor.write_training_text") as mock_w, \
             patch("domains.training.mobile_training_store.get_training_store") as mock_store, \
             patch("subprocess.run") as mock_proc:
            mock_w.return_value = Path("/tmp/test.txt")
            mock_proc.return_value = MagicMock(
                returncode=0,
                stdout='{"success": true, "loss": 1.5, "steps": 8}\n',
                stderr="",
            )
            resp = client.post("/mobile/train/from-sessions", json={"limit": 10})
            assert resp.status_code == 200
            mock_s.assert_called_once()
            # logs NOT called because sessions had enough
            mock_l.assert_not_called()

    def test_falls_back_to_logs(self):
        """Falls back to logs when sessions have < 5 pairs."""
        log_pairs = [
            {"user_msg": f"Question {i}", "assistant_msg": f"Answer {i} with enough text"}
            for i in range(10)
        ]
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[]), \
             patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=log_pairs), \
             patch("domains.training.pair_extractor.write_training_text") as mock_w, \
             patch("domains.training.mobile_training_store.get_training_store") as mock_store, \
             patch("subprocess.run") as mock_proc:
            mock_w.return_value = Path("/tmp/test.txt")
            mock_proc.return_value = MagicMock(
                returncode=0,
                stdout='{"success": true, "loss": 2.0, "steps": 5}\n',
                stderr="",
            )
            resp = client.post("/mobile/train/from-sessions", json={"limit": 10})
            assert resp.status_code == 200
            data = resp.json()
            assert data["loss"] == 2.0
            assert data["steps"] == 5

    def test_training_failure(self):
        """Returns 500 when subprocess fails."""
        pairs = [{"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough"} for i in range(5)]
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=pairs), \
             patch("domains.training.pair_extractor.write_training_text") as mock_w, \
             patch("domains.training.mobile_training_store.get_training_store") as mock_store, \
             patch("subprocess.run") as mock_proc:
            mock_w.return_value = Path("/tmp/test.txt")
            mock_proc.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="RuntimeError: CUDA out of memory",
            )
            resp = client.post("/mobile/train/from-sessions")
            assert resp.status_code == 500


class TestAutoTrainStatus:
    def test_status_returns_fields(self):
        resp = client.get("/mobile/train/auto-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "threshold" in data
        assert "interval_s" in data
        assert "pending_conversations" in data
        assert "total_trains" in data
        assert "last_train" in data
        assert "last_loss" in data
        assert "last_checkpoint" in data
