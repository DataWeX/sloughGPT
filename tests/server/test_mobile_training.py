"""Tests for mobile training data CRUD endpoints."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.mobile import MobileRouter

app = FastAPI()
register_all_handlers(app)
_m = MobileRouter()
app.include_router(_m.router)
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
        store.list_pairs.return_value = [
            {"_id": "pair_001", "user_msg": "hello", "assistant_msg": "hi",
             "quality": 0.8, "session_id": "s1", "timestamp": 1000.0},
        ]
        store.mark_used.return_value = None
        store.mark_synced.return_value = None
        mock.return_value = store
        yield store


class TestTrainStats:
    def test_stats_returns_fields(self):
        resp = client.get("/mobile/train/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 100
        assert data["pending"] == 30
        assert data["synced"] == 70
        assert data["used"] == 65
        assert "by_quality" in data


class TestTrainPending:
    def test_pending_returns_pairs(self):
        resp = client.get("/mobile/train/pending")
        assert resp.status_code == 200
        data = resp.json()["data"]
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
        data = resp.json()["data"]
        assert data["session_id"] == "s1"
        assert data["count"] == 1


class TestUpdateQuality:
    def test_update_quality(self):
        resp = client.patch("/mobile/train/pair/pair_001", json={"quality": 1.0})
        assert resp.status_code == 200
        assert resp.json()["data"]["quality"] == 1.0

    def test_update_quality_not_found(self, mock_store):
        mock_store.update_quality.return_value = False
        resp = client.patch("/mobile/train/pair/nonexistent", json={"quality": 1.0})
        assert resp.status_code == 404


class TestDeletePair:
    def test_delete_pair(self):
        resp = client.delete("/mobile/train/pair/pair_001")
        assert resp.status_code == 200
        assert resp.json()["data"]["pair_id"] == "pair_001"

    def test_delete_pair_not_found(self, mock_store):
        mock_store.delete_pair.return_value = False
        resp = client.delete("/mobile/train/pair/nonexistent")
        assert resp.status_code == 404


class TestDeleteSynced:
    def test_delete_synced(self):
        resp = client.delete("/mobile/train/synced")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 10


class TestCompactStore:
    def test_compact(self):
        resp = client.post("/mobile/train/compact")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 90


def _parse_sse_events(resp) -> list[dict]:
    """Extract JSON events from SSE response (handles both raw SSE and TestClient JSON wrapper)."""
    import json as _json
    text = resp.text
    # TestClient may JSON-encode the SSE text as a string, or concatenate multiple JSON strings
    # Try to decode the full text first
    if text.startswith('"'):
        try:
            decoded = _json.loads(text)
            if isinstance(decoded, str):
                text = decoded
        except _json.JSONDecodeError:
            pass
    events = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(_json.loads(line[6:]))
            except _json.JSONDecodeError:
                pass
        # Handle concatenated JSON strings: "data: {...}"\n"data: {...}"
        elif line.startswith('"data: '):
            try:
                decoded_line = _json.loads(line)
                if decoded_line.startswith("data: "):
                    events.append(_json.loads(decoded_line[6:]))
            except _json.JSONDecodeError:
                pass
    return events


def _make_mock_popen(stdout_lines: list[str], returncode: int = 0, stderr: str = ""):
    """Create a mock Popen that yields lines from stdout."""
    proc = MagicMock()
    proc.returncode = None  # Still running
    lines = list(stdout_lines)
    def _readline():
        if lines:
            return lines.pop(0)
        return ""
    def _wait(timeout=None):
        proc.returncode = returncode
        return None
    proc.stdout.readline = _readline
    proc.stderr.read.return_value = stderr
    proc.stderr.readline.return_value = ""
    proc.poll.return_value = returncode
    proc.wait = _wait
    proc.kill.return_value = None
    return proc


class TestTrainFromSessions:
    def test_insufficient_pairs(self):
        """Yields SSE error event when server logs have < 5 pairs."""
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[]), \
             patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[]):
            resp = client.post("/mobile/train/from-sessions", json={"limit": 10})
            assert resp.status_code == 200
            events = _parse_sse_events(resp)
            assert len(events) >= 1
            err_event = events[-1]
            assert err_event["status"] == "error"
            assert "5 training pairs" in err_event["data"]["error"]

    def test_calls_extractors_in_order(self):
        """Tries sessions first, falls back to logs."""
        sessions_pairs = [
            {"user_msg": f"Question {i}", "assistant_msg": f"Answer {i} with enough text"}
            for i in range(5)
        ]
        mock_proc = _make_mock_popen(
            ['{"phase":"TRAIN","status":"loading","message":"Loading model"}',
             '{"success":true,"loss":1.5,"steps":8,"elapsed_s":1.0,"model_path":"/tmp/final","phase":"COMPLETE","status":"complete"}'],
            returncode=0,
        )
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=sessions_pairs) as mock_s, \
             patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=[]) as mock_l, \
             patch("domains.training.pair_extractor.write_training_text") as mock_w, \
             patch("domains.training.mobile_training_store.get_training_store"), \
             patch("routers.mobile.subprocess.Popen", return_value=mock_proc):
            mock_w.return_value = Path("/tmp/test.txt")
            resp = client.post("/mobile/train/from-sessions", json={"limit": 10})
            assert resp.status_code == 200
            events = _parse_sse_events(resp)
            # Should have GENERATE_DATA + TRAIN events + COMPLETE
            phases = [e["phase"] for e in events]
            assert "GENERATE_DATA" in phases
            mock_s.assert_called_once()
            mock_l.assert_not_called()

    def test_falls_back_to_logs(self):
        """Falls back to logs when sessions have < 5 pairs."""
        log_pairs = [
            {"user_msg": f"Question {i}", "assistant_msg": f"Answer {i} with enough text"}
            for i in range(10)
        ]
        mock_proc = _make_mock_popen(
            ['{"phase":"TRAIN","status":"loading","message":"Loading model"}',
             '{"success":true,"loss":2.0,"steps":5,"elapsed_s":1.0,"model_path":"/tmp/final","phase":"COMPLETE","status":"complete"}'],
            returncode=0,
        )
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=[]), \
             patch("domains.training.pair_extractor.extract_pairs_from_logs", return_value=log_pairs), \
             patch("domains.training.pair_extractor.write_training_text") as mock_w, \
             patch("domains.training.mobile_training_store.get_training_store"), \
             patch("routers.mobile.subprocess.Popen", return_value=mock_proc):
            mock_w.return_value = Path("/tmp/test.txt")
            resp = client.post("/mobile/train/from-sessions", json={"limit": 10})
            assert resp.status_code == 200
            events = _parse_sse_events(resp)
            complete = [e for e in events if e.get("status") == "complete"]
            assert len(complete) >= 1
            assert complete[0]["data"]["loss"] == 2.0
            assert complete[0]["data"]["steps"] == 5

    def test_training_failure(self):
        """Yields SSE error event when subprocess fails."""
        pairs = [{"user_msg": f"Q{i}", "assistant_msg": f"A{i} long enough"} for i in range(5)]
        mock_proc = _make_mock_popen([], returncode=1, stderr="RuntimeError: CUDA out of memory")
        with patch("domains.training.pair_extractor.extract_pairs_from_sessions", return_value=pairs), \
             patch("domains.training.pair_extractor.write_training_text") as mock_w, \
             patch("domains.training.mobile_training_store.get_training_store"), \
             patch("routers.mobile.subprocess.Popen", return_value=mock_proc):
            mock_w.return_value = Path("/tmp/test.txt")
            resp = client.post("/mobile/train/from-sessions")
            assert resp.status_code == 200
            events = _parse_sse_events(resp)
            err_events = [e for e in events if e.get("status") == "error"]
            assert len(err_events) >= 1


class TestAutoTrainStatus:
    def test_status_returns_fields(self):
        resp = client.get("/mobile/train/auto-status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "enabled" in data
        assert "threshold" in data
        assert "interval_s" in data
        assert "pending_conversations" in data
        assert "total_trains" in data
        assert "last_train" in data
        assert "last_loss" in data
        assert "last_checkpoint" in data


class TestListTrainingPairs:
    def test_list_top_level_fields(self):
        resp = client.get("/mobile/train/pairs")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["offset"] == 0
        assert data["total"] == 100
        assert data["count"] == 1

    def test_passes_params(self, mock_store):
        client.get(
            "/mobile/train/pairs",
            params={"limit": 25, "offset": 5, "min_quality": 0.8, "session_id": "s9", "search": "hello"},
        )
        store = mock_store
        last_call = store.list_pairs.call_args
        assert last_call.kwargs["limit"] == 25
        assert last_call.kwargs["offset"] == 5
        assert last_call.kwargs["min_quality"] == 0.8
        assert last_call.kwargs["session_id"] == "s9"
        assert last_call.kwargs["search"] == "hello"

    def test_limit_out_of_range_is_422(self):
        assert client.get("/mobile/train/pairs", params={"limit": 0}).status_code == 422
        assert client.get("/mobile/train/pairs", params={"limit": 501}).status_code == 422

    def test_negative_offset_is_422(self):
        assert client.get("/mobile/train/pairs", params={"offset": -1}).status_code == 422


class TestExportTrainingPairs:
    def test_exports_ndjson(self):
        resp = client.get("/mobile/train/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [l for l in resp.text.strip().split("\n") if l]
        assert len(lines) == 1
        import json as _json
        parsed = _json.loads(lines[0])
        assert parsed["user_msg"] == "hello"
        assert parsed["quality"] == 0.8

    def test_export_limit_out_of_range_is_422(self):
        assert client.get("/mobile/train/export", params={"limit": 5001}).status_code == 422


class TestDeletePairsBulk:
    def test_bulk_delete_counts(self):
        resp = client.delete("/mobile/train/pairs/bulk", params={"ids": ["pair_001", "pair_002"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2

    def test_bulk_delete_missing_ids_is_422(self):
        assert client.delete("/mobile/train/pairs/bulk").status_code == 422

    def test_bulk_delete_skips_failed_deletes(self, mock_store):
        def _delete(pair_id):
            return pair_id != "pair_002"
        mock_store.delete_pair.side_effect = _delete
        resp = client.delete("/mobile/train/pairs/bulk", params={"ids": ["pair_001", "pair_002"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1


class TestMobileTrain:
    def test_insufficient_pairs_is_400(self):
        body = {"checkpoint": "mobile_base", "pairs": [
            {"id": f"p{i}", "user_msg": f"u{i}", "assistant_msg": f"a{i}"} for i in range(3)
        ]}
        resp = client.post("/mobile/train", json=body)
        assert resp.status_code == 400

    def test_missing_pairs_field_is_422(self):
        resp = client.post("/mobile/train", json={"checkpoint": "mobile_base"})
        assert resp.status_code == 422


class TestUpdateAutoConfig:
    def test_updates_threshold(self, mock_store):
        with patch("domains.training.auto_trainer.get_auto_trainer") as mock_get:
            trainer = MagicMock()
            trainer.status.return_value = {"threshold": 42, "interval_s": 60}
            mock_get.return_value = trainer
            resp = client.patch("/mobile/train/auto-config", params={"threshold": 42})
            assert resp.status_code == 200
            assert trainer.threshold == 42
            assert resp.json()["data"]["threshold"] == 42

    def test_updates_interval(self, mock_store):
        with patch("domains.training.auto_trainer.get_auto_trainer") as mock_get:
            trainer = MagicMock()
            trainer.status.return_value = {"threshold": 10, "interval_s": 300}
            mock_get.return_value = trainer
            resp = client.patch("/mobile/train/auto-config", params={"interval_s": 300})
            assert resp.status_code == 200
            assert trainer.interval_s == 300

    def test_threshold_out_of_range_is_422(self):
        assert client.patch("/mobile/train/auto-config", params={"threshold": 0}).status_code == 422
        assert client.patch("/mobile/train/auto-config", params={"threshold": 101}).status_code == 422

    def test_interval_out_of_range_is_422(self):
        assert client.patch("/mobile/train/auto-config", params={"interval_s": 10}).status_code == 422
        assert client.patch("/mobile/train/auto-config", params={"interval_s": 4000}).status_code == 422


class TestParamValidation:
    def test_from_sessions_limit_below_ge_is_422(self):
        assert client.post("/mobile/train/from-sessions", json={"limit": 4}).status_code == 422

    def test_from_sessions_limit_above_le_is_422(self):
        assert client.post("/mobile/train/from-sessions", json={"limit": 501}).status_code == 422

    def test_from_sessions_min_length_zero_is_422(self):
        assert client.post("/mobile/train/from-sessions", json={"min_length": 0}).status_code == 422

    def test_pending_limit_out_of_range_is_422(self):
        assert client.get("/mobile/train/pending", params={"limit": 0}).status_code == 422
        assert client.get("/mobile/train/pending", params={"limit": 501}).status_code == 422

    def test_stats_wrong_method_is_405(self):
        assert client.post("/mobile/train/stats").status_code == 405

    def test_export_wrong_method_is_405(self):
        assert client.post("/mobile/train/export").status_code == 405
