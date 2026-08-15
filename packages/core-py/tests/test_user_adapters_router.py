"""Tests for the user-adapters API router (routers/user_adapters.py).

Covers: UserAdaptersRouter CRUD, merge, aggregate-best, quality, prune, delete.
All domain calls are mocked; only HTTP-level behavior is tested.

Note: the user_adapters router imports get_per_user_lora INSIDE each handler,
so we must patch at 'domains.feedback.get_per_user_lora'.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.user_adapters import router  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(**overrides):
    defaults = dict(
        get_all_adapters=lambda: [
            {"user_id": "u1", "feedback_count": 10},
            {"user_id": "u2", "feedback_count": 5},
        ],
        get_stats=lambda: {"total_adapters": 2, "total_feedback": 15},
        get_adapter=lambda uid: SimpleNamespace(feedback_count=10) if uid == "u1" else None,
        update_adapter=lambda uid, rating: None,
        reset_user_adapter=lambda uid: None,
        merge_all=lambda: None,
        aggregate_best_adapters=lambda **kw: {
            "user_count": 2,
            "total_feedback": 15,
            "output_path": "/tmp/best.npz",
            "eval": {"delta": {"verdict": "better", "perplexity_delta": -0.5}},
        },
        get_quality_report=lambda **kw: {"avg_feedback": 7.5, "total": 15},
        delete_adapter=lambda uid: None,
        prune_low_quality=lambda min_feedback_count=1, max_age_days=30: ["u2"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Tests — patch at 'domains.feedback.get_per_user_lora' (lazy import in handler)
# ---------------------------------------------------------------------------

MOCK_TARGET = "domains.feedback.get_per_user_lora"


class TestListAdapters:
    @patch(MOCK_TARGET)
    def test_returns_list_and_stats(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.get("/user-adapters")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["adapters"]) == 2
        assert data["stats"]["total_adapters"] == 2

    @patch(MOCK_TARGET, side_effect=ImportError)
    def test_import_error_returns_503(self, _mock):
        client = TestClient(_app(), raise_server_exceptions=False)
        resp = client.get("/user-adapters")
        assert resp.status_code == 503


class TestGetAdapter:
    @patch(MOCK_TARGET)
    def test_existing_adapter(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.get("/user-adapters/u1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["exists"] is True
        assert data["feedback_count"] == 10

    @patch(MOCK_TARGET)
    def test_nonexistent_adapter(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.get("/user-adapters/u999")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["exists"] is False


class TestUpdateAdapter:
    @patch(MOCK_TARGET)
    def test_update_rating(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.post("/user-adapters/u1/update", json={"rating": "thumbs_up"})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "updated"


class TestResetAdapter:
    @patch(MOCK_TARGET)
    def test_reset(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.post("/user-adapters/u1/reset")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "reset"


class TestMergeAdapters:
    @patch(MOCK_TARGET)
    def test_merge_all(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.post("/user-adapters/merge")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "merged"


class TestAggregateBest:
    @patch(MOCK_TARGET)
    def test_aggregate_with_eval(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.post("/user-adapters/aggregate-best", json={
            "top_k": 5,
            "min_feedback_count": 3,
            "output_name": "best_v2",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "aggregated_with_eval"
        assert data["eval"]["verdict"] == "better"

    @patch(MOCK_TARGET)
    def test_aggregate_no_eval(self, mock_get):
        store = _make_store(
            aggregate_best_adapters=lambda **kw: {
                "user_count": 2,
                "total_feedback": 15,
                "output_path": "/tmp/best.npz",
                "eval": {"error": "no data"},
            }
        )
        mock_get.return_value = store
        client = TestClient(_app())
        resp = client.post("/user-adapters/aggregate-best", json={})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "aggregated"


class TestGetQuality:
    @patch(MOCK_TARGET)
    def test_returns_quality_report(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.get("/user-adapters/quality")
        assert resp.status_code == 200
        assert "avg_feedback" in resp.json()["data"]


class TestDeleteAdapter:
    @patch(MOCK_TARGET)
    def test_delete(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.delete("/user-adapters/u1")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"


class TestPruneAdapters:
    @patch(MOCK_TARGET)
    def test_prune(self, mock_get):
        mock_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.post("/user-adapters/prune", json={
            "min_feedback_count": 1,
            "max_age_days": 30,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "pruned"
        assert data["deleted_count"] == 1
        assert "u2" in data["deleted_users"]
