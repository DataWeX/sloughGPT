"""
Tests for the user adapters router — CRUD, merge, aggregate, quality, prune.

Uses a standalone FastAPI app with only the router under test.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.user_adapters import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

STORE_TARGET = "domains.feedback.get_per_user_lora"


def _make_store():
    """Create a mock PerUserLoRAStore with all required methods."""
    store = MagicMock()
    store.get_all_adapters.return_value = [
        {"user_id": "user1", "feedback_count": 12, "quality": 0.8},
        {"user_id": "user2", "feedback_count": 5, "quality": 0.6},
    ]
    store.get_stats.return_value = {"total_adapters": 2, "total_feedback": 17}
    store.get_adapter.return_value = MagicMock(feedback_count=12)
    store.get_quality_report.return_value = {"avg_quality": 0.7, "best_user": "user1"}
    store.prune_low_quality.return_value = ["user3", "user4"]
    store.delete_adapter.return_value = None
    store.update_adapter.return_value = None
    store.reset_adapter.return_value = None
    store.merge_all.return_value = None
    store.aggregate_best_adapters.return_value = {
        "output_path": "data/user_adapters/best.npz",
        "user_count": 3,
        "total_feedback": 15,
        "eval": {
            "delta": {
                "verdict": "improved",
                "perplexity_delta": -1.5,
                "bleu_delta": 0.08,
                "throughput_delta": 5.0,
            },
            "report": "Report text",
        },
    }
    return store


class TestListAdapters:
    """GET /user-adapters"""

    @patch(STORE_TARGET)
    def test_list_adapters(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.get("/user-adapters")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "adapters" in data
        assert "stats" in data
        assert len(data["adapters"]) == 2
        assert data["stats"]["total_adapters"] == 2

    @patch(STORE_TARGET)
    def test_list_adapters_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.get("/user-adapters")
        assert resp.status_code == 503


class TestGetAdapter:
    """GET /user-adapters/{user_id}"""

    @patch(STORE_TARGET)
    def test_get_adapter_exists(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.get("/user-adapters/user1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["user_id"] == "user1"
        assert data["exists"] is True
        assert data["feedback_count"] == 12

    @patch(STORE_TARGET)
    def test_get_adapter_not_found(self, mock_get):
        store = MagicMock()
        store.get_adapter.return_value = None
        mock_get.return_value = store

        resp = client.get("/user-adapters/unknown_user")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["user_id"] == "unknown_user"
        assert data["exists"] is False

    @patch(STORE_TARGET)
    def test_get_adapter_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.get("/user-adapters/user1")
        assert resp.status_code == 503


class TestUpdateAdapter:
    """POST /user-adapters/{user_id}/update"""

    @patch(STORE_TARGET)
    def test_update_adapter(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.post("/user-adapters/user1/update", json={"rating": "good"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "updated"
        assert data["user_id"] == "user1"
        store.update_adapter.assert_called_once_with("user1", rating="good")

    @patch(STORE_TARGET)
    def test_update_adapter_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.post("/user-adapters/user1/update", json={"rating": "good"})
        assert resp.status_code == 503


class TestResetAdapter:
    """POST /user-adapters/{user_id}/reset"""

    @patch(STORE_TARGET)
    def test_reset_adapter(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.post("/user-adapters/user1/reset")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "reset"
        assert data["user_id"] == "user1"
        store.reset_adapter.assert_called_once_with("user1")

    @patch(STORE_TARGET)
    def test_reset_adapter_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.post("/user-adapters/user1/reset")
        assert resp.status_code == 503


class TestMergeAdapters:
    """POST /user-adapters/merge"""

    @patch(STORE_TARGET)
    def test_merge_adapters(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.post("/user-adapters/merge")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "merged"
        store.merge_all.assert_called_once()

    @patch(STORE_TARGET)
    def test_merge_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.post("/user-adapters/merge")
        assert resp.status_code == 503


class TestAggregateBest:
    """POST /user-adapters/aggregate-best"""

    @patch(STORE_TARGET)
    def test_aggregate_best_with_eval(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.post("/user-adapters/aggregate-best", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "aggregated_with_eval"
        assert data["eval"]["verdict"] == "improved"
        assert data["output_path"] == "data/user_adapters/best.npz"

    @patch(STORE_TARGET)
    def test_aggregate_best_eval_error(self, mock_get):
        store = _make_store()
        store.aggregate_best_adapters.return_value = {
            "output_path": "data/user_adapters/best.npz",
            "user_count": 3,
            "total_feedback": 15,
            "eval": {"error": "eval failed"},
        }
        mock_get.return_value = store

        resp = client.post("/user-adapters/aggregate-best", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "aggregated"

    @patch(STORE_TARGET)
    def test_aggregate_best_passes_params(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        client.post(
            "/user-adapters/aggregate-best",
            json={"top_k": 20, "min_feedback_count": 10, "output_name": "custom_agg"},
        )
        store.aggregate_best_adapters.assert_called_once_with(
            top_k=20,
            min_feedback_count=10,
            output_name="custom_agg",
        )

    @patch(STORE_TARGET)
    def test_aggregate_best_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.post("/user-adapters/aggregate-best", json={})
        assert resp.status_code == 503

    @patch(STORE_TARGET)
    def test_aggregate_best_default_params(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        client.post("/user-adapters/aggregate-best", json={})
        store.aggregate_best_adapters.assert_called_once_with(
            top_k=10,
            min_feedback_count=5,
            output_name="best_aggregated",
        )


class TestGetQuality:
    """GET /user-adapters/quality"""

    @patch(STORE_TARGET)
    def test_get_quality(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.get("/user-adapters/quality")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, dict)

    @patch(STORE_TARGET)
    def test_get_quality_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.get("/user-adapters/quality")
        assert resp.status_code == 503


class TestDeleteAdapter:
    """DELETE /user-adapters/{user_id}"""

    @patch(STORE_TARGET)
    def test_delete_adapter(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.delete("/user-adapters/user1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "deleted"
        assert data["user_id"] == "user1"
        store.delete_adapter.assert_called_once_with("user1")

    @patch(STORE_TARGET)
    def test_delete_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.delete("/user-adapters/user1")
        assert resp.status_code == 503


class TestPruneAdapters:
    """POST /user-adapters/prune"""

    @patch(STORE_TARGET)
    def test_prune_adapters(self, mock_get):
        store = _make_store()
        mock_get.return_value = store

        resp = client.post("/user-adapters/prune", json={"min_feedback_count": 3, "max_age_days": 15})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "pruned"
        assert data["deleted_count"] == 2
        assert "user3" in data["deleted_users"]
        store.prune_low_quality.assert_called_once_with(min_feedback_count=3, max_age_days=15)

    @patch(STORE_TARGET)
    def test_prune_no_deletions(self, mock_get):
        store = _make_store()
        store.prune_low_quality.return_value = []
        mock_get.return_value = store

        resp = client.post("/user-adapters/prune", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["deleted_count"] == 0

    @patch(STORE_TARGET)
    def test_prune_default_params(self, mock_get):
        store = _make_store()
        store.prune_low_quality.return_value = []
        mock_get.return_value = store

        client.post("/user-adapters/prune", json={})
        store.prune_low_quality.assert_called_once_with(min_feedback_count=1, max_age_days=30)

    @patch(STORE_TARGET)
    def test_prune_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        resp = client.post("/user-adapters/prune", json={})
        assert resp.status_code == 503


class TestUserAdapterMethods:
    """405s for disallowed methods"""

    def test_list_post_is_405(self):
        resp = client.post("/user-adapters")
        assert resp.status_code == 405

    def test_quality_post_is_405(self):
        resp = client.post("/user-adapters/quality")
        assert resp.status_code == 405

    def test_update_get_is_405(self):
        resp = client.get("/user-adapters/user1/update")
        assert resp.status_code == 405

    def test_reset_get_is_405(self):
        resp = client.get("/user-adapters/user1/reset")
        assert resp.status_code == 405

    @patch(STORE_TARGET)
    def test_get_on_merge_routes_to_get_adapter(self, mock_get):
        """GET /merge is shadowed by /{user_id} — returns adapter lookup, not 405."""
        store = _make_store()
        store.get_adapter.return_value = None
        mock_get.return_value = store
        resp = client.get("/user-adapters/merge")
        assert resp.status_code == 200
        assert resp.json()["data"]["user_id"] == "merge"
        assert resp.json()["data"]["exists"] is False

    @patch(STORE_TARGET)
    def test_get_on_aggregate_best_routes_to_get_adapter(self, mock_get):
        store = _make_store()
        store.get_adapter.return_value = None
        mock_get.return_value = store
        resp = client.get("/user-adapters/aggregate-best")
        assert resp.status_code == 200
        assert resp.json()["data"]["user_id"] == "aggregate-best"

    @patch(STORE_TARGET)
    def test_get_on_prune_routes_to_get_adapter(self, mock_get):
        store = _make_store()
        store.get_adapter.return_value = None
        mock_get.return_value = store
        resp = client.get("/user-adapters/prune")
        assert resp.status_code == 200
        assert resp.json()["data"]["user_id"] == "prune"


class TestUserAdapterValidation:
    """422s for malformed request bodies"""

    def test_update_missing_rating_is_422(self):
        resp = client.post("/user-adapters/user1/update", json={})
        assert resp.status_code == 422

    def test_update_rating_wrong_type_is_422(self):
        resp = client.post("/user-adapters/user1/update", json={"rating": 5})
        assert resp.status_code == 422

    def test_prune_wrong_types_is_422(self):
        resp = client.post("/user-adapters/prune", json={"min_feedback_count": "three"})
        assert resp.status_code == 422
        resp = client.post("/user-adapters/prune", json={"max_age_days": "long"})
        assert resp.status_code == 422

    @patch(STORE_TARGET)
    def test_aggregate_best_invalid_types_422(self, mock_get):
        resp = client.post("/user-adapters/aggregate-best", json={"top_k": "ten"})
        assert resp.status_code == 422
        resp = client.post("/user-adapters/aggregate-best", json={"min_feedback_count": "five"})
        assert resp.status_code == 422


class TestAggregateBestEvalDefaults:
    """POST /user-adapters/aggregate-best — missing delta fields default to unknown"""

    @patch(STORE_TARGET)
    def test_eval_without_delta_uses_unknown_verdict(self, mock_get):
        store = _make_store()
        store.aggregate_best_adapters.return_value = {
            "output_path": "x.npz",
            "user_count": 1,
            "total_feedback": 2,
            "eval": {"report": "no delta"},
        }
        mock_get.return_value = store
        resp = client.post("/user-adapters/aggregate-best", json={})
        data = resp.json()["data"]
        assert data["status"] == "aggregated_with_eval"
        assert data["eval"]["verdict"] == "unknown"
        assert data["eval"]["perplexity_delta"] is None

    @patch(STORE_TARGET)
    def test_eval_error_returns_aggregated(self, mock_get):
        store = _make_store()
        store.aggregate_best_adapters.return_value = {
            "user_count": 4,
            "eval": {"error": "crashed"},
        }
        mock_get.return_value = store
        resp = client.post("/user-adapters/aggregate-best", json={})
        data = resp.json()["data"]
        assert data["status"] == "aggregated"
        assert data["count"] == 4
