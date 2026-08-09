"""
Tests for the meta-weights router — POST /meta-weights/get and GET /meta-weights/stats.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.meta_weights import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _make_weight(temperature=0.8, repetition_penalty=1.0, top_p=0.9, top_k=50):
    return type("W", (), {
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k,
    })()


class TestGetMetaWeights:
    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_adjustment(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight(
            temperature=0.9, repetition_penalty=1.1, top_p=0.95, top_k=40,
        )
        mgr._weight_history = [1, 2, 3]
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["temperature"] == 0.9
        assert resp.json()["based_on_samples"] == 3

    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_503_when_unavailable(self, mock_get_mgr, client):
        mock_get_mgr.return_value = None
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        assert resp.status_code == 503

    @patch("domains.feedback.get_meta_weight_manager")
    def test_default_k_value(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        client.post("/meta-weights/get", json={"user_message": "hello"})
        args, kwargs = mgr.get_adjustment.call_args
        assert kwargs.get("k") == 5 or (len(args) > 1 and args[1] == 5)

    @patch("domains.feedback.get_meta_weight_manager")
    def test_custom_k_value(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        client.post("/meta-weights/get", json={"user_message": "hello", "k": 10})
        args, kwargs = mgr.get_adjustment.call_args
        assert kwargs.get("k") == 10 or (len(args) > 1 and args[1] == 10)

    @patch("domains.feedback.get_meta_weight_manager")
    def test_empty_user_message(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        resp = client.post("/meta-weights/get", json={"user_message": ""})
        assert resp.status_code == 200

    @patch("domains.feedback.get_meta_weight_manager")
    def test_custom_user_id(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        client.post("/meta-weights/get", json={"user_message": "hello", "user_id": "user-42"})
        args, kwargs = mgr.get_adjustment.call_args
        assert kwargs.get("user_id") == "user-42" or (len(args) > 2 and args[2] == "user-42")

    @patch("domains.feedback.get_meta_weight_manager")
    def test_default_user_id(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        client.post("/meta-weights/get", json={"user_message": "hello"})
        args, kwargs = mgr.get_adjustment.call_args
        assert kwargs.get("user_id") == "default" or (len(args) > 2 and args[2] == "default")

    @patch("domains.feedback.get_meta_weight_manager")
    def test_response_contains_all_weight_fields(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight(
            temperature=0.7, repetition_penalty=1.3, top_p=0.85, top_k=30,
        )
        mgr._weight_history = [1]
        resp = client.post("/meta-weights/get", json={"user_message": "test"})
        body = resp.json()
        assert body["temperature"] == 0.7
        assert body["repetition_penalty"] == 1.3
        assert body["top_p"] == 0.85
        assert body["top_k"] == 30

    @patch("domains.feedback.get_meta_weight_manager")
    def test_manager_exception_propagates(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.side_effect = RuntimeError("db unavailable")
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        assert resp.status_code == 500

    @patch("domains.feedback.get_meta_weight_manager")
    def test_many_samples_reflected(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = list(range(1000))
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        assert resp.json()["based_on_samples"] == 1000

    @patch("domains.feedback.get_meta_weight_manager")
    def test_zero_k_falls_back_to_default(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        client.post("/meta-weights/get", json={"user_message": "hi", "k": 0})
        args, kwargs = mgr.get_adjustment.call_args
        assert kwargs.get("k") == 5 or (len(args) > 1 and args[1] == 5)

    @patch("domains.feedback.get_meta_weight_manager")
    def test_negative_k_passes_through(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        client.post("/meta-weights/get", json={"user_message": "hi", "k": -3})
        args, kwargs = mgr.get_adjustment.call_args
        assert kwargs.get("k") == -3 or (len(args) > 1 and args[1] == -3)

    def test_missing_user_message_is_422(self, client):
        resp = client.post("/meta-weights/get", json={"k": 5})
        assert resp.status_code == 422

    def test_empty_body_is_422(self, client):
        resp = client.post("/meta-weights/get", json={})
        assert resp.status_code == 422

    @patch("domains.feedback.get_meta_weight_manager")
    def test_extra_fields_ignored(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight()
        mgr._weight_history = []
        resp = client.post(
            "/meta-weights/get",
            json={"user_message": "hello", "extra": "ignored"},
        )
        assert resp.status_code == 200

    @patch("domains.feedback.get_meta_weight_manager")
    def test_all_zero_weight_values_pass_through(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = _make_weight(
            temperature=0.0, repetition_penalty=0.0, top_p=0.0, top_k=0,
        )
        mgr._weight_history = []
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        body = resp.json()
        assert body["temperature"] == 0.0
        assert body["repetition_penalty"] == 0.0
        assert body["top_p"] == 0.0
        assert body["top_k"] == 0


class TestGetStats:
    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_stats(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_stats.return_value = {"total_adjustments": 10}
        resp = client.get("/meta-weights/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_adjustments"] == 10

    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_503_when_unavailable(self, mock_get_mgr, client):
        mock_get_mgr.return_value = None
        resp = client.get("/meta-weights/stats")
        assert resp.status_code == 503

    @patch("domains.feedback.get_meta_weight_manager")
    def test_empty_stats(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_stats.return_value = {"total_adjustments": 0, "users": 0}
        resp = client.get("/meta-weights/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_adjustments"] == 0

    @patch("domains.feedback.get_meta_weight_manager")
    def test_stats_has_success_status(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_stats.return_value = {"total_adjustments": 5}
        resp = client.get("/meta-weights/stats")
        assert resp.json()["status"] == "success"

    @patch("domains.feedback.get_meta_weight_manager")
    def test_stats_manager_exception(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_stats.side_effect = RuntimeError("corrupted")
        resp = client.get("/meta-weights/stats")
        assert resp.status_code == 500

    @patch("domains.feedback.get_meta_weight_manager")
    def test_stats_passthrough_full_data(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        full = {"total_adjustments": 42, "users": 7, "last_adjustment": "2026-08-01"}
        mgr.get_stats.return_value = full
        resp = client.get("/meta-weights/stats")
        body = resp.json()
        assert body["status"] == "success"
        for key, value in full.items():
            assert body["data"][key] == value


class TestMetaWeightsMethodMismatch:
    """Wrong HTTP methods on meta-weights routes."""

    def test_get_get_405(self, client):
        resp = client.get("/meta-weights/get")
        assert resp.status_code == 405

    def test_stats_post_405(self, client):
        resp = client.post("/meta-weights/stats")
        assert resp.status_code == 405


class TestMetaWeightsValidation:
    """Request body validation bounds."""

    def test_user_message_wrong_type_422(self, client):
        resp = client.post("/meta-weights/get", json={"user_message": 42})
        assert resp.status_code == 422

    def test_k_wrong_type_422(self, client):
        resp = client.post("/meta-weights/get", json={"user_message": "hi", "k": "five"})
        assert resp.status_code == 422

    def test_user_id_wrong_type_422(self, client):
        resp = client.post("/meta-weights/get", json={"user_message": "hi", "user_id": 9})
        assert resp.status_code == 422


class TestPing:
    """GET /meta-weights/ping"""

    def test_ping_ok(self, client):
        resp = client.get("/meta-weights/ping")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"] == {"status": "ok"}

    def test_ping_wrong_method_405(self, client):
        assert client.post("/meta-weights/ping").status_code == 405
        assert client.put("/meta-weights/ping").status_code == 405

    def test_ping_no_manager_required(self, client):
        """ping does not consult the manager — works without a DB-backed store."""
        resp = client.get("/meta-weights/ping")
        assert resp.status_code == 200
