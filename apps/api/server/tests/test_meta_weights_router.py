"""Tests for the /meta-weights router (feedback-driven weight adaptation)."""

from unittest.mock import patch, MagicMock
from test_support import get_test_client


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


def _mock_weights():
    """Create a mock MetaWeight object."""
    w = MagicMock()
    w.temperature = 0.85
    w.repetition_penalty = 1.1
    w.top_p = 0.9
    w.top_k = 50
    w.style_bias = 0.1
    w.confidence_boost = 0.05
    return w


def _mock_manager():
    """Create a mock MetaWeightManager."""
    mgr = MagicMock()
    mgr.get_adjustment.return_value = _mock_weights()
    mgr.get_stats.return_value = {
        "total_feedback": 10,
        "quality_trend": "improving",
        "avg_temperature": 0.82,
        "history_length": 5,
    }
    mgr._weight_history = [MagicMock()] * 5
    return mgr


class TestPing:
    def test_ping(self):
        client = get_test_client()
        resp = client.get("/meta-weights/ping")
        assert resp.status_code == 200


class TestGetMetaWeights:
    def test_get_weights_success(self):
        client = get_test_client()
        mgr = _mock_manager()
        with patch("domains.feedback.get_meta_weight_manager", return_value=mgr):
            resp = client.post("/meta-weights/get", json={
                "user_message": "Hello world",
                "k": 5,
                "user_id": "default",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "temperature" in data
        assert "repetition_penalty" in data
        assert "top_p" in data
        assert "top_k" in data
        assert "style_bias" in data
        assert "confidence_boost" in data
        assert "based_on_samples" in data
        assert data["temperature"] == 0.85
        assert data["based_on_samples"] == 5

    def test_get_weights_manager_none(self):
        client = get_test_client()
        with patch("domains.feedback.get_meta_weight_manager", return_value=None):
            resp = client.post("/meta-weights/get", json={
                "user_message": "test",
            })
        assert resp.status_code == 503

    def test_get_weights_calls_manager_with_params(self):
        client = get_test_client()
        mgr = _mock_manager()
        with patch("domains.feedback.get_meta_weight_manager", return_value=mgr):
            client.post("/meta-weights/get", json={
                "user_message": "test message",
                "k": 10,
                "user_id": "user123",
            })
        mgr.get_adjustment.assert_called_once_with(
            user_message="test message", k=10, user_id="user123"
        )

    def test_get_weights_defaults(self):
        client = get_test_client()
        mgr = _mock_manager()
        with patch("domains.feedback.get_meta_weight_manager", return_value=mgr):
            resp = client.post("/meta-weights/get", json={
                "user_message": "test",
            })
        assert resp.status_code == 200
        mgr.get_adjustment.assert_called_once_with(
            user_message="test", k=5, user_id="default"
        )


class TestGetMetaWeightStats:
    def test_get_stats_success(self):
        client = get_test_client()
        mgr = _mock_manager()
        with patch("domains.feedback.get_meta_weight_manager", return_value=mgr):
            resp = client.get("/meta-weights/stats")
        assert resp.status_code == 200
        data = _data(resp)
        assert "total_feedback" in data
        assert "quality_trend" in data

    def test_get_stats_manager_none(self):
        client = get_test_client()
        with patch("domains.feedback.get_meta_weight_manager", return_value=None):
            resp = client.get("/meta-weights/stats")
        assert resp.status_code == 503
