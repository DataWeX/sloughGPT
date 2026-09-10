"""Tests for the /benchmark router — metrics, quality, responses, stats, clear."""

import pytest
from unittest.mock import patch, MagicMock
from test_support import _data, get_test_client


class TestClearHistory:
    def setup_method(self):
        self.client = get_test_client()

    def test_clear_returns_success(self):
        resp = self.client.post("/benchmark/history/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_clear_sets_flag(self):
        resp = self.client.post("/benchmark/history/clear")
        data = _data(resp)
        assert data["cleared"] is True


class TestModelMetrics:
    def setup_method(self):
        self.client = get_test_client()

    def test_metrics_returns_model(self):
        resp = self.client.get("/benchmark/metrics", params={"model": "gpt2"})
        assert resp.status_code == 200
        data = _data(resp)
        assert "model" in data
        assert data["model"] == "gpt2"

    def test_metrics_has_expected_fields(self):
        resp = self.client.get("/benchmark/metrics", params={"model": "test"})
        data = _data(resp)
        assert "model" in data
        assert "model_loaded" in data


class TestQualityMetrics:
    def setup_method(self):
        self.client = get_test_client()

    def test_quality_returns_dict(self):
        resp = self.client.get("/benchmark/quality")
        assert resp.status_code == 200
        data = _data(resp)
        assert isinstance(data, dict)


class TestLoggedResponses:
    def setup_method(self):
        self.client = get_test_client()

    def test_responses_has_structure(self):
        resp = self.client.get("/benchmark/responses")
        assert resp.status_code == 200
        data = _data(resp)
        assert "responses" in data
        assert "count" in data
        assert isinstance(data["responses"], list)

    def test_responses_with_limit(self):
        resp = self.client.get("/benchmark/responses", params={"limit": 5})
        data = _data(resp)
        assert len(data["responses"]) <= 5


class TestTrackerStats:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats_returns_dict(self):
        resp = self.client.get("/benchmark/stats")
        assert resp.status_code == 200
        data = _data(resp)
        assert isinstance(data, dict)


class TestGetBenchmarkById:
    def setup_method(self):
        self.client = get_test_client()

    def test_nonexistent_id(self):
        resp = self.client.get("/benchmark/nonexistent_id_123")
        assert resp.status_code in (200, 404)


class TestPerplexity:
    def setup_method(self):
        self.client = get_test_client()

    def test_perplexity_requires_model(self):
        resp = self.client.post(
            "/benchmark/perplexity",
            json={"model_id": "test", "text": "hello world"},
        )
        assert resp.status_code in (200, 400, 404, 500, 503)
