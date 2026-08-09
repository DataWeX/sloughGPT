"""
Tests for the metrics router — GET /metrics and GET /metrics/prometheus.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.metrics import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetMetrics:
    """GET /metrics"""

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_returns_metrics_json(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 100.0
        coll._model_loaded = True
        coll._model_name = "gpt2"
        coll._active_requests = 2
        coll._inference_count = 42
        coll._tokens_generated = 5000
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["inferences_total"] == 42
        assert data["tokens_generated_total"] == 5000

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", None)
    @patch("state.provider", None)
    def test_model_not_loaded(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 0.0
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_loaded"] is False
        assert data["model_name"] is None
        assert data["active_requests"] == 0

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_uptime_is_string(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 12345.678
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        assert isinstance(resp.json()["data"]["uptime_seconds"], str)

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_zero_inferences(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 0.0
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["inferences_total"] == 0
        assert data["tokens_generated_total"] == 0

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_large_inference_count(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 0.0
        coll._model_loaded = True
        coll._model_name = "gpt2"
        coll._active_requests = 0
        coll._inference_count = 999999
        coll._tokens_generated = 50000000
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["inferences_total"] == 999999
        assert data["tokens_generated_total"] == 50000000

    def test_wrong_method_returns_405(self, client):
        resp = client.post("/metrics")
        assert resp.status_code == 405

    def test_wrong_method_prometheus_returns_405(self, client):
        resp = client.post("/metrics/prometheus")
        assert resp.status_code == 405

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_success_wrapper_status_key(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 0.0
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        body = resp.json()
        assert body["status"] == "success"
        assert set(body["data"].keys()) == {
            "uptime_seconds",
            "model_loaded",
            "model_name",
            "active_requests",
            "inferences_total",
            "tokens_generated_total",
        }

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_uptime_exact_string(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 100.0
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        assert resp.json()["data"]["uptime_seconds"] == "100.0"


class TestStateOverride:
    """GET /metrics — state.model / state.provider override branch"""

    def _coll(self):
        coll = MagicMock()
        coll._start_time = 0.0
        coll._model_loaded = False
        coll._model_name = "gpt2"
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        return coll

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", object())
    @patch("state.provider", None)
    @patch("state.model_type", "qwen2.5")
    def test_state_model_forces_loaded_and_type_wins(self, mock_get_coll, client):
        mock_get_coll.return_value = self._coll()
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["model_name"] == "qwen2.5"

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", None)
    @patch("state.provider", object())
    @patch("state.model_type", None)
    def test_provider_only_forces_loaded_fallback_name(self, mock_get_coll, client):
        mock_get_coll.return_value = self._coll()
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["model_name"] == "gpt2"

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("state.model_type", None)
    def test_no_state_uses_collector_values(self, mock_get_coll, client):
        coll = self._coll()
        coll._model_loaded = True
        coll._model_name = "gpt2-medium"
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["model_name"] == "gpt2-medium"

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", object())
    @patch("state.provider", None)
    @patch("state.model_type", "tinyllama")
    def test_active_requests_passthrough(self, mock_get_coll, client):
        coll = self._coll()
        coll._active_requests = 3
        coll._inference_count = 7
        coll._tokens_generated = 1234
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["active_requests"] == 3
        assert data["inferences_total"] == 7
        assert data["tokens_generated_total"] == 1234

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", object())
    @patch("state.provider", None)
    @patch("state.model_type", "")
    def test_empty_model_type_falls_back_to_collector_name(self, mock_get_coll, client):
        coll = self._coll()
        coll._model_loaded = False
        coll._model_name = "fallback-model"
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["model_name"] == "fallback-model"

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    @patch("state.model", object())
    @patch("state.provider", None)
    @patch("state.model_type", "qwen2.5")
    def test_model_type_trumps_collector_name(self, mock_get_coll, client):
        coll = self._coll()
        coll._model_loaded = False
        coll._model_name = "collector-name"
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_name"] == "qwen2.5"


class TestPrometheusMetrics:
    """GET /metrics/prometheus"""

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_returns_prometheus_text(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = "# HELP ..."
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert resp.text == "# HELP ..."
        assert resp.headers["content-type"].startswith("text/plain")

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_content_type_version(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = "# EMPTY"
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert "version=0.0.4" in resp.headers["content-type"]

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_empty_render(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = ""
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_render_called_once(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = "# data"
        mock_get_coll.return_value = coll
        client.get("/metrics/prometheus")
        coll.render.assert_called_once()

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_charset_utf8(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = "# HELP test"
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert "charset=utf-8" in resp.headers["content-type"]

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_render_none_returns_empty(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = None
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert resp.text == ""

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_renders_multiple_lines_preserved(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = "# HELP a\n# TYPE b\n"
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert resp.text == "# HELP a\n# TYPE b\n"

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_collector_import_error_returns_500(self, mock_get_coll, client):
        mock_get_coll.side_effect = RuntimeError("collector broken")
        resp = client.get("/metrics")
        assert resp.status_code == 500

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_prometheus_render_error_returns_500(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.side_effect = RuntimeError("render broken")
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 500

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_metrics_json_content_type(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 0.0
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        assert resp.headers["content-type"].startswith("application/json")

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_uptime_int_start_time(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 123
        coll._model_loaded = False
        coll._model_name = None
        coll._active_requests = 0
        coll._inference_count = 0
        coll._tokens_generated = 0
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        assert resp.json()["data"]["uptime_seconds"] == "123"
