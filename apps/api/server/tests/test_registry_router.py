"""Tests for the /registry router (model registry proxy)."""

from unittest.mock import MagicMock, patch

from test_support import get_test_client


def _data(resp):
    body = resp.json()
    return body.get("data", body)


MOCK_MODELS = [
    {"model_id": "m1", "type": "local", "status": "loaded"},
    {"model_id": "m2", "type": "remote", "status": "ready"},
]

MOCK_HEALTH = {
    "models_loaded": 1,
    "models_registered": 2,
    "healthy": True,
    "degraded": False,
    "has_errors": False,
    "default_model": "m1",
    "models": MOCK_MODELS,
}


def _make_registry():
    reg = MagicMock()
    reg.list_models.return_value = MOCK_MODELS
    reg.health_summary.return_value = MOCK_HEALTH
    return reg


class TestRegistryListModels:
    def test_list_models(self):
        with patch(
            "domains.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/models")
            assert resp.status_code == 200
            data = _data(resp)
            assert "models" in data
            assert data["count"] == 2

    def test_list_models_empty(self):
        reg = _make_registry()
        reg.list_models.return_value = []
        with patch("domains.infrastructure.model_registry.get_model_registry", return_value=reg):
            client = get_test_client()
            resp = client.get("/registry/models")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["count"] == 0


class TestRegistryGetModel:
    def test_get_model_found(self):
        with patch(
            "domains.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/models/m1")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["model_id"] == "m1"

    def test_get_model_not_found(self):
        with patch(
            "domains.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/models/nonexistent")
            assert resp.status_code == 404


class TestRegistryBest:
    def test_best_model(self):
        with patch(
            "domains.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/best")
            assert resp.status_code == 200
            data = _data(resp)
            assert "models_loaded" in data
            assert data["models_registered"] == 2


class TestRegistryStats:
    def test_stats(self):
        with patch(
            "domains.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/stats")
            assert resp.status_code == 200
            data = _data(resp)
            assert "models_loaded" in data
            assert data["healthy"] is True
