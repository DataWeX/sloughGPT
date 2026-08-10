"""Tests for meta_weights router — ping, get weights, stats delegation."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

pytest.importorskip("fastapi")

# Ensure apps/api/server is on the path for schemas.common import
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.meta_weights import MetaWeightsRouter, GetMetaWeightsRequest


@pytest.fixture
def mock_manager():
    """Create a mock MetaWeightManager."""
    mgr = MagicMock()
    mgr.get_adjustment.return_value = SimpleNamespace(
        temperature=0.7, repetition_penalty=1.1, top_p=0.85, top_k=40
    )
    mgr._weight_history = [1, 2, 3]
    mgr.get_stats.return_value = {"total_adjustments": 3, "avg_temperature": 0.75}
    return mgr


@pytest.fixture
def app(mock_manager):
    """Create FastAPI app with mocked meta-weight manager."""
    router_instance = MetaWeightsRouter()
    app = FastAPI()
    app.include_router(router_instance.router)

    def _fake_get_manager():
        return mock_manager

    with patch(
        "apps.api.server.routers.meta_weights.MetaWeightsRouter.get_meta_weights.__wrapped__"
        if hasattr(MetaWeightsRouter.get_meta_weights, "__wrapped__")
        else "domains.feedback.get_meta_weight_manager",
        _fake_get_manager,
    ):
        yield app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestPing:
    def test_ping(self):
        """Ping works without any mocked manager."""
        from apps.api.server.routers.meta_weights import MetaWeightsRouter
        app = FastAPI()
        app.include_router(MetaWeightsRouter().router)
        client = TestClient(app)
        resp = client.get("/meta-weights/ping")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "ok"


class TestGetMetaWeights:
    def test_returns_weights(self, mock_manager):
        from apps.api.server.routers.meta_weights import MetaWeightsRouter
        app = FastAPI()
        router_instance = MetaWeightsRouter()
        app.include_router(router_instance.router)

        with patch("domains.feedback.get_meta_weight_manager", return_value=mock_manager):
            client = TestClient(app)
            resp = client.post("/meta-weights/get", json={
                "user_message": "hello",
                "k": 5,
                "user_id": "test",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["temperature"] == 0.7
            assert data["repetition_penalty"] == 1.1
            assert data["top_p"] == 0.85
            assert data["top_k"] == 40
            assert data["based_on_samples"] == 3

    def test_503_when_no_manager(self):
        from apps.api.server.routers.meta_weights import MetaWeightsRouter
        app = FastAPI()
        app.include_router(MetaWeightsRouter().router)

        with patch("domains.feedback.get_meta_weight_manager", return_value=None):
            client = TestClient(app)
            resp = client.post("/meta-weights/get", json={
                "user_message": "hello",
            })
            assert resp.status_code == 503


class TestGetMetaWeightStats:
    def test_returns_stats(self, mock_manager):
        from apps.api.server.routers.meta_weights import MetaWeightsRouter
        app = FastAPI()
        app.include_router(MetaWeightsRouter().router)

        with patch("domains.feedback.get_meta_weight_manager", return_value=mock_manager):
            client = TestClient(app)
            resp = client.get("/meta-weights/stats")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["total_adjustments"] == 3

    def test_503_when_no_manager(self):
        from apps.api.server.routers.meta_weights import MetaWeightsRouter
        app = FastAPI()
        app.include_router(MetaWeightsRouter().router)

        with patch("domains.feedback.get_meta_weight_manager", return_value=None):
            client = TestClient(app)
            resp = client.get("/meta-weights/stats")
            assert resp.status_code == 503
