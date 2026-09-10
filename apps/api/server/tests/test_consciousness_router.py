"""Tests for the Consciousness API router."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.config.is_enabled.return_value = True
    engine.config.level = 1
    engine.config.lora_rank = 4
    engine.config.lora_alpha = 8.0
    engine.get_status.return_value = {
        "enabled": True,
        "level": 1,
        "current_qualia": {
            "valence": 0.5,
            "arousal": 0.3,
            "novelty": 0.6,
            "coherence": 0.7,
            "salience": 0.4,
            "certainty": 0.3,
            "complexity": 0.5,
        },
        "beliefs": {"helpful": 0.9},
        "episodes": 5,
        "narrative": "Test narrative",
        "training": {
            "is_training": False,
            "total_pairs": 10,
            "current_epoch": 0,
            "loss": 0.0,
        },
    }
    engine.reflect.return_value = "I am reflecting."
    return engine


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.is_training = False
    trainer.config.min_pairs_for_training = 5
    trainer.config.model_path = ""
    trainer.should_train.return_value = True
    trainer.get_status.return_value = {
        "is_training": False,
        "total_pairs": 10,
        "current_epoch": 0,
        "loss": 0.0,
    }
    trainer._pairs = list(range(10))
    train_result = MagicMock()
    train_result.to_dict.return_value = {"loss": 0.5, "epochs": 1}
    trainer.train.return_value = train_result
    return trainer


@pytest.fixture
def client(mock_engine, mock_trainer):
    from apps.api.server.routers.consciousness import ConsciousnessRouter
    from infrastructure.exception_handlers import register_app_error_handler

    router_obj = ConsciousnessRouter()
    router_obj._engine = mock_engine
    router_obj._trainer = mock_trainer

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(router_obj.router)
    return TestClient(app)


class TestConsciousnessAPI:
    """Tests for the consciousness API endpoints."""

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_status(self, _auth, client):
        res = client.get("/consciousness/status")
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["enabled"] is True
        assert data["data"]["level"] == 1
        assert "training" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_self_model(self, _auth, client):
        res = client.get("/consciousness/self-model")
        assert res.status_code == 200
        data = res.json()
        assert "identity" in data["data"]
        assert "beliefs" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_qualia(self, _auth, client):
        res = client.get("/consciousness/qualia")
        assert res.status_code == 200
        data = res.json()
        assert "current" in data["data"]
        assert "narrative" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_reflect(self, _auth, client):
        res = client.post("/consciousness/reflect")
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["reflection"] == "I am reflecting."

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_update_config(self, _auth, client):
        res = client.patch("/consciousness/config", json={"level": 2})
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["level"] == 2
        assert data["data"]["enabled"] is True

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_update_config_invalid_level(self, _auth, client):
        res = client.patch("/consciousness/config", json={"level": 5})
        assert res.status_code == 422  # Validation error

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_train_status(self, _auth, client):
        res = client.get("/consciousness/train/status")
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["is_training"] is False

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_train_start_success(self, _auth, client):
        res = client.post("/consciousness/train/start", json={"model_path": ""})
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["loss"] == 0.5
        assert data["data"]["epochs"] == 1

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_train_start_busy(self, _auth, mock_engine):
        """Test that train_start returns 409 when already training."""
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler

        router_obj = ConsciousnessRouter()
        router_obj._engine = mock_engine
        mock_trainer_busy = MagicMock()
        mock_trainer_busy.is_training = True
        mock_trainer_busy.should_train.return_value = True
        router_obj._trainer = mock_trainer_busy

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.post("/consciousness/train/start", json={"model_path": ""})
        assert res.status_code == 409

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_train_start_insufficient_data(self, _auth, mock_engine):
        """Test that train_start returns 400 when insufficient data."""
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler

        router_obj = ConsciousnessRouter()
        router_obj._engine = mock_engine
        mock_trainer_insufficient = MagicMock()
        mock_trainer_insufficient.is_training = False
        mock_trainer_insufficient.should_train.return_value = False
        mock_trainer_insufficient.config.min_pairs_for_training = 5
        mock_trainer_insufficient._pairs = [1, 2]
        router_obj._trainer = mock_trainer_insufficient

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.post("/consciousness/train/start", json={"model_path": ""})
        assert res.status_code == 400

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_evaluate(self, _auth, client):
        res = client.get("/consciousness/evaluate")
        assert res.status_code == 200
        data = res.json()
        assert "overall_score" in data["data"]
        # The evaluator returns a flat dict with metrics directly
        assert "narrative_coherence" in data["data"] or "metrics" in data["data"]
