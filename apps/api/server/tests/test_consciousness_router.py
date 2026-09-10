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

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_episode_history(self, _auth, client, mock_engine):
        res = client.get("/consciousness/history/episodes")
        assert res.status_code == 200
        data = res.json()
        assert "episodes" in data["data"]
        assert "total" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_qualia_history(self, _auth, client, mock_engine):
        res = client.get("/consciousness/history/qualia")
        assert res.status_code == 200
        data = res.json()
        assert "history" in data["data"]
        assert "total" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_beliefs_history_empty(self, _auth, client):
        res = client.get("/consciousness/history/beliefs")
        assert res.status_code == 200
        data = res.json()
        assert "beliefs" in data["data"]
        assert "labels" in data["data"]
        assert isinstance(data["data"]["beliefs"], list)

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_beliefs_history_with_episodes(self, _auth, mock_engine):
        """Test beliefs history reconstructs evolution from episodes."""
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler
        import time

        router_obj = ConsciousnessRouter()

        # Create a mock engine with episodes
        mock_eng = MagicMock()
        mock_eng.config.is_enabled.return_value = True
        mock_eng.config.level = 1

        # Create fake episodes with qualia
        from domains.consciousness.self_model import SelfEpisode
        episodes = [
            SelfEpisode(
                timestamp=time.time() - 100 + i,
                input_text=f"test {i}",
                response=f"response {i}",
                qualia={"novelty": 0.8 if i % 2 == 0 else 0.3, "coherence": 0.5},
                self_insight="test insight",
                growth_delta=0.05 if i % 3 == 0 else -0.01,
            )
            for i in range(10)
        ]
        mock_eng.self_model.episodes = episodes

        router_obj._engine = mock_eng

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.get("/consciousness/history/beliefs")
        assert res.status_code == 200
        data = res.json()
        beliefs = data["data"]["beliefs"]
        assert len(beliefs) > 0
        assert "competence" in beliefs[0]
        assert "helpfulness" in beliefs[0]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_submit_feedback(self, _auth, mock_engine):
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler
        from domains.consciousness.self_model import SelfEpisode
        import time

        router_obj = ConsciousnessRouter()
        mock_eng = MagicMock()
        mock_eng.config.is_enabled.return_value = True
        mock_eng.config.level = 1

        episode = SelfEpisode(
            timestamp=time.time(),
            input_text="test input",
            response="test response that is longer",
            qualia={"novelty": 0.5, "valence": 0.2},
            self_insight="test insight",
            growth_delta=0.01,
        )
        mock_eng.self_model.episodes = [episode]
        mock_eng.self_model._compute_growth.return_value = 0.05
        mock_eng.self_model._update_beliefs_from_episode = MagicMock()
        router_obj._engine = mock_eng

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.post("/consciousness/feedback", json={
            "episode_index": 0,
            "rating": 4,
        })
        assert res.status_code == 200
        data = res.json()
        assert "new_growth_delta" in data["data"]
        assert "beliefs" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_submit_feedback_invalid_index(self, _auth):
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler

        router_obj = ConsciousnessRouter()
        mock_eng = MagicMock()
        mock_eng.config.is_enabled.return_value = True
        mock_eng.config.level = 1
        mock_eng.self_model.episodes = []
        router_obj._engine = mock_eng

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.post("/consciousness/feedback", json={
            "episode_index": 999,
            "rating": 3,
        })
        assert res.status_code == 404

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_seed_data(self, _auth):
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler
        from domains.consciousness.self_model import SelfModel
        from domains.consciousness.qualia import QualiaEngine
        from domains.consciousness.meta_cognition import MetaCognition
        from domains.consciousness.config import ConsciousnessConfig
        from domains.consciousness.narrative import NarrativeGenerator

        router_obj = ConsciousnessRouter()
        real_engine = MagicMock()
        real_engine.config = ConsciousnessConfig()
        real_engine.self_model = SelfModel()
        real_engine.qualia = QualiaEngine()
        router_obj._engine = real_engine

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.post("/consciousness/seed?count=10")
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["seeded"] == 10
        assert data["data"]["total_episodes"] >= 10

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_seed_data_default_count(self, _auth, client, mock_engine):
        res = client.post("/consciousness/seed")
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["seeded"] == 30

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_personality(self, _auth, client, mock_engine):
        res = client.get("/consciousness/personality")
        assert res.status_code == 200
        data = res.json()
        assert "values" in data["data"]
        assert "goals" in data["data"]
        assert "voice" in data["data"]
        assert "traits" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_update_personality(self, _auth, client, mock_engine):
        res = client.patch("/consciousness/personality", json={
            "values": ["courage", "wisdom"],
            "voice": {"humor": 0.8},
        })
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["values"] == ["courage", "wisdom"]
        assert data["data"]["voice"]["humor"] == 0.8

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_reset_personality(self, _auth, client, mock_engine):
        client.patch("/consciousness/personality", json={"values": ["test"]})
        res = client.post("/consciousness/personality/reset")
        assert res.status_code == 200
        data = res.json()
        assert "helpfulness" in data["data"]["values"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_personality_history_empty(self, _auth, client, mock_engine):
        res = client.get("/consciousness/personality/history")
        assert res.status_code == 200
        data = res.json()
        assert "history" in data["data"]
        assert "labels" in data["data"]
        assert isinstance(data["data"]["history"], list)

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_personality_history_with_episodes(self, _auth):
        from apps.api.server.routers.consciousness import ConsciousnessRouter
        from infrastructure.exception_handlers import register_app_error_handler
        from domains.consciousness.self_model import SelfEpisode
        import time

        router_obj = ConsciousnessRouter()
        mock_eng = MagicMock()
        mock_eng.config.is_enabled.return_value = True
        mock_eng.config.level = 1

        episodes = [
            SelfEpisode(
                timestamp=time.time() - 100 + i,
                input_text=f"test {i}",
                response=f"response {i}",
                qualia={"novelty": 0.8 if i % 2 == 0 else 0.3, "coherence": 0.5, "valence": 0.6},
                self_insight="test insight",
                growth_delta=0.06 if i % 3 == 0 else 0.01,
            )
            for i in range(10)
        ]
        mock_eng.self_model.episodes = episodes
        router_obj._engine = mock_eng

        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        res = client.get("/consciousness/personality/history")
        assert res.status_code == 200
        data = res.json()
        history = data["data"]["history"]
        assert len(history) > 0
        assert "voice" in history[0]
        assert "traits" in history[0]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_presets(self, _auth, client, mock_engine):
        res = client.get("/consciousness/personality/presets")
        assert res.status_code == 200
        data = res.json()
        assert "presets" in data["data"]
        assert "names" in data["data"]
        assert "default" in data["data"]["names"]
        assert "formal" in data["data"]["names"]
        assert "creative" in data["data"]["names"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_apply_preset(self, _auth, client, mock_engine):
        res = client.post("/consciousness/personality/presets/apply", json={"preset": "formal"})
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["voice"]["formality"] > 0.8

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_apply_preset_invalid(self, _auth, client, mock_engine):
        res = client.post("/consciousness/personality/presets/apply", json={"preset": "nonexistent"})
        assert res.status_code == 400

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_conflicts(self, _auth, client, mock_engine):
        res = client.get("/consciousness/personality/conflicts")
        assert res.status_code == 200
        data = res.json()
        assert "conflicts" in data["data"]
        assert "count" in data["data"]
        assert isinstance(data["data"]["conflicts"], list)

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_list_personas_empty(self, _auth, client, mock_engine):
        res = client.get("/consciousness/personas")
        assert res.status_code == 200
        data = res.json()
        assert "personas" in data["data"]
        assert "count" in data["data"]

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_save_and_activate_persona(self, _auth, client, mock_engine):
        # Save a persona
        res = client.post("/consciousness/personas/save", json={
            "persona_id": "test-hero",
            "name": "Test Hero",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["data"]["id"] == "test-hero"
        assert data["data"]["name"] == "Test Hero"

        # List should show it
        res = client.get("/consciousness/personas")
        assert res.status_code == 200
        personas = res.json()["data"]["personas"]
        assert any(p["id"] == "test-hero" for p in personas)

        # Activate it
        res = client.post("/consciousness/personas/test-hero/activate")
        assert res.status_code == 200

        # Get it
        res = client.get("/consciousness/personas/test-hero")
        assert res.status_code == 200

        # Delete it
        res = client.delete("/consciousness/personas/test-hero")
        assert res.status_code == 200

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_get_persona_not_found(self, _auth, client, mock_engine):
        res = client.get("/consciousness/personas/nonexistent")
        assert res.status_code == 404

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_activate_persona_not_found(self, _auth, client, mock_engine):
        res = client.post("/consciousness/personas/nonexistent/activate")
        assert res.status_code == 404

    @patch("apps.api.server.routers.consciousness.require_auth_if_enabled", return_value=None)
    def test_delete_persona_not_found(self, _auth, client, mock_engine):
        res = client.delete("/consciousness/personas/nonexistent")
        assert res.status_code == 404
