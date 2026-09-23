"""Tests for the consciousness API router (routers/consciousness.py).

Covers: status, self-model, qualia, reflect, config, evaluate, personality, personas,
backup/restore, health, stats, batch, episodes/beliefs history, feedback, seed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

_PM = "domain.consciousness._internal.personality.PersonalityManager"
_PP = "domain.consciousness._internal.personality.PersonalityProfile"
_CE = "domain.consciousness._internal.evaluation.ConsciousnessEvaluator"
_GO = "domain.consciousness.get_consciousness"


def _mock_engine():
    engine = MagicMock()
    engine.get_status.return_value = {
        "enabled": True,
        "level": 1,
        "episodes": 0,
        "current_qualia": {"valence": 0.5, "arousal": 0.3},
        "beliefs": {},
        "last_reflection": None,
    }
    engine.config.level = 1
    engine.config.is_enabled.return_value = True
    engine.config.lora_rank = 4
    engine.config.lora_alpha = 8
    engine.config.reflection_interval = 5
    engine.config.max_tokens = 256
    engine.config.training_enabled = False
    engine.config.training_interval = 10
    engine.config.save = MagicMock()

    sm = MagicMock()
    sm.identity.name = "test"
    sm.identity.capabilities = []
    sm.identity.limitations = []
    sm.identity.values = []
    sm.self_beliefs = {"competence": 0.7, "helpfulness": 0.8}
    sm.self_doubts = []
    sm.episodes = []
    sm._compute_growth.return_value = 0.05
    sm._update_beliefs_from_episode = MagicMock()
    engine.self_model = sm

    qualia = MagicMock()
    qualia.current.to_dict.return_value = {"valence": 0.5, "arousal": 0.3}
    qualia.get_narrative.return_value = "test narrative"
    qualia.history = []
    qualia.experience.return_value = MagicMock()
    qualia.experience.return_value.to_dict.return_value = {"valence": 0.5}
    engine.qualia = qualia

    engine.reflect.return_value = MagicMock()
    engine.reflect.return_value.to_dict.return_value = {
        "narrative": "test",
        "belief_deltas": {},
        "avg_growth": 0.05,
        "trajectory": "improving",
        "episode_count": 0,
    }
    engine.clear_episodes.return_value = 0
    engine.reset_beliefs.return_value = {"competence": 0.7}
    return engine


def _app(engine=None, trainer=None, trainer_missing=False):
    from routers.consciousness import ConsciousnessRouter

    router_inst = ConsciousnessRouter()
    if engine is not None:
        router_inst._engine = engine
    if trainer is not None:
        router_inst._trainer = trainer
    if trainer_missing:
        router_inst._trainer_missing = True

    app = FastAPI()
    app.include_router(router_inst.router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


class TestConsciousnessStatus:
    def test_get_status(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert data["data"]["enabled"] is True

    def test_get_status_with_trainer(self):
        trainer = MagicMock()
        trainer.get_status.return_value = {"available": True, "training": False}
        client = TestClient(_app(engine=_mock_engine(), trainer=trainer))
        resp = client.get("/consciousness/status")
        assert resp.status_code == 200


class TestSelfModel:
    def test_get_self_model(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/self-model")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "identity" in data
        assert "beliefs" in data
        assert "episode_count" in data


class TestQualia:
    def test_get_qualia(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/qualia")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "current" in data
        assert "narrative" in data


class TestReflect:
    def test_reflect(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/reflect")
        assert resp.status_code == 200
        assert "reflection" in resp.json()["data"]


class TestConfig:
    def test_update_config(self):
        engine = _mock_engine()
        client = TestClient(_app(engine=engine))
        resp = client.patch("/consciousness/config", json={"level": 2})
        assert resp.status_code == 200
        assert resp.json()["data"]["level"] == 2

    def test_config_validation(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.patch("/consciousness/config", json={"level": 5})
        assert resp.status_code == 422


class TestEvaluate:
    @patch(_CE)
    def test_evaluate(self, mock_eval_cls):
        evaluator = MagicMock()
        evaluator.evaluate.return_value = MagicMock()
        evaluator.evaluate.return_value.to_dict.return_value = {"score": 80}
        mock_eval_cls.return_value = evaluator
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/evaluate")
        assert resp.status_code == 200


class TestPersonality:
    @patch(_PM)
    def test_get_personality(self, mock_pm_cls):
        manager = MagicMock()
        profile = MagicMock()
        profile.to_dict.return_value = {"traits": {}, "voice": {}}
        manager.get_profile.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/personality")
        assert resp.status_code == 200

    @patch(_PM)
    def test_update_personality(self, mock_pm_cls):
        manager = MagicMock()
        profile = MagicMock()
        profile.to_dict.return_value = {"traits": {"openness": 0.8}, "voice": {}}
        profile.traits = {}
        profile.voice = {}
        manager.get_profile.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.patch(
            "/consciousness/personality",
            json={"traits": {"openness": 0.8}},
        )
        assert resp.status_code == 200

    @patch(_PP)
    @patch(_PM)
    def test_reset_personality(self, mock_pm_cls, mock_pp_cls):
        manager = MagicMock()
        mock_pm_cls.return_value = manager
        default = MagicMock()
        default.to_dict.return_value = {"traits": {}, "voice": {}}
        mock_pp_cls.return_value = default
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/personality/reset")
        assert resp.status_code == 200

    @patch(_PM)
    def test_get_presets(self, mock_pm_cls):
        preset_profile = MagicMock()
        preset_profile.to_dict.return_value = {"traits": {}}
        mock_pm_cls.get_presets = MagicMock(return_value={"default": preset_profile})
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/personality/presets")
        assert resp.status_code == 200

    @patch(_PM)
    def test_apply_preset(self, mock_pm_cls):
        manager = MagicMock()
        profile = MagicMock()
        profile.to_dict.return_value = {"traits": {}}
        manager.apply_preset.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/consciousness/personality/presets/apply",
            json={"preset": "default"},
        )
        assert resp.status_code == 200

    @patch(_PM)
    def test_apply_preset_invalid(self, mock_pm_cls):
        manager = MagicMock()
        manager.apply_preset.side_effect = ValueError("no such preset")
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/consciousness/personality/presets/apply",
            json={"preset": "nonexistent"},
        )
        assert resp.status_code == 400

    @patch(_PM)
    def test_get_conflicts(self, mock_pm_cls):
        manager = MagicMock()
        manager.get_conflicts.return_value = []
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/personality/conflicts")
        assert resp.status_code == 200


class TestPersonas:
    @patch(_PM)
    def test_list_personas(self, mock_pm_cls):
        manager = MagicMock()
        manager.list_personas.return_value = []
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/personas")
        assert resp.status_code == 200

    @patch(_PM)
    def test_save_persona(self, mock_pm_cls):
        manager = MagicMock()
        manager.save_persona.return_value = {"id": "test"}
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/consciousness/personas/save",
            json={"persona_id": "test", "name": "Test"},
        )
        assert resp.status_code == 200

    @patch(_PM)
    def test_get_persona(self, mock_pm_cls):
        manager = MagicMock()
        profile = MagicMock()
        profile.to_dict.return_value = {"traits": {}}
        manager.load_persona.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/personas/test")
        assert resp.status_code == 200

    @patch(_PM)
    def test_get_persona_not_found(self, mock_pm_cls):
        manager = MagicMock()
        manager.load_persona.return_value = None
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/personas/missing")
        assert resp.status_code == 404

    @patch(_PM)
    def test_activate_persona(self, mock_pm_cls):
        manager = MagicMock()
        profile = MagicMock()
        profile.to_dict.return_value = {"traits": {}}
        manager.activate_persona.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/personas/test/activate")
        assert resp.status_code == 200

    @patch(_PM)
    def test_activate_persona_not_found(self, mock_pm_cls):
        manager = MagicMock()
        manager.activate_persona.return_value = None
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/personas/missing/activate")
        assert resp.status_code == 404

    @patch(_PM)
    def test_delete_persona(self, mock_pm_cls):
        manager = MagicMock()
        manager.delete_persona.return_value = True
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.delete("/consciousness/personas/test")
        assert resp.status_code == 200

    @patch(_PM)
    def test_delete_persona_not_found(self, mock_pm_cls):
        manager = MagicMock()
        manager.delete_persona.return_value = False
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.delete("/consciousness/personas/missing")
        assert resp.status_code == 404


class TestBackupRestore:
    @patch(_PM)
    def test_backup(self, mock_pm_cls):
        manager = MagicMock()
        manager.list_personas.return_value = []
        profile = MagicMock()
        profile.to_dict.return_value = {"traits": {}}
        manager.get_profile.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/backup")
        assert resp.status_code == 200
        assert resp.json()["data"]["version"] == 1

    @patch(_PM)
    def test_restore(self, mock_pm_cls):
        engine = _mock_engine()
        manager = MagicMock()
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=engine))
        backup = {
            "version": 1,
            "episodes": [],
            "beliefs": {"competence": 0.9},
            "qualia_history": [],
            "personality": {"traits": {}},
            "personas": {},
            "config": {"level": 2},
        }
        resp = client.post("/consciousness/restore", json={"backup": backup})
        assert resp.status_code == 200
        assert resp.json()["data"]["restored"] is True

    def test_restore_invalid_backup(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/consciousness/restore",
            json={
                "backup": {
                    "version": 99,
                    "episodes": [],
                    "beliefs": {},
                    "personality": {},
                    "config": {},
                }
            },
        )
        assert resp.status_code == 400

    def test_restore_missing_fields(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/restore", json={"backup": {"version": 1}})
        assert resp.status_code == 400


class TestHealth:
    def test_health_check(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/health")
        assert resp.status_code == 200
        assert "health_score" in resp.json()["data"]


class TestStats:
    @patch(_PM)
    def test_get_stats(self, mock_pm_cls):
        manager = MagicMock()
        manager.list_personas.return_value = []
        profile = MagicMock()
        profile.voice = {}
        profile.traits = {}
        manager.get_profile.return_value = profile
        mock_pm_cls.return_value = manager
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/stats")
        assert resp.status_code == 200


class TestClearData:
    def test_clear_episodes(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/clear/episodes")
        assert resp.status_code == 200

    def test_clear_beliefs(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/clear/beliefs")
        assert resp.status_code == 200


class TestSeedData:
    def test_seed_data(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/consciousness/seed", params={"count": 10})
        assert resp.status_code == 200
        assert resp.json()["data"]["seeded"] == 10


class TestEpisodeHistory:
    def test_get_episode_history(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/history/episodes")
        assert resp.status_code == 200
        assert "episodes" in resp.json()["data"]

    def test_get_qualia_history(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/history/qualia")
        assert resp.status_code == 200

    def test_get_beliefs_history(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/consciousness/history/beliefs")
        assert resp.status_code == 200


class TestFeedback:
    def test_submit_feedback(self):
        engine = _mock_engine()
        ep = MagicMock()
        ep.self_insight = "test"
        ep.growth_delta = 0.05
        ep.qualia = {"valence": 0.5}
        ep.input_text = "hello"
        ep.response = "world"
        engine.self_model.episodes = [ep]
        client = TestClient(_app(engine=engine))
        resp = client.post(
            "/consciousness/feedback",
            json={"episode_index": 0, "rating": 4},
        )
        assert resp.status_code == 200

    def test_submit_feedback_not_found(self):
        engine = _mock_engine()
        engine.self_model.episodes = []
        client = TestClient(_app(engine=engine))
        resp = client.post(
            "/consciousness/feedback",
            json={"episode_index": 0, "rating": 3},
        )
        assert resp.status_code == 404


class TestBatch:
    def test_batch_operations(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/consciousness/batch",
            json={
                "operations": [
                    {"action": "config", "params": {"level": 1}},
                    {"action": "reflect", "params": {}},
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2


class TestTrainStatus:
    def test_train_status_builds_adapter(self):
        client = TestClient(_app(engine=_mock_engine(), trainer_missing=True))
        resp = client.get("/consciousness/train/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["pairs_collected"] == 0
