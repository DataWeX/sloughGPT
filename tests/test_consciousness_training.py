"""Tests for the consciousness training adapter.

``domain/consciousness/training.py`` was deleted in the layout refactor and the
router's ``_get_trainer()`` still imports it — every training-touching
endpoint (``/consciousness/status``, ``/train/status``, ``/train/start``,
``/stats``, ``/stream``) 500s with ``ModuleNotFoundError``. These tests pin the
router contract and the adapter's behaviour (collect pairs → delegate the real
loop to the consolidated ``HFLoraTrainer`` → fall back to ``data_saved`` when no
model is available).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.consciousness.training import (
    ConsciousnessPair,
    ConsciousnessTrainer,
    TrainingConfig,
)


class TestImports:
    def test_router_import_path_works(self):
        # Regression for the dead import: the router does exactly this.
        import domain.consciousness.training as training

        assert training.ConsciousnessTrainer is ConsciousnessTrainer
        assert training.TrainingConfig is TrainingConfig

    def test_router_get_trainer_returns_real_trainer(self, monkeypatch):
        from types import SimpleNamespace

        from apps.api.server.routers.consciousness import ConsciousnessRouter

        class Cfg:
            lora_rank = 4
            lora_alpha = 8.0

        class Engine:
            config = Cfg()
            self_model = SimpleNamespace(episodes=[])

        monkeypatch.setattr("domain.consciousness.get_consciousness", lambda: Engine())

        router_obj = ConsciousnessRouter()
        trainer = router_obj._get_trainer()
        assert router_obj._trainer is not None
        # Contract the router depends on — the concrete class may live in the
        # router or in domain.consciousness.training, but it must satisfy this.
        assert trainer.config.rank == 4
        assert trainer.config.alpha == 8.0
        assert trainer.config.min_pairs_for_training == 10
        assert callable(trainer.get_status)
        assert callable(trainer.should_train)
        assert hasattr(trainer, "is_training")
        assert hasattr(trainer, "_pairs")


class TestTrainingConfig:
    def test_validate_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path is required"):
            TrainingConfig().validate()

    def test_validate_rank_and_epochs(self):
        with pytest.raises(ValueError, match="rank"):
            TrainingConfig(model_path="m.slnc", rank=0).validate()
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(model_path="m.slnc", epochs=0).validate()


class TestConsciousnessPair:
    def test_rating_clamped(self):
        assert ConsciousnessPair("q", "a", "n", 7).rating == 5
        assert ConsciousnessPair("q", "a", "n", 0).rating == 1

    def test_training_text_format(self):
        pair = ConsciousnessPair("user", "resp", "insight", 4)
        text = pair.to_training_text()
        assert "[INPUT] user" in text
        assert "[RESPONSE] resp" in text
        assert "[CONSCIOUSNESS] insight" in text
        assert "[RATING] 4" in text


class TestConsciousnessTrainer:
    def test_empty_status(self, tmp_path):
        trainer = ConsciousnessTrainer(config=TrainingConfig(data_dir=str(tmp_path)))
        status = trainer.get_status()
        assert status["pairs_collected"] == 0
        assert status["should_train"] is False
        assert status["is_training"] is False
        assert status["training_runs"] == 0
        assert trainer.should_train() is False

    def test_add_pair_persists_and_reloads(self, tmp_path):
        config = TrainingConfig(data_dir=str(tmp_path))
        trainer = ConsciousnessTrainer(config)
        trainer.add_pair("hi there", "hello", "I noticed a pattern", 4, {"novelty": 0.9})
        assert len(trainer._pairs) == 1

        reloaded = ConsciousnessTrainer(config)
        assert len(reloaded._pairs) == 1
        pair = reloaded._pairs[0]
        assert pair.input_text == "hi there"
        assert pair.response == "hello"
        assert pair.narrative == "I noticed a pattern"
        assert pair.rating == 4
        assert pair.qualia == {"novelty": 0.9}

    def test_should_train_once_min_pairs_reached(self, tmp_path):
        config = TrainingConfig(data_dir=str(tmp_path), min_pairs_for_training=2)
        trainer = ConsciousnessTrainer(config)
        assert trainer.should_train() is False
        trainer.add_pair("q", "a", "n", 3)
        assert trainer.should_train() is False
        trainer.add_pair("q2", "a2", "n2", 3)
        assert trainer.should_train() is True
        assert trainer.get_status()["should_train"] is True

    def test_clear_pairs(self, tmp_path):
        config = TrainingConfig(data_dir=str(tmp_path), min_pairs_for_training=1)
        trainer = ConsciousnessTrainer(config)
        trainer.add_pair("q", "a", "n", 3)
        assert trainer.clear_pairs() == 1
        assert len(trainer._pairs) == 0
        trainer.add_pair("q2", "a2", "n2", 3)
        # Persisted store was reset on clear, so reload sees only the new pair.
        reloaded = ConsciousnessTrainer(config)
        assert len(reloaded._pairs) == 1
        assert reloaded._pairs[0].input_text == "q2"

    def test_export_import_pairs(self, tmp_path):
        trainer = ConsciousnessTrainer(config=TrainingConfig(data_dir=str(tmp_path)))
        trainer.add_pair("q1", "a1", "n1", 5)
        trainer.add_pair("q2", "a2", "n2", 2)
        export_path = str(tmp_path / "pairs.json")
        assert trainer.export_pairs(export_path) == 2

        fresh = ConsciousnessTrainer(config=TrainingConfig(data_dir=str(tmp_path / "other")))
        assert fresh.import_pairs(export_path) == 2
        assert [p.input_text for p in fresh._pairs] == ["q1", "q2"]

    def test_train_without_pairs_returns_no_data(self, tmp_path):
        trainer = ConsciousnessTrainer(config=TrainingConfig(data_dir=str(tmp_path)))
        result = trainer.train()
        assert result.success is False
        assert result.status == "no_data"
        assert result.to_dict()["status"] == "no_data"

    def test_train_saves_data_when_model_missing(self, tmp_path):
        config = TrainingConfig(
            data_dir=str(tmp_path),
            adapter_dir=str(tmp_path / "adapters"),
            model_path=str(tmp_path / "missing.slnc"),
            rank=2,
            alpha=4.0,
            epochs=1,
        )
        trainer = ConsciousnessTrainer(config)
        trainer.add_pair("q", "a", "n", 3)
        result = trainer.train()
        assert result.success is True
        assert result.status == "data_saved"
        assert result.to_dict()["success"] is True
        saved = list(Path(tmp_path).glob("consciousness_data_*.jsonl"))
        assert saved, "fallback should persist the training data for later use"
        # A completed run is recorded in status history.
        assert trainer.get_status()["training_runs"] == 1
        assert trainer.get_status()["last_result"]["status"] == "data_saved"
