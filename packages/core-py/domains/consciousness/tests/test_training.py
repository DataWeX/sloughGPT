"""Tests for ConsciousnessTrainer."""

import json
import tempfile
from pathlib import Path

import pytest
from domains.consciousness.training import (
    ConsciousnessPair,
    ConsciousnessTrainer,
    TrainingConfig,
)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.rank == 4
        assert cfg.epochs == 2
        assert cfg.min_pairs_for_training == 10

    def test_validate_requires_model(self):
        cfg = TrainingConfig(model_path="")
        with pytest.raises(ValueError, match="model_path"):
            cfg.validate()

    def test_validate_rejects_low_rank(self):
        cfg = TrainingConfig(model_path="test.slnc", rank=0)
        with pytest.raises(ValueError, match="rank"):
            cfg.validate()


class TestConsciousnessPair:
    def test_creation(self):
        pair = ConsciousnessPair(
            input_text="hello",
            response="world",
            narrative="I notice something",
            rating=4,
        )
        assert pair.input_text == "hello"
        assert pair.rating == 4
        assert pair.timestamp > 0

    def test_rating_clamped(self):
        pair = ConsciousnessPair("a", "b", "c", rating=10)
        assert pair.rating == 5
        pair = ConsciousnessPair("a", "b", "c", rating=-1)
        assert pair.rating == 1

    def test_to_training_text(self):
        pair = ConsciousnessPair("q", "a", "narrative", rating=3)
        text = pair.to_training_text()
        assert "[INPUT] q" in text
        assert "[RESPONSE] a" in text
        assert "[CONSCIOUSNESS] narrative" in text
        assert "[RATING] 3" in text


class TestConsciousnessTrainer:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir)
            trainer = ConsciousnessTrainer(cfg)
            assert trainer.is_training is False
            assert len(trainer._pairs) == 0

    def test_add_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir)
            trainer = ConsciousnessTrainer(cfg)
            trainer.add_pair("input", "response", "narrative", rating=4)
            assert len(trainer._pairs) == 1

    def test_should_train_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir, min_pairs_for_training=5)
            trainer = ConsciousnessTrainer(cfg)
            assert trainer.should_train() is False

    def test_should_train_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir, min_pairs_for_training=3)
            trainer = ConsciousnessTrainer(cfg)
            for i in range(3):
                trainer.add_pair(f"q{i}", f"a{i}", f"n{i}", rating=4)
            assert trainer.should_train() is True

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir, min_pairs_for_training=5)
            trainer = ConsciousnessTrainer(cfg)
            status = trainer.get_status()
            assert status["pairs_collected"] == 0
            assert status["should_train"] is False

    def test_pairs_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir)
            trainer = ConsciousnessTrainer(cfg)
            trainer.add_pair("q1", "a1", "n1", rating=4)
            trainer.add_pair("q2", "a2", "n2", rating=3)

            # Load from disk
            trainer2 = ConsciousnessTrainer(cfg)
            assert len(trainer2._pairs) == 2

    def test_clear_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir)
            trainer = ConsciousnessTrainer(cfg)
            trainer.add_pair("q", "a", "n", rating=4)
            count = trainer.clear_pairs()
            assert count == 1
            assert len(trainer._pairs) == 0

    def test_export_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir)
            trainer = ConsciousnessTrainer(cfg)
            trainer.add_pair("q1", "a1", "n1", rating=4)
            trainer.add_pair("q2", "a2", "n2", rating=3)

            export_path = Path(tmpdir) / "export.json"
            trainer.export_pairs(str(export_path))
            assert export_path.exists()

            trainer2 = ConsciousnessTrainer(TrainingConfig(data_dir=tmpdir + "/empty"))
            count = trainer2.import_pairs(str(export_path))
            assert count == 2

    def test_train_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(data_dir=tmpdir)
            trainer = ConsciousnessTrainer(cfg)
            result = trainer.train()
            assert result.success is False
            assert result.status == "no_data"

    def test_train_no_model_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(
                data_dir=tmpdir,
                model_path="/nonexistent/model.slnc",
                min_pairs_for_training=2,
            )
            trainer = ConsciousnessTrainer(cfg)
            for i in range(3):
                trainer.add_pair(f"q{i}", f"a{i}", f"n{i}", rating=4)
            result = trainer.train()
            # Should fallback to data_saved since model doesn't exist
            assert result.success is True
            assert result.status == "data_saved"
