"""
Training Model Card, Auto-Config, and Dataset Manifest Tests.

Tests model card generation, auto-configuration, and dataset manifest handling.

Usage:
    .venv/bin/python -m pytest tests/test_training_model_card_config.py -x -v
"""
import tempfile
import time

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

FAST_CONFIG = {
    "method": "sft",
    "data_quality_threshold": 0.0,
    "epochs": 1,
    "batch_size": 8,
    "block_size": 32,
    "max_steps": 2,
    "n_embed": 32,
    "n_layer": 2,
    "n_head": 2,
}


class TestModelCard:
    """Tests for model card generation."""

    def test_model_card_creation(self):
        from domains.training.model_card import ModelCard

        card = ModelCard(
            name="test_model",
            version="1.0",
            description="A test model",
        )
        assert card.name == "test_model"
        assert card.version == "1.0"

    def test_model_card_to_dict(self):
        from domains.training.model_card import ModelCard

        card = ModelCard(
            name="test_model",
            version="1.0",
            description="A test model",
        )
        d = card.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_model"

    def test_generate_model_card(self):
        from domains.training.model_card import generate_model_card

        card = generate_model_card(
            name="test_model",
            model_type="slnet",
            training_info={"epochs": 3, "loss": 0.5},
        )
        assert card is not None
        assert card.name == "test_model"

    def test_model_card_from_outcome(self):
        from domains.training.model_card import generate_model_card_from_outcome
        from domains.training.outcome_tracker import TrainingOutcome

        outcome = TrainingOutcome(
            run_id="test_run",
            timestamp=time.time(),
            dataset_size=1000,
            model="gpt2",
            method="sft",
            final_loss=0.5,
        )
        card = generate_model_card_from_outcome(outcome)
        assert card is not None
        assert card.name is not None


class TestAutoConfig:
    """Tests for auto-configuration system."""

    def test_analyse_dataset(self):
        from domains.training.auto_config import analyse_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            analysis = analyse_dataset(f.name)
            assert analysis is not None
            assert hasattr(analysis, "total_chars")

    def test_auto_configure(self):
        from domains.training.auto_config import auto_configure

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            config = auto_configure(f.name)
            assert isinstance(config, dict)
            assert len(config) > 0

    def test_auto_configure_has_recommended_keys(self):
        from domains.training.auto_config import auto_configure

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            config = auto_configure(f.name)
            assert "method" in config or "model" in config or "batch_size" in config


class TestDatasetManifest:
    """Tests for dataset manifest handling."""

    def test_load_manifest_nonexistent(self):
        from domains.training.dataset_manifest import load_manifest
        import pytest

        with pytest.raises(Exception):
            load_manifest("/nonexistent/manifest.json")

    def test_manifest_error_is_valueerror(self):
        from domains.training.dataset_manifest import ManifestError

        assert issubclass(ManifestError, ValueError)


class TestAutoConfigExtended:
    """Extended tests for auto-configuration."""

    def test_auto_configure_produces_valid_config(self):
        from domains.training.auto_config import auto_configure

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            config = auto_configure(f.name)
            if "batch_size" in config:
                assert config["batch_size"] > 0
            if "epochs" in config:
                assert config["epochs"] > 0

    def test_plain_language_verdict(self):
        from domains.training.auto_config import plain_language_verdict

        verdict = plain_language_verdict({"delta": 0.1, "improved": True})
        assert isinstance(verdict, str)
        assert len(verdict) > 0


class TestModelCardExtended:
    """Extended tests for model card."""

    def test_model_card_with_training_info(self):
        from domains.training.model_card import generate_model_card

        card = generate_model_card(
            name="test_model",
            model_type="slnet",
            training_info={
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 0.001,
                "loss": 0.3,
            },
        )
        assert card is not None

    def test_model_card_serializable(self):
        from domains.training.model_card import generate_model_card

        card = generate_model_card(
            name="test_model",
            model_type="slnet",
            training_info={"epochs": 3},
        )
        d = card.to_dict()
        assert isinstance(d, dict)
        assert "name" in d


class TestComprehensiveTrainerWithAutoConfig:
    """Tests for ComprehensiveTrainer with auto-config."""

    def test_trainer_with_auto_config(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "auto_config": True},
            )
            assert result.success

    def test_trainer_with_adaptive_config(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "adaptive": True},
            )
            assert result.success
