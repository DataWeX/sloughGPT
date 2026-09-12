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
            model_name="test_model",
            model_type="slnet",
            description="A test model",
        )
        assert card.model_name == "test_model"
        assert card.model_type == "slnet"

    def test_model_card_to_markdown(self):
        from domains.training.model_card import ModelCard

        card = ModelCard(
            model_name="test_model",
            model_type="slnet",
            description="A test model",
        )
        md = card.to_markdown()
        assert isinstance(md, str)
        assert "test_model" in md

    def test_generate_model_card(self):
        from domains.training.model_card import generate_model_card

        card = generate_model_card(
            name="test_model",
            base_model="gpt2",
            description="A test model",
            epochs=3,
            learning_rate=0.001,
        )
        assert card is not None
        assert card.model_name == "test_model"
        assert card.epochs == 3

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
        assert card.model_name is not None

    def test_model_card_to_dict(self):
        from domains.training.model_card import ModelCard

        card = ModelCard(
            model_name="test_model",
            model_type="slnet",
            description="A test model",
        )
        d = card.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "test_model"


class TestAutoConfig:
    """Tests for auto-configuration system."""

    def test_analyse_dataset(self):
        from domains.training.auto_config import analyse_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            analysis = analyse_dataset(f.name)
            assert analysis is not None
            assert analysis.char_count > 0
            assert analysis.word_count > 0

    def test_auto_configure(self):
        from domains.training.auto_config import auto_configure

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            config = auto_configure(
                dataset="test_dataset",
                dataset_path=f.name,
            )
            assert config is not None
            assert config.dataset == "test_dataset"

    def test_dataset_analysis_format(self):
        from domains.training.auto_config import analyse_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            analysis = analyse_dataset(f.name)
            assert analysis.format == "text"

    def test_dataset_analysis_size_category(self):
        from domains.training.auto_config import analyse_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            analysis = analyse_dataset(f.name)
            assert analysis.size_category in ("tiny", "small", "medium", "large")

    def test_plain_language_verdict(self):
        from domains.training.auto_config import plain_language_verdict

        verdict = plain_language_verdict({"verdict": "improved", "perplexity_improvement_pct": 10})
        assert isinstance(verdict, str)
        assert len(verdict) > 0

    def test_plain_language_verdict_degraded(self):
        from domains.training.auto_config import plain_language_verdict

        verdict = plain_language_verdict({"verdict": "degraded", "perplexity_improvement_pct": -5})
        assert isinstance(verdict, str)
        assert len(verdict) > 0


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


class TestModelCardExtended:
    """Extended tests for model card."""

    def test_model_card_with_training_info(self):
        from domains.training.model_card import generate_model_card

        card = generate_model_card(
            name="test_model",
            base_model="gpt2",
            description="A test model",
            epochs=5,
            batch_size=8,
            learning_rate=0.001,
            final_loss=0.3,
        )
        assert card is not None
        assert card.epochs == 5
        assert card.batch_size == 8

    def test_model_card_serializable(self):
        from domains.training.model_card import generate_model_card

        card = generate_model_card(
            name="test_model",
            base_model="gpt2",
            description="A test model",
            epochs=3,
        )
        d = card.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d

    def test_model_card_from_dict(self):
        from domains.training.model_card import ModelCard

        d = {
            "model_name": "test_model",
            "model_type": "slnet",
            "description": "A test model",
        }
        card = ModelCard.from_dict(d)
        assert card.model_name == "test_model"
        assert card.model_type == "slnet"


class TestComprehensiveTrainerWithAutoConfig:
    """Tests for ComprehensiveTrainer with auto-config."""

    def test_trainer_with_explicit_config(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success

    def test_trainer_multiple_runs(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for _ in range(2):
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
                assert result.success
