"""
Training Module Coverage Tests.

Tests for training modules that don't have dedicated test files.

Usage:
    .venv/bin/python -m pytest tests/test_training_module_coverage.py -x -v
"""

import tempfile
import time

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50


class TestExportModule:
    """Tests for domain.training._internal.export.py."""

    def test_export_config_creation(self):
        from domain.training._internal.export import ExportConfig

        config = ExportConfig(format="gguf")
        assert config.format == "gguf"

    def test_list_export_formats(self):
        from domain.training._internal.export import list_export_formats

        formats = list_export_formats()
        assert isinstance(formats, dict)
        assert len(formats) > 0

    def test_model_metadata_creation(self):
        from domain.training._internal.export import ModelMetadata

        meta = ModelMetadata(name="test_model", version="1.0")
        assert meta.name == "test_model"
        assert meta.version == "1.0"


class TestPresetsModule:
    """Tests for domain.training._internal.presets.py."""

    def test_presets_exist(self):
        from domain.training._internal.presets import PRESETS

        assert isinstance(PRESETS, dict)
        assert len(PRESETS) > 0

    def test_list_presets(self):
        from domain.training._internal.presets import list_presets

        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_apply_preset(self):
        from domain.training._internal.presets import PRESETS, apply_preset

        preset_names = list(PRESETS.keys())
        if preset_names:
            config = apply_preset(preset_names[0])
            assert isinstance(config, dict)


class TestOutcomeTrackerModule:
    """Tests for domain.training._internal.outcome_tracker.py."""

    def test_outcome_creation(self):
        from domain.training._internal.outcome_tracker import TrainingOutcome

        outcome = TrainingOutcome(
            run_id="test_run",
            timestamp=time.time(),
            dataset_size=1000,
            model="gpt2",
            method="sft",
            final_loss=0.5,
        )
        assert outcome.run_id == "test_run"
        assert outcome.final_loss == 0.5

    def test_outcome_to_dict(self):
        from domain.training._internal.outcome_tracker import TrainingOutcome

        outcome = TrainingOutcome(
            run_id="test_run",
            timestamp=time.time(),
            dataset_size=1000,
            model="gpt2",
            method="sft",
            final_loss=0.5,
        )
        d = outcome.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "test_run"


class TestQualityScorerModule:
    """Tests for domain.training._internal.quality_scorer.py."""

    def test_score_pair(self):
        from domain.training._internal.quality_scorer import score_pair

        score = score_pair("Hello world", "This is a test")
        assert isinstance(score, float)
        assert score >= 0

    def test_score_text_chunk(self):
        from domain.training._internal.quality_scorer import score_text_chunk

        score = score_text_chunk("This is a test paragraph with some content.")
        assert isinstance(score, float)
        assert score >= 0


class TestAdaptiveConfigModule:
    """Tests for domain.training._internal.adaptive_config.py."""

    def test_adaptive_config_engine_exists(self):
        from domain.training._internal.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        assert engine is not None


class TestCheckpointsModule:
    """Tests for domain.training._internal.checkpoints.py."""

    def test_find_checkpoint_nonexistent(self):
        from domain.training._internal.checkpoints import find_checkpoint

        result = find_checkpoint("nonexistent_model")
        assert result is None


class TestDatasetModule:
    """Tests for domain.training._internal.dataset.py."""

    def test_dataset_creation(self):
        from domain.training._internal.dataset import TrainingDataset

        dataset = TrainingDataset(source_text=DATA_TEXT)
        assert dataset is not None
        assert len(dataset.chunks) > 0


class TestPairExtractorModule:
    """Tests for domain.training._internal.pair_extractor.py."""

    def test_extract_pairs_from_sessions(self):
        from domain.training._internal.pair_extractor import extract_pairs_from_sessions

        pairs = extract_pairs_from_sessions([])
        assert isinstance(pairs, list)


class TestPerformanceModule:
    """Tests for domain.training._internal.performance.py."""

    def test_performance_monitor_exists(self):
        from domain.training._internal.performance import PerformanceMonitor

        monitor = PerformanceMonitor()
        assert monitor is not None


class TestLRSchedulersModule:
    """Tests for domain.training._internal.lr_schedulers.py."""

    def test_scheduler_classes_importable(self):
        from domain.training._internal.lr_schedulers import (
            LinearWarmupScheduler,
            PolynomialDecayScheduler,
            SchedulerConfig,
            WarmupCosineScheduler,
        )

        assert SchedulerConfig is not None
        assert WarmupCosineScheduler is not None
        assert PolynomialDecayScheduler is not None
        assert LinearWarmupScheduler is not None


class TestEWCModule:
    """Tests for domain.training._internal.ewc.py."""

    def test_ewc_classes_importable(self):
        from domain.training._internal.ewc import (
            DiagonalFisherEstimator,
            EwcContinualLearner,
            EWCParameters,
            TaskSnapshot,
        )

        assert EWCParameters is not None
        assert TaskSnapshot is not None
        assert DiagonalFisherEstimator is not None
        assert EwcContinualLearner is not None


class TestComprehensiveTrainerExtended:
    """Extended tests for ComprehensiveTrainer."""

    def test_trainer_phases(self):
        from domain.training._internal.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={
                    "method": "sft",
                    "data_quality_threshold": 0.0,
                    "epochs": 1,
                    "batch_size": 8,
                    "block_size": 32,
                    "max_steps": 2,
                    "n_embed": 32,
                    "n_layer": 2,
                    "n_head": 2,
                },
            )
            assert result.success
            assert result.total_duration_s > 0

    def test_trainer_result_has_phases(self):
        from domain.training._internal.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={
                    "method": "sft",
                    "data_quality_threshold": 0.0,
                    "epochs": 1,
                    "batch_size": 8,
                    "block_size": 32,
                    "max_steps": 2,
                    "n_embed": 32,
                    "n_layer": 2,
                    "n_head": 2,
                },
            )
            assert result.success
            assert hasattr(result, "performance")
            assert isinstance(result.performance, dict)
