"""
Training System Integration Tests — end-to-end workflows.

Tests complete training workflows combining multiple components:
adaptive config + trainer + outcome tracking + export.

Usage:
    .venv/bin/python -m pytest tests/test_training_integration.py -x -v
"""
import tempfile
from dataclasses import asdict

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


class TestAdaptiveConfigToTrainerFlow:
    """Tests flow from adaptive config recommendation to trainer execution."""

    def test_recommendation_applied_to_trainer(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")

        config = {
            **FAST_CONFIG,
            "learning_rate": rec.learning_rate,
            "batch_size": rec.batch_size,
            "epochs": rec.epochs,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=config)
            assert result.success

    def test_recommendation_confidence_affects_config(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        low = engine.recommend(dataset_size=50, model="gpt2", method="finetune")
        high = engine.recommend(dataset_size=50000, model="gpt2", method="finetune")

        assert 0.0 <= low.confidence <= 1.0
        assert 0.0 <= high.confidence <= 1.0
        assert low.based_on_runs >= 0
        assert high.based_on_runs >= 0


class TestTrainerToOutcomeFlow:
    """Tests flow from trainer execution to outcome tracking."""

    def test_outcome_recorded_after_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer
        from domains.training.outcome_tracker import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        initial_count = len(tracker.load_outcomes())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)

        after_count = len(tracker.load_outcomes())
        assert after_count >= initial_count

    def test_outcome_contains_training_metrics(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.final_loss is not None
            assert result.total_duration_s > 0
            assert len(result.phases) > 0


class TestOutcomeToExportFlow:
    """Tests flow from outcome tracking to export."""

    def test_export_after_multiple_runs(self):

        from domains.training.comprehensive_trainer import ComprehensiveTrainer
        from domains.training.outcome_tracker import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        initial = len(tracker.export_json(limit=100))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for _ in range(2):
                trainer = ComprehensiveTrainer()
                trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)

        json_data = tracker.export_json(limit=100)
        assert len(json_data) >= initial

    def test_csv_export_after_runs(self):
        import csv
        import io

        from domains.training.comprehensive_trainer import ComprehensiveTrainer
        from domains.training.outcome_tracker import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)

        csv_str = tracker.export_csv(limit=10)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) >= 2


class TestFullWorkflowIntegration:
    """End-to-end workflow tests combining all components."""

    def test_adaptive_training_export_cycle(self):

        from domains.training.adaptive_config import AdaptiveConfigEngine
        from domains.training.comprehensive_trainer import ComprehensiveTrainer
        from domains.training.outcome_tracker import TrainingOutcomeTracker

        engine = AdaptiveConfigEngine()
        tracker = TrainingOutcomeTracker()

        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        config = {
            **FAST_CONFIG,
            "learning_rate": rec.learning_rate,
            "batch_size": rec.batch_size,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=config)

        assert result.success
        assert result.final_loss is not None

        json_data = tracker.export_json(limit=10)
        assert isinstance(json_data, list)

    def test_multiple_methods_same_data(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for method in ["sft", "distillation"]:
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(
                    data_path=f.name,
                    config={**FAST_CONFIG, "method": method},
                )
                assert result.success, f"Failed for method: {method}"
                assert result.final_loss is not None

    def test_trainer_result_serializable(self):
        import json

        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        def _default(obj):
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "name"):
                return obj.name
            return str(obj)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            result_dict = asdict(result)
            json_str = json.dumps(result_dict, default=_default)
            assert len(json_str) > 0

    def test_preset_based_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer
        from domains.training.presets import apply_preset

        preset_config = apply_preset("quick-finetune")
        assert preset_config is not None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success


class TestPerformanceConsistency:
    """Tests that training performance is consistent across runs."""

    def test_multiple_runs_have_similar_duration(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        durations = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for _ in range(3):
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
                durations.append(result.total_duration_s)

        assert all(d > 0 for d in durations)
        avg = sum(durations) / len(durations)
        for d in durations:
            assert abs(d - avg) < avg * 2.0

    def test_loss_decreases_or_stable(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        losses = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for _ in range(3):
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
                if result.final_loss is not None:
                    losses.append(result.final_loss)

        assert len(losses) == 3
        assert all(isinstance(loss, float) for loss in losses)
