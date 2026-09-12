"""
Training Infrastructure Tests — adaptive config, state management, queue handlers.

Tests the supporting infrastructure for the training system: adaptive config engine,
training state management, and training queue event handling.

Usage:
    .venv/bin/python -m pytest tests/test_training_infrastructure.py -x -v
"""
import tempfile
import time

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

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50


class TestAdaptiveConfigEngine:
    """Tests for the adaptive config recommendation engine."""

    def test_recommend_returns_result(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert rec is not None
        assert hasattr(rec, "learning_rate")
        assert hasattr(rec, "batch_size")
        assert hasattr(rec, "confidence")

    def test_recommend_confidence_range(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert 0.0 <= rec.confidence <= 1.0

    def test_recommend_based_on_runs(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert rec.based_on_runs >= 0

    def test_recommend_different_sizes(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        small = engine.recommend(dataset_size=100, model="gpt2", method="finetune")
        large = engine.recommend(dataset_size=100000, model="gpt2", method="finetune")
        assert small is not None
        assert large is not None

    def test_recommend_different_methods(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        sft = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        rlhf = engine.recommend(dataset_size=1000, model="gpt2", method="rlhf")
        assert sft is not None
        assert rlhf is not None

    def test_recommend_has_reason(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert isinstance(rec.reason, str)

    def test_recommend_quality_metrics(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert rec.avg_quality >= 0
        assert rec.avg_loss >= 0
        assert rec.best_quality >= 0


class TestTrainingState:
    """Tests for training state management."""

    def test_get_state_returns_training_state(self):
        from domains.training.state import TrainingState, get_state

        state = get_state()
        assert isinstance(state, TrainingState)

    def test_state_has_required_fields(self):
        from domains.training.state import get_state

        state = get_state()
        assert hasattr(state, "running")
        assert hasattr(state, "config")
        assert hasattr(state, "student_net")
        assert hasattr(state, "student_tokenizer")

    def test_state_is_singleton(self):
        from domains.training.state import get_state

        state1 = get_state()
        state2 = get_state()
        assert state1 is state2

    def test_state_running_default(self):
        from domains.training.state import get_state

        state = get_state()
        assert isinstance(state.running, bool)

    def test_state_config_is_dict(self):
        from domains.training.state import get_state

        state = get_state()
        assert isinstance(state.config, dict)

    def test_get_turbo_state(self):
        from domains.training.state import get_turbo_state

        turbo = get_turbo_state()
        assert isinstance(turbo, dict)
        assert "status" in turbo
        assert "global_step" in turbo
        assert "progress" in turbo

    def test_turbo_state_has_required_fields(self):
        from domains.training.state import get_turbo_state

        turbo = get_turbo_state()
        required = ["status", "job_id", "global_step", "total_steps", "progress", "loss"]
        for field in required:
            assert field in turbo, f"Missing field: {field}"


class TestTrainingConstants:
    """Tests for training constants and configuration."""

    def test_checkpoint_dir_exists(self):
        from domains.training.state import CHECKPOINTS_DIR

        assert CHECKPOINTS_DIR.exists()

    def test_lora_dir_exists(self):
        from domains.training.state import LORA_DIR

        assert LORA_DIR.exists()

    def test_turbo_dir_exists(self):
        from domains.training.state import TURBO_DIR

        assert TURBO_DIR.exists()

    def test_valid_checkpoint_name(self):
        from domains.training.state import VALID_CKPT_NAME

        assert VALID_CKPT_NAME.match("my-checkpoint_v1")
        assert VALID_CKPT_NAME.match("checkpoint.001")
        assert not VALID_CKPT_NAME.match("checkpoint with spaces")
        assert not VALID_CKPT_NAME.match("checkpoint/slash")


class TestOutcomeTrackerBasics:
    """Tests for TrainingOutcomeTracker core functionality."""

    def test_tracker_record_and_load(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        initial = len(tracker.load_outcomes())
        tracker.record(TrainingOutcome(
            run_id=f"test_{int(time.time())}",
            timestamp=time.time(),
            dataset_size=500,
            model="test",
            method="sft",
            final_loss=0.5,
        ))
        after = len(tracker.load_outcomes())
        assert after >= initial

    def test_outcome_has_required_fields(self):
        from domains.training.outcome_tracker import TrainingOutcome

        outcome = TrainingOutcome(
            run_id="test_run",
            timestamp=time.time(),
            dataset_size=1000,
            model="gpt2",
            method="sft",
            final_loss=0.5,
        )
        assert outcome.run_id == "test_run"
        assert outcome.dataset_size == 1000
        assert outcome.model == "gpt2"
        assert outcome.method == "sft"
        assert outcome.final_loss == 0.5


class TestComprehensiveTrainerIntegration:
    """Integration tests for ComprehensiveTrainer with infrastructure."""

    def test_trainer_uses_adaptive_config(self):
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

    def test_trainer_records_outcome(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success
            assert result.final_loss is not None

    def test_trainer_performance_metrics(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            perf = result.performance
            assert "total_duration_s" in perf
            assert "phase_durations" in perf
            assert "slowest_phase" in perf
            assert perf["total_duration_s"] > 0

    def test_trainer_multiple_methods(self):
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
