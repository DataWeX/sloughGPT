"""
Training Robustness Tests — error recovery, config validation, concurrent safety.

Tests the training system's resilience: invalid configs, missing files,
error recovery, and thread safety.

Usage:
    .venv/bin/python -m pytest tests/test_training_robustness.py -x -v
"""
import tempfile
import threading
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


class TestErrorRecovery:
    """Tests that training recovers gracefully from errors."""

    def test_missing_file_returns_error(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        trainer = ComprehensiveTrainer()
        result = trainer.run_full_cycle(
            data_path="/nonexistent/file.txt",
            config=FAST_CONFIG,
        )
        assert not result.success
        assert result.error is not None

    def test_empty_file_returns_error(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert not result.success

    def test_short_data_returns_error(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("short")
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert not result.success

    def test_result_has_error_message(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        trainer = ComprehensiveTrainer()
        result = trainer.run_full_cycle(
            data_path="/nonexistent/file.txt",
            config=FAST_CONFIG,
        )
        assert result.error is not None
        assert len(result.error) > 0

    def test_result_phases_recorded_on_error(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        trainer = ComprehensiveTrainer()
        result = trainer.run_full_cycle(
            data_path="/nonexistent/file.txt",
            config=FAST_CONFIG,
        )
        assert len(result.phases) > 0

    def test_failed_result_has_duration(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        trainer = ComprehensiveTrainer()
        result = trainer.run_full_cycle(
            data_path="/nonexistent/file.txt",
            config=FAST_CONFIG,
        )
        assert result.total_duration_s >= 0


class TestConfigValidation:
    """Tests that invalid configs are handled properly."""

    def test_invalid_method_uses_default(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "method": "invalid_method_xyz"},
            )
            assert result.success

    def test_zero_epochs_handled(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "epochs": 0},
            )
            assert result.success

    def test_negative_epochs_handled(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "epochs": -1},
            )
            assert result.success

    def test_large_batch_size_handled(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "batch_size": 1024},
            )
            assert result.success

    def test_zero_batch_size_handled(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "batch_size": 0},
            )
            assert result.success

    def test_missing_config_keys_use_defaults(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft"},
            )
            assert result.success

    def test_empty_config_uses_defaults(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config={})
            assert result.success


class TestConcurrentSafety:
    """Tests that training is safe under concurrent access."""

    def test_sequential_runs_are_safe(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            results = []
            for _ in range(3):
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
                results.append(result)
            assert all(r.success for r in results)

    def test_thread_safety(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        results = []
        errors = []

        def run_training():
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                    f.write(DATA_TEXT)
                    f.flush()
                    trainer = ComprehensiveTrainer()
                    result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_training) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_multiple_trainers_same_data(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainers = [ComprehensiveTrainer() for _ in range(3)]
            results = [t.run_full_cycle(data_path=f.name, config=FAST_CONFIG) for t in trainers]
            assert all(r.success for r in results)

    def test_outcome_tracker_thread_safety(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        errors = []

        def record_outcome():
            try:
                for i in range(5):
                    tracker.record(TrainingOutcome(
                        run_id=f"thread_test_{threading.current_thread().name}_{i}",
                        timestamp=time.time(),
                        dataset_size=100,
                        model="test",
                        method="sft",
                        final_loss=1.0,
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_outcome, name=f"t{i}") for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestResultSerialization:
    """Tests that results can be serialized and deserialized."""

    def test_result_to_dict(self):
        from dataclasses import asdict

        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            result_dict = asdict(result)
            assert isinstance(result_dict, dict)
            assert "run_id" in result_dict
            assert "success" in result_dict

    def test_result_summary_string(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            summary = result.summary()
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_phase_results_have_required_fields(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            for phase in result.phases:
                assert hasattr(phase, "phase")
                assert hasattr(phase, "success")
                assert hasattr(phase, "duration_s")
                assert phase.duration_s >= 0
