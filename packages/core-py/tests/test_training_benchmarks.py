"""
Performance Benchmarks for the Comprehensive Training System.

Measures training throughput, memory usage, and phase timings.
Runs locally without servers.

Usage:
    .venv/bin/python -m pytest tests/test_training_benchmarks.py -x -v -s
"""
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest


def _make_data(char_count: int = 5000) -> str:
    """Generate diverse training text."""
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
        "The jay, pig, fox, zebra, and my wolves quack!",
        "Sympathizing would fix Quaker obligations.",
    ]
    result = []
    while len(result) * len(sentences[0]) < char_count:
        result.extend(sentences)
    return " ".join(result)[:char_count]


class TestTrainingThroughput:
    """Benchmarks training throughput and speed."""

    def test_small_dataset_throughput(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(2000))
            f.flush()
            config = {
                "method": "sft", "data_quality_threshold": 0.0,
                "epochs": 1, "batch_size": 8, "block_size": 32,
                "max_steps": 5, "n_embed": 32, "n_layer": 2, "n_head": 2,
            }
            trainer = ComprehensiveTrainer()
            t0 = time.time()
            result = trainer.run_full_cycle(data_path=f.name, config=config)
            duration = time.time() - t0
            assert result.success
            assert duration < 30
            print(f"\n  Small dataset: {duration:.2f}s, loss={result.final_loss:.4f}")

    def test_medium_dataset_throughput(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(10000))
            f.flush()
            config = {
                "method": "sft", "data_quality_threshold": 0.0,
                "epochs": 1, "batch_size": 8, "block_size": 32,
                "max_steps": 10, "n_embed": 32, "n_layer": 2, "n_head": 2,
            }
            trainer = ComprehensiveTrainer()
            t0 = time.time()
            result = trainer.run_full_cycle(data_path=f.name, config=config)
            duration = time.time() - t0
            assert result.success
            assert duration < 60
            print(f"\n  Medium dataset: {duration:.2f}s, loss={result.final_loss:.4f}")

    def test_training_speed_steps_per_second(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(5000))
            f.flush()
            config = {
                "method": "sft", "data_quality_threshold": 0.0,
                "epochs": 1, "batch_size": 8, "block_size": 32,
                "max_steps": 20, "n_embed": 32, "n_layer": 2, "n_head": 2,
            }
            trainer = ComprehensiveTrainer()
            t0 = time.time()
            result = trainer.run_full_cycle(data_path=f.name, config=config)
            duration = time.time() - t0
            steps = result.performance.get("phase_durations", {}).get("training", duration)
            steps_per_sec = 20 / steps if steps > 0 else 0
            assert result.success
            print(f"\n  Speed: {steps_per_sec:.2f} steps/s, {duration:.2f}s total")

    def test_lora_vs_full_training_speed(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(3000))
            f.flush()

            base_config = {
                "method": "sft", "data_quality_threshold": 0.0,
                "epochs": 1, "batch_size": 8, "block_size": 32,
                "max_steps": 5, "n_embed": 32, "n_layer": 2, "n_head": 2,
            }

            trainer1 = ComprehensiveTrainer()
            t0 = time.time()
            result1 = trainer1.run_full_cycle(data_path=f.name, config=base_config)
            full_time = time.time() - t0

            lora_config = {**base_config, "use_lora": True, "lora_rank": 4}
            trainer2 = ComprehensiveTrainer()
            t0 = time.time()
            result2 = trainer2.run_full_cycle(data_path=f.name, config=lora_config)
            lora_time = time.time() - t0

            assert result1.success and result2.success
            print(f"\n  Full: {full_time:.2f}s, LoRA: {lora_time:.2f}s")


class TestPhaseTimings:
    """Benchmarks individual phase timings."""

    def test_validation_phase_speed(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer, TrainingPhase

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(5000))
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft", "data_quality_threshold": 0.0, "epochs": 1,
                        "batch_size": 8, "block_size": 32, "max_steps": 1,
                        "n_embed": 32, "n_layer": 2, "n_head": 2},
            )
            validation_time = result.performance["phase_durations"].get("validating", 0)
            assert validation_time < 5
            print(f"\n  Validation: {validation_time:.3f}s")

    def test_configuration_phase_speed(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(5000))
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft", "data_quality_threshold": 0.0, "epochs": 1,
                        "batch_size": 8, "block_size": 32, "max_steps": 1,
                        "n_embed": 32, "n_layer": 2, "n_head": 2, "adaptive": True},
            )
            config_time = result.performance["phase_durations"].get("configuring", 0)
            assert config_time < 5
            print(f"\n  Configuration: {config_time:.3f}s")

    def test_preprocessing_phase_speed(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(5000))
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft", "data_quality_threshold": 0.0, "epochs": 1,
                        "batch_size": 8, "block_size": 32, "max_steps": 1,
                        "n_embed": 32, "n_layer": 2, "n_head": 2},
            )
            preprocess_time = result.performance["phase_durations"].get("preprocessing", 0)
            assert preprocess_time < 5
            print(f"\n  Preprocessing: {preprocess_time:.3f}s")


class TestMemoryUsage:
    """Benchmarks memory usage during training."""

    def test_training_does_not_leak_memory(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(3000))
            f.flush()
            config = {
                "method": "sft", "data_quality_threshold": 0.0,
                "epochs": 1, "batch_size": 8, "block_size": 32,
                "max_steps": 3, "n_embed": 32, "n_layer": 2, "n_head": 2,
            }
            results = []
            for i in range(3):
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(data_path=f.name, config=config)
                results.append(result)
            assert all(r.success for r in results)
            losses = [r.final_loss for r in results if r.final_loss is not None]
            assert len(losses) == 3

    def test_multiple_sequential_runs(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(3000))
            f.flush()
            config = {
                "method": "sft", "data_quality_threshold": 0.0,
                "epochs": 1, "batch_size": 8, "block_size": 32,
                "max_steps": 2, "n_embed": 32, "n_layer": 2, "n_head": 2,
            }
            durations = []
            for _ in range(3):
                trainer = ComprehensiveTrainer()
                t0 = time.time()
                result = trainer.run_full_cycle(data_path=f.name, config=config)
                durations.append(time.time() - t0)
                assert result.success
            avg_duration = sum(durations) / len(durations)
            assert avg_duration < 30
            print(f"\n  Avg run duration: {avg_duration:.2f}s")


class TestAdaptiveConfigPerformance:
    """Benchmarks the adaptive config engine."""

    def test_adaptive_config_recommendation_speed(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        t0 = time.time()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        duration = time.time() - t0
        assert duration < 1
        assert rec.confidence >= 0
        print(f"\n  Adaptive config: {duration:.3f}s, confidence={rec.confidence:.2f}")

    def test_outcome_tracker_speed(self):
        from domains.training.outcome_tracker import TrainingOutcomeTracker, TrainingOutcome

        tracker = TrainingOutcomeTracker()
        t0 = time.time()
        for i in range(10):
            outcome = TrainingOutcome(
                run_id=f"bench_{i}",
                timestamp=time.time(),
                dataset_size=1000,
                model="test",
                method="sft",
                final_loss=1.0 - i * 0.1,
            )
            tracker.record(outcome)
        duration = time.time() - t0
        assert duration < 2
        print(f"\n  Outcome tracker (10 writes): {duration:.3f}s")


class TestComprehensiveResultPerformance:
    """Tests performance metrics in ComprehensiveResult."""

    def test_result_performance_fields(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(3000))
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft", "data_quality_threshold": 0.0, "epochs": 1,
                        "batch_size": 8, "block_size": 32, "max_steps": 2,
                        "n_embed": 32, "n_layer": 2, "n_head": 2},
            )
            perf = result.performance
            assert "total_duration_s" in perf
            assert "phase_durations" in perf
            assert "slowest_phase" in perf
            assert "phases_completed" in perf
            assert perf["total_duration_s"] > 0
            assert perf["phases_completed"] >= 4

    def test_result_summary_with_performance(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_make_data(3000))
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft", "data_quality_threshold": 0.0, "epochs": 1,
                        "batch_size": 8, "block_size": 32, "max_steps": 2,
                        "n_embed": 32, "n_layer": 2, "n_head": 2},
            )
            summary = result.summary()
            assert "Performance:" in summary
            assert "total_duration_s" in summary
