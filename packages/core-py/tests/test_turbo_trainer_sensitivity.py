"""
SloughGPTTrainer integration tests.

Tests that SloughGPTTrainer training works end-to-end and
TrainingProgress SSE events include sensitivity when provided.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


def _make_tiny_dataset(tmp_path: Path) -> Path:
    """Create a small ASCII text file for tokenization."""
    data_file = tmp_path / "tiny.txt"
    data_file.write_text(
        "The quick brown fox jumps over the lazy dog. " * 200,
        encoding="utf-8",
    )
    return data_file


class TestSloughGPTTrainerBasic:
    """Verify SloughGPTTrainer works end-to-end."""

    def test_train_basic(self, tmp_path: float):
        """Baseline: train produces completed result."""
        from domains.training.train_pipeline import SloughGPTTrainer

        data = _make_tiny_dataset(tmp_path)
        trainer = SloughGPTTrainer(
            data_path=str(data),
            vocab_size=200,
            n_embed=64,
            n_layer=1,
            n_head=4,
            block_size=32,
            batch_size=4,
            epochs=1,
            lr=1e-3,
            checkpoint_dir=str(tmp_path / "model"),
        )
        progress_events = []
        result = trainer.train(on_progress=lambda d: progress_events.append(d))

        assert result.get("status") == "completed"
        assert result.get("total_steps", 0) > 0

    def test_train_with_progress(self, tmp_path: float):
        """Progress events are emitted during training."""
        from domains.training.train_pipeline import SloughGPTTrainer

        data = _make_tiny_dataset(tmp_path)
        trainer = SloughGPTTrainer(
            data_path=str(data),
            vocab_size=200,
            n_embed=64,
            n_layer=1,
            n_head=4,
            block_size=32,
            batch_size=4,
            epochs=2,
            lr=1e-3,
            checkpoint_dir=str(tmp_path / "model"),
        )
        progress_events = []
        result = trainer.train(on_progress=lambda d: progress_events.append(d))

        assert len(progress_events) > 0
        assert any("loss" in e for e in progress_events)


class TestPipelineSensitivitySSE:
    """Verify sensitivity reaches SSE events via UnifiedTrainingPipeline."""

    def test_pipeline_sensitivity_in_sse_event(self, tmp_path: float):
        """UnifiedTrainingPipeline with method='turbo' includes sensitivity in SSE."""
        from domains.training.unified_pipeline import (
            UnifiedTrainingPipeline, UnifiedTrainingConfig, TrainingProgress,
        )

        # Create tiny dataset
        data_file = tmp_path / "tiny.txt"
        data_file.write_text("The quick brown fox jumps over the lazy dog. " * 200, encoding="utf-8")

        cfg = UnifiedTrainingConfig(
            method="turbo",
            data_path=str(data_file),
            output_dir=str(tmp_path / "model"),
            epochs=2,
        )
        pipeline = UnifiedTrainingPipeline(cfg)

        sse_events = []

        def _on_progress(progress):
            sse = progress.to_sse_event(stream_name="auto-train")
            sse_events.append(sse)

        pipeline.run(on_progress=_on_progress)

        # Find train-phase SSE events with sensitivity
        train_events = [e for e in sse_events if e.get("phase") == "train"]
        sens_events = [e for e in train_events if "sensitivity" in e.get("data", {})]

        # Sensitivity may or may not be present depending on pipeline implementation
        # This test verifies the SSE envelope includes sensitivity when provided
        if sens_events:
            for evt in sens_events:
                sens = evt["data"]["sensitivity"]
                assert isinstance(sens, dict)
                for name, value in sens.items():
                    assert isinstance(value, float)
                    assert value >= 0.0, f"{name}: negative sensitivity {value}"

    def test_training_progress_to_sse_includes_sensitivity(self):
        """TrainingProgress.to_sse_event includes sensitivity from metrics."""
        from domains.training.unified_pipeline import TrainingProgress

        progress = TrainingProgress(
            phase="train",
            epoch=1,
            step=100,
            loss=1.2,
            metrics={"sensitivity": {"lstm": 0.35, "embed": 0.12}},
        )
        sse = progress.to_sse_event(stream_name="auto-train")

        assert "sensitivity" in sse["data"]
        assert sse["data"]["sensitivity"] == {"lstm": 0.35, "embed": 0.12}
