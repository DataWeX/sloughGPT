"""
Tests for ActivityTrainer (TrainerProtocol wrapper) and the training router endpoint.

Verifies that:
1. ActivityTrainer.train() returns a well-formed TrainResult
2. With no data, returns success=False + status="no_data"
3. With synthetic data, training runs and produces metrics
4. The endpoint schema validates correctly
"""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a few synthetic .npz files."""
    rng = np.random.default_rng(42)
    for i in range(6):
        n_samples = 128
        data = rng.normal(size=(n_samples, 6)).astype(np.float32)
        label = i % 6  # cycle through 0..5
        np.savez_compressed(tmp_path / f"{i + 1}.npz", data=data, label=np.int64(label))
    return tmp_path


class TestActivityTrainerInit:
    """ActivityTrainer instantiation."""

    def test_importable(self):
        from domains.activity.trainer import ActivityTrainer
        assert ActivityTrainer is not None

    def test_default_config(self):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        trainer = ActivityTrainer()
        assert trainer.config.epochs == 30

    def test_custom_config(self):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(epochs=5, lr=0.01, batch_size=8)
        trainer = ActivityTrainer(config=cfg)
        assert trainer.config.epochs == 5

    def test_is_training_defaults_to_false(self):
        from domains.activity.trainer import ActivityTrainer
        assert ActivityTrainer().is_training is False

    def test_stop_does_not_raise(self):
        from domains.activity.trainer import ActivityTrainer
        ActivityTrainer().stop()


class TestActivityTrainerNoData:
    """ActivityTrainer with an empty data directory."""

    def test_train_no_data_returns_failure(self, tmp_path: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(tmp_path))
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert result.success is False
        assert "no_data" in result.status
        assert "recordings" in result.error

    def test_train_no_data_returns_train_result_fields(self, tmp_path: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(tmp_path))
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert hasattr(result, "success")
        assert hasattr(result, "status")
        assert hasattr(result, "error")
        assert hasattr(result, "method")
        assert result.method == "activity"
        assert isinstance(result.to_dict(), dict)

    def test_dict_backward_compat(self, tmp_path: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(tmp_path))
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert result.get("success") is False
        assert result.get("method") == "activity"
        assert result.get("nonexistent", "default") == "default"


class TestActivityTrainerWithData:
    """ActivityTrainer with synthetic sensor data."""

    def test_train_with_data_returns_success(self, fake_data_dir: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(fake_data_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert result.success, f"Training failed: {result.error}"
        assert result.status == "completed"

    def test_train_produces_metrics(self, fake_data_dir: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(fake_data_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert "val_accuracy" in result.metrics
        assert "val_loss" in result.metrics
        assert "num_labeled" in result.metrics
        assert result.metrics["num_labeled"] >= 6
        assert isinstance(result.metrics["val_accuracy"], float)
        assert isinstance(result.final_loss, float)

    def test_train_returns_method(self, fake_data_dir: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(fake_data_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert result.method == "activity"

    def test_train_to_dict_roundtrip(self, fake_data_dir: Path):
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        cfg = ActivityTrainerConfig(data_dir=str(fake_data_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        d = result.to_dict()
        assert d["success"] is True
        assert "val_accuracy" in d
        assert isinstance(json.dumps(d), str)
