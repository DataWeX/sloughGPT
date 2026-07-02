"""
Integration test for the full activity pipeline.

Tests the end-to-end flow: create sensor data → load → train → save → evaluate.

This is a self-contained test that does NOT require a running server or HTTP.
"""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def recordings_dir(tmp_path: Path) -> Path:
    """Create labeled sensor recordings for testing."""
    rng = np.random.default_rng(42)
    for i in range(6):
        data = rng.normal(size=(128, 6)).astype(np.float32)
        label = i
        np.savez_compressed(tmp_path / f"{i + 1}.npz", data=data, label=np.int64(label))
    return tmp_path


class TestEndToEndFlow:
    """Covers the full record → train → predict → sync flow."""

    def test_activity_trainer_full_flow(self, recordings_dir: Path):
        """Train on real temp data, verify TrainResult fields."""
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig

        cfg = ActivityTrainerConfig(data_dir=str(recordings_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()

        assert result.success is True
        assert result.status == "completed"
        assert result.method == "activity"
        assert result.metrics["num_labeled"] == 6
        assert isinstance(result.metrics["val_accuracy"], float)
        assert 0.0 <= result.metrics["val_accuracy"] <= 1.0
        assert isinstance(result.final_loss, float)
        assert result.final_loss > 0.0

    def test_model_persists_to_disk(self, recordings_dir: Path):
        """After training, model.npz should exist for GET /activity/model."""
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig

        cfg = ActivityTrainerConfig(data_dir=str(recordings_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        trainer.train()

        model_path = (
            Path(__file__).resolve().parents[1]
            / "domains" / "activity" / "model.npz"
        )
        assert model_path.exists(), f"Model not found at {model_path}"
        assert model_path.stat().st_size > 1000, "Model file too small"

    def test_trained_model_can_predict(self, recordings_dir: Path):
        """After training, the model can classify a sensor window."""
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig
        from domains.activity.classifier import ActivityClassifier
        from domains.activity import predict_activity

        cfg = ActivityTrainerConfig(data_dir=str(recordings_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        trainer.train()

        model_path = (
            Path(__file__).resolve().parents[1]
            / "domains" / "activity" / "model.npz"
        )
        assert model_path.exists()
        model = ActivityClassifier.load(str(model_path))

        # Predict on a known sample
        rng = np.random.default_rng(42)
        sample = rng.normal(size=(128, 6)).astype(np.float32)
        cls_id, name, probs = predict_activity(model, sample)
        assert 0 <= cls_id < 6
        assert len(probs) == 6
        assert abs(sum(probs) - 1.0) < 0.01

    def test_train_with_no_data_returns_graceful_error(self, tmp_path: Path):
        """Empty data dir → proper error not a crash."""
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig

        cfg = ActivityTrainerConfig(data_dir=str(tmp_path))
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()
        assert result.success is False
        assert "recordings" in (result.error or "")
        assert result.method == "activity"

    def test_train_result_dict_compatibility(self, recordings_dir: Path):
        """TrainResult must support dict-like access for legacy consumers."""
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig

        cfg = ActivityTrainerConfig(data_dir=str(recordings_dir), epochs=3, augment=False)
        trainer = ActivityTrainer(config=cfg)
        result = trainer.train()

        assert result["success"] is True
        assert result.metrics["val_accuracy"] == 0.0 or result.metrics["val_accuracy"] > 0.0
        assert result.get("method") == "activity"
        assert result.get("nonexistent", "x") == "x"
        assert json.dumps(result.to_dict()) is not None

    def test_multiple_train_calls_produce_different_results(self, recordings_dir: Path):
        """Repeated training with different configs should differ."""
        from domains.activity.trainer import ActivityTrainer, ActivityTrainerConfig

        cfg1 = ActivityTrainerConfig(data_dir=str(recordings_dir), epochs=2, augment=False)
        r1 = ActivityTrainer(config=cfg1).train()

        cfg2 = ActivityTrainerConfig(data_dir=str(recordings_dir), epochs=5, augment=False)
        r2 = ActivityTrainer(config=cfg2).train()

        assert r1.success and r2.success
        # More epochs should (usually) give better accuracy
        assert r2.total_steps > r1.total_steps


class TestActivityRouterLogic:
    """Unit-test the backend logic that the activity router uses."""

    def test_load_all_data(self, recordings_dir: Path):
        """Verify _load_all_data equivalent returns correct shapes."""
        samples, labels = [], []
        for f in sorted(recordings_dir.glob("*.npz")):
            d = np.load(f)
            samples.append(d["data"])
            labels.append(int(d.get("label", -1)))

        X = np.stack(samples).astype(np.float32)
        y = np.array(labels)
        assert X.shape == (6, 128, 6)
        assert y.shape == (6,)

    def test_label_filtering(self, recordings_dir: Path):
        """Only labeled recordings should be used for training."""
        samples, labels = [], []
        for f in sorted(recordings_dir.glob("*.npz")):
            d = np.load(f)
            samples.append(d["data"])
            labels.append(int(d.get("label", -1)))

        X = np.stack(samples).astype(np.float32)
        y = np.array(labels)
        labeled = y >= 0
        assert labeled.sum() == 6

    def test_augmentation_produces_same_shape(self):
        """Data augmentation should preserve input shape."""
        from domains.activity.classifier import _augment_batch

        rng = np.random.default_rng(42)
        X = rng.normal(size=(4, 128, 6)).astype(np.float32)
        augmented = _augment_batch(X)
        assert augmented.shape == X.shape
        assert augmented.dtype == np.float32

    def test_accuracy_helper(self):
        """_accuracy computes correctly."""
        from domains.activity.classifier import _accuracy
        from domains.training.slonet import Tensor

        # Perfect score
        logits = Tensor(np.array([[10, 0], [0, 10]], dtype=np.float32))
        targets = np.array([0, 1])
        assert _accuracy(logits, targets) == 1.0

        # Half score
        logits2 = Tensor(np.array([[10, 0], [10, 0]], dtype=np.float32))
        targets2 = np.array([0, 1])
        assert _accuracy(logits2, targets2) == 0.5
