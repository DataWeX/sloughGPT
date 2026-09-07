"""Tests for training.training_advisor — training advisor."""

from __future__ import annotations

import pytest
from domains.training.training_advisor import (
    TrainingRecommendation,
    recommend_training_config,
)


# ── TrainingRecommendation ─────────────────────────────────────────────────


class TestTrainingRecommendation:

    def test_init(self):
        rec = TrainingRecommendation(
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            warmup_steps=50,
            early_stopping_patience=5,
            reason="test",
            confidence=0.8,
        )
        assert rec.learning_rate == 0.001
        assert rec.confidence == 0.8


# ── recommend_training_config ──────────────────────────────────────────────


class TestRecommendTrainingConfig:

    def test_small_dataset(self):
        rec = recommend_training_config(dataset_size=50)
        assert rec.learning_rate < 3e-4
        assert rec.batch_size <= 32

    def test_medium_dataset(self):
        rec = recommend_training_config(dataset_size=5000)
        assert rec.learning_rate > 0
        assert rec.batch_size > 0

    def test_large_dataset(self):
        rec = recommend_training_config(dataset_size=50000)
        assert rec.learning_rate > 0

    def test_very_large_dataset(self):
        rec = recommend_training_config(dataset_size=500000)
        assert rec.learning_rate > 0

    def test_finetune_method(self):
        rec = recommend_training_config(dataset_size=1000, method="finetune")
        assert rec.learning_rate > 0

    def test_native_method(self):
        rec = recommend_training_config(dataset_size=1000, method="native")
        assert rec.learning_rate > 0

    def test_with_model_params(self):
        rec = recommend_training_config(dataset_size=1000, model_params=500000)
        assert rec.learning_rate > 0

    def test_with_quality(self):
        rec = recommend_training_config(dataset_size=1000, avg_quality=4.0)
        assert rec.learning_rate > 0

    def test_confidence_range(self):
        rec = recommend_training_config(dataset_size=1000)
        assert 0 <= rec.confidence <= 1
