"""Tests for training.training_advisor — TrainingRecommendation, recommend_training_config, get_training_tips."""

from __future__ import annotations

import pytest

from domains.training.training_advisor import (
    TrainingRecommendation, recommend_training_config, get_training_tips,
)


# ── TrainingRecommendation ─────────────────────────────────────────────────


class TestTrainingRecommendation:

    def test_init(self):
        r = TrainingRecommendation(
            learning_rate=0.001, batch_size=8, epochs=10,
            warmup_steps=50, early_stopping_patience=3,
            reason="test", confidence=0.8,
        )
        assert r.learning_rate == 0.001
        assert r.confidence == 0.8


# ── recommend_training_config ──────────────────────────────────────────────


class TestRecommendTrainingConfig:

    def test_very_small_dataset(self):
        r = recommend_training_config(dataset_size=50)
        assert r.confidence == 0.6
        assert "Very small" in r.reason

    def test_small_dataset(self):
        r = recommend_training_config(dataset_size=500)
        assert r.confidence == 0.7
        assert "Small" in r.reason

    def test_medium_dataset(self):
        r = recommend_training_config(dataset_size=5000)
        assert r.confidence == 0.8
        assert "Medium" in r.reason

    def test_large_dataset(self):
        r = recommend_training_config(dataset_size=50000)
        assert r.confidence == 0.85
        assert "Large" in r.reason

    def test_very_large_dataset(self):
        r = recommend_training_config(dataset_size=500000)
        assert r.confidence == 0.9
        assert "Very large" in r.reason

    def test_finetune_method(self):
        r = recommend_training_config(dataset_size=5000, method="finetune")
        assert r.learning_rate < 1e-3

    def test_native_method(self):
        r = recommend_training_config(dataset_size=5000, method="native")
        assert r.epochs >= 10

    def test_small_model_adjustment(self):
        r = recommend_training_config(dataset_size=5000, model_params=500_000)
        assert r.learning_rate > 0

    def test_large_model_adjustment(self):
        r = recommend_training_config(dataset_size=5000, model_params=200_000_000)
        assert r.learning_rate > 0
        assert r.batch_size >= 4

    def test_low_quality_data(self):
        r = recommend_training_config(dataset_size=5000, avg_quality=1.0)
        assert r.confidence < 0.8

    def test_high_quality_data(self):
        r_low = recommend_training_config(dataset_size=5000, avg_quality=1.0)
        r_high = recommend_training_config(dataset_size=5000, avg_quality=4.5)
        assert r_high.learning_rate >= r_low.learning_rate

    def test_warmup_steps(self):
        r = recommend_training_config(dataset_size=5000)
        assert 10 <= r.warmup_steps <= 100

    def test_early_stopping_patience(self):
        r = recommend_training_config(dataset_size=5000)
        assert r.early_stopping_patience >= 3

    def test_batch_size_clamped(self):
        r = recommend_training_config(dataset_size=1_000_000)
        assert r.batch_size <= 128
        assert r.batch_size >= 4

    def test_epochs_clamped(self):
        r = recommend_training_config(dataset_size=50)
        assert r.epochs >= 5

    def test_learning_rate_clamped(self):
        r = recommend_training_config(dataset_size=1_000_000, model_params=200_000_000)
        assert r.learning_rate >= 1e-6
        assert r.learning_rate <= 1e-2


# ── get_training_tips ──────────────────────────────────────────────────────


class TestGetTrainingTips:

    def test_small_dataset_tips(self):
        tips = get_training_tips(dataset_size=50)
        assert any("Very small" in t for t in tips)

    def test_medium_dataset_tips(self):
        tips = get_training_tips(dataset_size=500)
        assert any("Small" in t for t in tips)

    def test_overfitting_tip(self):
        tips = get_training_tips(dataset_size=5000, current_loss=1.0, best_loss=0.3)
        assert any("overfitting" in t.lower() for t in tips)

    def test_converged_tip(self):
        tips = get_training_tips(dataset_size=5000, current_loss=0.5, best_loss=0.495)
        assert any("converged" in t.lower() for t in tips)

    def test_diverging_tip(self):
        tips = get_training_tips(dataset_size=5000, trend=0.5)
        assert any("increasing" in t.lower() for t in tips)

    def test_improving_tip(self):
        tips = get_training_tips(dataset_size=5000, trend=-0.5)
        assert any("decreasing" in t.lower() for t in tips)

    def test_healthy_tip(self):
        tips = get_training_tips(dataset_size=5000)
        assert any("healthy" in t.lower() for t in tips)
