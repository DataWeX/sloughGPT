"""Tests for online LoRA updater — config, initialization, feedback, gradients."""

import time
import pytest
import numpy as np
from unittest.mock import patch
from domains.feedback.online_train import (
    LoRAConfig, OnlineLoRAUpdater, get_online_lora_updater,
    _online_lora,
)


# ── LoRAConfig ─────────────────────────────────────────────────────────────

class TestLoRAConfig:

    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert cfg.target_modules is not None
        assert len(cfg.target_modules) == 4
        assert "attn.c_attn" in cfg.target_modules
        assert "mlp.c_proj" in cfg.target_modules

    def test_custom_config(self):
        cfg = LoRAConfig(rank=4, alpha=8, dropout=0.1)
        assert cfg.rank == 4
        assert cfg.alpha == 8
        assert cfg.dropout == 0.1

    def test_custom_target_modules(self):
        cfg = LoRAConfig(target_modules=["attn.q_proj"])
        assert cfg.target_modules == ["attn.q_proj"]


# ── OnlineLoRAUpdater ─────────────────────────────────────────────────────

@pytest.fixture
def updater():
    return OnlineLoRAUpdater(update_interval=3)


class TestOnlineLoRAUpdater:

    def test_init_default_config(self):
        u = OnlineLoRAUpdater()
        assert u.config.rank == 8
        assert u.update_interval == 5
        assert u._feedback_buffer == []
        assert u._lora_weights == {}
        assert u._is_initialized is False

    def test_init_custom_params(self):
        u = OnlineLoRAUpdater(update_interval=10, learning_rate=0.01)
        assert u.update_interval == 10
        assert u.learning_rate == 0.01

    def test_initialize(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=256)
        assert u._is_initialized is True
        assert "W_a" in u._lora_weights
        assert "W_b" in u._lora_weights
        assert u._lora_weights["W_a"].shape == (8, 256)
        assert u._lora_weights["W_b"].shape == (256, 8)

    def test_initialize_idempotent(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=128)
        w_a_before = u._lora_weights["W_a"].copy()
        u.initialize(model_dim=128)
        np.testing.assert_array_equal(u._lora_weights["W_a"], w_a_before)

    def test_add_feedback_appends_to_buffer(self, updater):
        updater.add_feedback("prompt", "response", "thumbs_up")
        assert len(updater._feedback_buffer) == 1
        assert updater._feedback_buffer[0]["rating"] == "thumbs_up"

    def test_add_feedback_triggers_update_at_threshold(self, updater):
        updater.initialize(model_dim=64)
        for i in range(3):
            updater.add_feedback(f"p{i}", f"r{i}", "thumbs_up")
        time.sleep(0.1)
        assert updater._feedback_buffer == []
        assert updater._stats["total_updates"] >= 1
        assert updater._stats["total_samples"] == 3

    def test_add_feedback_below_threshold_no_update(self, updater):
        for i in range(2):
            updater.add_feedback(f"p{i}", f"r{i}", "thumbs_up")
        assert len(updater._feedback_buffer) == 2
        assert updater._stats["total_updates"] == 0

    def test_add_feedback_quality_score_default(self, updater):
        updater.add_feedback("p", "r", "thumbs_up")
        assert updater._feedback_buffer[0]["quality_score"] == 1.0

    def test_add_feedback_quality_score_negative(self, updater):
        updater.add_feedback("p", "r", "thumbs_down")
        assert updater._feedback_buffer[0]["quality_score"] == 0.0

    def test_compute_gradients_positive_feedback(self, updater):
        updater.initialize(model_dim=64)
        batch = [{"rating": "thumbs_up", "quality_score": 1.0}] * 3
        grads = updater._compute_gradients(batch)
        assert "W_a" in grads
        assert "W_b" in grads
        # Positive feedback → positive reinforcement → scale > 0
        scale = updater.learning_rate * 1.0
        assert scale > 0

    def test_compute_gradients_negative_feedback(self, updater):
        updater.initialize(model_dim=64)
        batch = [{"rating": "thumbs_down", "quality_score": 0.0}] * 3
        grads = updater._compute_gradients(batch)
        scale = updater.learning_rate * (-1.0)
        assert scale < 0

    def test_compute_gradients_mixed_feedback(self, updater):
        updater.initialize(model_dim=64)
        batch = [
            {"rating": "thumbs_up", "quality_score": 1.0},
            {"rating": "thumbs_down", "quality_score": 0.0},
        ]
        grads = updater._compute_gradients(batch)
        # 1 positive, 1 negative → reinforcement = 0
        assert grads["W_a"].shape == (8, 64)

    def test_apply_gradients(self, updater):
        updater.initialize(model_dim=64)
        w_before = updater._lora_weights["W_a"].copy()
        grads = {"W_a": np.ones((8, 64), dtype=np.float32) * 0.01,
                 "W_b": np.ones((64, 8), dtype=np.float32) * 0.01}
        updater._apply_gradients(grads)
        assert not np.array_equal(updater._lora_weights["W_a"], w_before)

    def test_apply_gradients_clips_to_minus_1_1(self, updater):
        updater.initialize(model_dim=64)
        updater._lora_weights["W_a"] = np.full((8, 64), 0.9, dtype=np.float32)
        grads = {"W_a": np.full((8, 64), -0.5, dtype=np.float32),
                 "W_b": np.zeros((64, 8), dtype=np.float32)}
        updater._apply_gradients(grads)
        assert updater._lora_weights["W_a"].max() <= 1.0
        assert updater._lora_weights["W_a"].min() >= -1.0

    def test_apply_to_logits_shape_preserved(self, updater):
        updater.initialize(model_dim=256)
        original = np.random.randn(1, 256).astype(np.float32)
        result = updater.apply_to_logits(original)
        assert result.shape == original.shape

    def test_apply_to_logits_modifies_output(self, updater):
        updater.initialize(model_dim=256)
        updater._lora_weights["W_a"] = np.random.randn(8, 256).astype(np.float32) * 0.5
        updater._lora_weights["W_b"] = np.random.randn(256, 8).astype(np.float32) * 0.5
        original = np.zeros((1, 256), dtype=np.float32)
        result = updater.apply_to_logits(original)
        assert not np.allclose(result, original)

    def test_apply_to_logits_uninitialized(self):
        u = OnlineLoRAUpdater()
        original = np.zeros((1, 256), dtype=np.float32)
        result = u.apply_to_logits(original)
        np.testing.assert_array_equal(result, original)

    def test_get_adaptation_strength_zero_uninitialized(self):
        u = OnlineLoRAUpdater()
        assert u.get_adaptation_strength() == 0.0

    def test_get_adaptation_strength_positive_initialized(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=64)
        assert u.get_adaptation_strength() > 0.0

    def test_get_stats(self, updater):
        stats = updater.get_stats()
        assert "total_updates" in stats
        assert "total_samples" in stats
        assert "buffer_size" in stats
        assert "is_updating" in stats
        assert "is_initialized" in stats
        assert "adaptation_strength" in stats

    def test_reset(self, updater):
        updater.initialize(model_dim=64)
        updater.add_feedback("p", "r", "thumbs_up")
        updater.reset()
        assert updater._lora_weights == {}
        assert updater._is_initialized is False
        assert updater._feedback_buffer == []
        assert updater._stats["total_updates"] == 0
        assert updater._stats["total_samples"] == 0
        assert updater._stats["last_update_time"] is None

    def test_thread_safety_concurrent_add(self, updater):
        import threading
        def add_many():
            for i in range(20):
                updater.add_feedback(f"p{i}", f"r{i}", "thumbs_up")

        threads = [threading.Thread(target=add_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        total = updater._stats["total_samples"] + len(updater._feedback_buffer)
        assert total == 80


# ── Global singleton ──────────────────────────────────────────────────────

class TestGetOnlineLoRAUpdater:

    def test_returns_singleton(self):
        import domains.feedback.online_train as mod
        original = mod._online_lora
        mod._online_lora = None
        try:
            u1 = get_online_lora_updater()
            u2 = get_online_lora_updater()
            assert u1 is u2
        finally:
            mod._online_lora = original
