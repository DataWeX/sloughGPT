"""Tests for online_train — LoRA feedback buffer and weight updates."""

import numpy as np
import time
from domains.feedback.online_train import (
    LoRAConfig,
    OnlineLoRAUpdater,
    get_online_lora_updater,
)


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert isinstance(cfg.target_modules, list)

    def test_custom(self):
        cfg = LoRAConfig(rank=4, alpha=8)
        assert cfg.rank == 4
        assert cfg.alpha == 8


class TestOnlineLoRAUpdater:
    def test_init(self):
        u = OnlineLoRAUpdater()
        assert u.engine is None
        assert u.update_interval == 5
        assert not u._is_initialized

    def test_init_custom(self):
        u = OnlineLoRAUpdater(update_interval=3, learning_rate=0.01)
        assert u.update_interval == 3
        assert u.learning_rate == 0.01

    def test_initialize_creates_weights(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=32)
        assert u._is_initialized
        assert "W_a" in u._lora_weights
        assert "W_b" in u._lora_weights
        assert u._lora_weights["W_a"].shape == (8, 32)
        assert u._lora_weights["W_b"].shape == (32, 8)

    def test_initialize_idempotent(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=32)
        w_a_before = u._lora_weights["W_a"].copy()
        u.initialize(model_dim=32)
        assert np.array_equal(u._lora_weights["W_a"], w_a_before)

    def test_add_feedback_buffers(self):
        u = OnlineLoRAUpdater(update_interval=10)
        u.initialize(model_dim=16)
        u.add_feedback("hi", "hello", "thumbs_up")
        assert len(u._feedback_buffer) == 1
        assert u._feedback_buffer[0]["rating"] == "thumbs_up"

    def test_add_feedback_triggers_update(self):
        u = OnlineLoRAUpdater(update_interval=2)
        u.initialize(model_dim=16)
        u.add_feedback("q1", "a1", "thumbs_up")
        assert not u._is_updating
        u.add_feedback("q2", "a2", "thumbs_down")
        time.sleep(0.05)
        assert u._stats["total_updates"] >= 1

    def test_quality_score_default(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        u.add_feedback("q", "a", "thumbs_up")
        assert u._feedback_buffer[0]["quality_score"] == 1.0
        u.add_feedback("q", "a", "thumbs_down")
        assert u._feedback_buffer[1]["quality_score"] == 0.0

    def test_compute_gradients_reinforcement(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        batch = [
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "a", "response": "b"},
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "c", "response": "d"},
        ]
        grads = u._compute_gradients(batch)
        assert "W_a" in grads
        assert "W_b" in grads

    def test_compute_gradients_all_negative(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        batch = [{"rating": "thumbs_down", "quality_score": 0.0, "prompt": "a", "response": "b"}]
        grads = u._compute_gradients(batch)
        assert all(np.isfinite(g).all() for g in grads.values())

    def test_apply_gradients_clips(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        grads = {"W_a": np.ones((8, 16), dtype=np.float32) * 10.0,
                 "W_b": np.ones((16, 8), dtype=np.float32) * -10.0}
        u._apply_gradients(grads)
        assert u._lora_weights["W_a"].max() <= 1.0
        assert u._lora_weights["W_b"].min() >= -1.0

    def test_apply_to_logits_uninit(self):
        u = OnlineLoRAUpdater()
        logits = np.zeros((1, 10))
        result = u.apply_to_logits(logits)
        assert np.array_equal(result, logits)

    def test_apply_to_logits_init(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        logits = np.zeros((1, 16), dtype=np.float32)
        result = u.apply_to_logits(logits)
        assert result.shape == logits.shape
        assert np.all(np.isfinite(result))

    def test_adaptation_strength_zero_when_empty(self):
        u = OnlineLoRAUpdater()
        assert u.get_adaptation_strength() == 0.0

    def test_adaptation_strength_positive_when_init(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        assert u.get_adaptation_strength() > 0.0

    def test_get_stats(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        stats = u.get_stats()
        assert "total_updates" in stats
        assert "buffer_size" in stats
        assert "is_initialized" in stats
        assert stats["is_initialized"] is True

    def test_reset(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u.add_feedback("q", "a", "thumbs_up")
        u.reset()
        assert not u._is_initialized
        assert len(u._lora_weights) == 0
        assert len(u._feedback_buffer) == 0
        assert u._stats["total_updates"] == 0

    def test_singleton(self):
        a = get_online_lora_updater()
        b = get_online_lora_updater()
        assert a is b


class TestThreadSafety:
    def test_concurrent_add_feedback(self):
        u = OnlineLoRAUpdater(update_interval=100)
        u.initialize(model_dim=16)
        import threading
        def _add():
            for i in range(10):
                u.add_feedback("q", "a", "thumbs_up")
        threads = [threading.Thread(target=_add) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(u._feedback_buffer) == 50
