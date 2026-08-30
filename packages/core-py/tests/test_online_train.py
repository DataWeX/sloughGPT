"""Tests for online_train — LoRA feedback buffer and weight updates."""

import numpy as np
import time
import threading
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

    def test_target_modules_default(self):
        cfg = LoRAConfig()
        assert "attn.c_attn" in cfg.target_modules
        assert "attn.c_proj" in cfg.target_modules
        assert "mlp.c_fc" in cfg.target_modules
        assert "mlp.c_proj" in cfg.target_modules

    def test_target_modules_custom(self):
        cfg = LoRAConfig(target_modules=["attn.c_attn"])
        assert cfg.target_modules == ["attn.c_attn"]

    def test_dropout_custom(self):
        cfg = LoRAConfig(dropout=0.1)
        assert cfg.dropout == 0.1

    def test_rank_positive(self):
        cfg = LoRAConfig(rank=16)
        assert cfg.rank == 16

    def test_alpha_positive(self):
        cfg = LoRAConfig(alpha=32)
        assert cfg.alpha == 32


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


class TestGradientComputation:
    def test_mixed_feedback(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        batch = [
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "a", "response": "b"},
            {"rating": "thumbs_down", "quality_score": 0.0, "prompt": "c", "response": "d"},
        ]
        grads = u._compute_gradients(batch)
        # Reinforcement = (1-1)/2 = 0 → scale = 0
        for g in grads.values():
            assert np.allclose(g, 0.0, atol=1e-6)

    def test_all_positive_feedback(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        batch = [
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "a", "response": "b"}
            for _ in range(5)
        ]
        grads = u._compute_gradients(batch)
        # Positive reinforcement → positive scale
        for g in grads.values():
            assert np.isfinite(g).all()

    def test_all_negative_feedback(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        batch = [
            {"rating": "thumbs_down", "quality_score": 0.0, "prompt": "a", "response": "b"}
            for _ in range(5)
        ]
        grads = u._compute_gradients(batch)
        for g in grads.values():
            assert np.isfinite(g).all()

    def test_empty_batch(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        grads = u._compute_gradients([])
        # Empty batch → reinforcement = 0/1 = 0
        for g in grads.values():
            assert np.allclose(g, 0.0, atol=1e-6)

    def test_gradient_shapes_match_weights(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=32)
        batch = [{"rating": "thumbs_up", "quality_score": 1.0, "prompt": "q", "response": "a"}]
        grads = u._compute_gradients(batch)
        assert grads["W_a"].shape == u._lora_weights["W_a"].shape
        assert grads["W_b"].shape == u._lora_weights["W_b"].shape


class TestGradientApplication:
    def test_apply_small_gradients(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        w_before = u._lora_weights["W_a"].copy()
        grads = {"W_a": np.ones((8, 16), dtype=np.float32) * 0.001}
        u._apply_gradients(grads)
        # Weights should change
        assert not np.array_equal(u._lora_weights["W_a"], w_before)

    def test_apply_unknown_key_ignored(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        w_before = u._lora_weights["W_a"].copy()
        grads = {"unknown_key": np.ones((8, 16), dtype=np.float32)}
        u._apply_gradients(grads)
        assert np.array_equal(u._lora_weights["W_a"], w_before)

    def test_apply_preserves_finiteness(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        grads = {
            "W_a": np.random.randn(8, 16).astype(np.float32) * 100,
            "W_b": np.random.randn(16, 8).astype(np.float32) * -100,
        }
        u._apply_gradients(grads)
        assert np.isfinite(u._lora_weights["W_a"]).all()
        assert np.isfinite(u._lora_weights["W_b"]).all()

    def test_repeated_apply_converges_to_clip(self):
        """Repeated large gradient application clips to [-1, 1]."""
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        for _ in range(20):
            grads = {"W_a": np.ones((8, 8), dtype=np.float32) * 10.0}
            u._apply_gradients(grads)
        assert u._lora_weights["W_a"].max() <= 1.0


class TestApplyToLogits:
    def test_shape_preserved(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        logits = np.random.randn(1, 16).astype(np.float32)
        result = u.apply_to_logits(logits)
        assert result.shape == logits.shape

    def test_modifies_logits(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        # Set W_b to non-zero so the LoRA adjustment is non-trivial
        u._lora_weights["W_b"] = np.ones((16, 8), dtype=np.float32) * 0.1
        logits = np.zeros((1, 16), dtype=np.float32)
        result = u.apply_to_logits(logits)
        # LoRA adjustment should change logits
        assert not np.array_equal(result, logits)

    def test_uninit_returns_original(self):
        u = OnlineLoRAUpdater()
        logits = np.ones((2, 10))
        result = u.apply_to_logits(logits)
        assert np.array_equal(result, logits)

    def test_layer_name_ignored(self):
        """layer_name param doesn't affect result."""
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=8)
        logits = np.zeros((1, 8), dtype=np.float32)
        r1 = u.apply_to_logits(logits, layer_name="attention")
        r2 = u.apply_to_logits(logits, layer_name="mlp")
        np.testing.assert_array_equal(r1, r2)


class TestAdaptationStrength:
    def test_zero_initially(self):
        u = OnlineLoRAUpdater()
        assert u.get_adaptation_strength() == 0.0

    def test_positive_after_init(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        strength = u.get_adaptation_strength()
        assert strength > 0

    def test_scales_with_rank(self):
        u1 = OnlineLoRAUpdater(config=LoRAConfig(rank=4))
        u1.initialize(model_dim=16)
        s1 = u1.get_adaptation_strength()

        u2 = OnlineLoRAUpdater(config=LoRAConfig(rank=16))
        u2.initialize(model_dim=16)
        s2 = u2.get_adaptation_strength()
        assert s2 > s1


class TestStats:
    def test_initial_stats(self):
        u = OnlineLoRAUpdater()
        stats = u.get_stats()
        assert stats["total_updates"] == 0
        assert stats["total_samples"] == 0
        assert stats["last_update_time"] is None
        assert stats["average_update_ms"] == 0
        assert stats["buffer_size"] == 0
        assert stats["is_updating"] is False
        assert stats["is_initialized"] is False

    def test_stats_after_init(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        stats = u.get_stats()
        assert stats["is_initialized"] is True
        assert stats["adaptation_strength"] > 0

    def test_stats_after_feedback(self):
        u = OnlineLoRAUpdater(update_interval=10)
        u.initialize(model_dim=16)
        u.add_feedback("q", "a", "thumbs_up")
        stats = u.get_stats()
        assert stats["buffer_size"] == 1


class TestReset:
    def test_reset_clears_all(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u.add_feedback("q", "a", "thumbs_up")
        u.reset()
        stats = u.get_stats()
        assert stats["total_updates"] == 0
        assert stats["total_samples"] == 0
        assert stats["buffer_size"] == 0
        assert stats["is_initialized"] is False

    def test_reset_allows_reinit(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u.reset()
        u.initialize(model_dim=32)
        assert u._lora_weights["W_a"].shape == (8, 32)


class TestThreadSafety:
    def test_concurrent_add_feedback(self):
        u = OnlineLoRAUpdater(update_interval=100)
        u.initialize(model_dim=16)
        def _add():
            for i in range(10):
                u.add_feedback("q", "a", "thumbs_up")
        threads = [threading.Thread(target=_add) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(u._feedback_buffer) == 50

    def test_concurrent_add_different_ratings(self):
        u = OnlineLoRAUpdater(update_interval=100)
        u.initialize(model_dim=16)
        def _add_up():
            for _ in range(5):
                u.add_feedback("q", "a", "thumbs_up")
        def _add_down():
            for _ in range(5):
                u.add_feedback("q", "a", "thumbs_down")
        threads = [
            threading.Thread(target=_add_up),
            threading.Thread(target=_add_down),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(u._feedback_buffer) == 10

    def test_trigger_update_reentrant(self):
        """_trigger_update is a no-op when already updating."""
        u = OnlineLoRAUpdater(update_interval=1)
        u.initialize(model_dim=16)
        u._is_updating = True
        # Should not start a new update
        u._trigger_update()
        assert u._is_updating is True
        u._is_updating = False


class TestPerformUpdate:
    def test_perform_update_empty_buffer(self):
        """_perform_update with empty buffer → no crash."""
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u._perform_update()
        assert u._stats["total_updates"] == 0

    def test_perform_update_with_buffer(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u._feedback_buffer = [
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "q", "response": "a"},
        ]
        u._perform_update()
        assert u._stats["total_updates"] == 1
        assert u._stats["total_samples"] == 1
        assert u._stats["last_update_time"] is not None

    def test_perform_update_clears_buffer(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u._feedback_buffer = [
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "q", "response": "a"},
        ]
        u._perform_update()
        assert len(u._feedback_buffer) == 0

    def test_perform_update_sets_is_updating_false(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        u._feedback_buffer = [
            {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "q", "response": "a"},
        ]
        u._perform_update()
        assert u._is_updating is False

    def test_perform_update_multiple(self):
        u = OnlineLoRAUpdater()
        u.initialize(model_dim=16)
        for _ in range(3):
            u._feedback_buffer = [
                {"rating": "thumbs_up", "quality_score": 1.0, "prompt": "q", "response": "a"},
            ]
            u._perform_update()
        assert u._stats["total_updates"] == 3
        assert u._stats["total_samples"] == 3
