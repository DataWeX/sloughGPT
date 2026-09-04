"""Tests for RLHF config and metrics."""
from __future__ import annotations

from domains.training.rlhf import RLHFConfig, RLHFMetric


class TestRLHFMetric:
    def test_all_metrics(self):
        assert len(RLHFMetric) == 6
        assert RLHFMetric.REWARD.value == "reward"
        assert RLHFMetric.KL_DIVERGENCE.value == "kl_divergence"


class TestRLHFConfig:
    def test_defaults(self):
        cfg = RLHFConfig()
        assert cfg.ppo_epochs == 4
        assert cfg.num_mini_batches == 4
        assert cfg.clip_epsilon == 0.2
        assert cfg.value_loss_coef == 0.5
        assert cfg.entropy_coef == 0.01
        assert cfg.gamma == 1.0
        assert cfg.lam == 0.95
        assert cfg.use_ref_model is True

    def test_override(self):
        cfg = RLHFConfig(ppo_epochs=8, clip_epsilon=0.3)
        assert cfg.ppo_epochs == 8
        assert cfg.clip_epsilon == 0.3

    def test_generation_params(self):
        cfg = RLHFConfig(gen_max_length=256, gen_temperature=0.8)
        assert cfg.gen_max_length == 256
        assert cfg.gen_temperature == 0.8
