"""Tests for domains/training/rlhf.py (RLHFConfig, RLHFMetric, RewardModel, PPOTrainer, _as_array, create_rlhf_trainer)."""

import numpy as np
import pytest

from domains.training.slonet import SloLinear, Tensor
from domains.training.rlhf import (
    RLHFConfig,
    RLHFMetric,
    RewardModel,
    ValueHead,
    PPOTrainer,
    _as_array,
    _compute_gae,
    _get_logprobs,
    create_rlhf_trainer,
)


class FakeBaseModel:
    """A fake model that returns fixed-shape tensors for testing."""
    def __init__(self, output):
        self.output = output

    def __call__(self, input_ids):
        return self.output


class FakeTransformerModel:
    """A fake transformer that behaves like SloTransformer for testing."""
    def __init__(self, vocab_size: int = 32, hidden: int = 16, seq_len: int = 8):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.seq_len = seq_len
        self.embed = SloLinear(vocab_size, hidden)
        self.head = SloLinear(hidden, vocab_size)

    def __call__(self, input_ids):
        arr = _as_array(input_ids)
        B = arr.shape[0] if arr.ndim > 0 else 1
        # Simple embedding projection
        x = Tensor(np.random.randn(B, self.seq_len, self.hidden).astype(np.float32))
        logits = self.head.forward(x)
        return (logits, None)


def make_tensor(shape, dtype=np.float32):
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=False)


# ---------------------------------------------------------------------------
# RLHFMetric
# ---------------------------------------------------------------------------

class TestRLHFMetric:
    def test_members(self):
        assert RLHFMetric.REWARD.value == "reward"
        assert RLHFMetric.KL_DIVERGENCE.value == "kl_divergence"
        assert RLHFMetric.VALUE_LOSS.value == "value_loss"
        assert RLHFMetric.POLICY_LOSS.value == "policy_loss"
        assert RLHFMetric.ENTROPY.value == "entropy"
        assert RLHFMetric.ADVANTAGE.value == "advantage"


# ---------------------------------------------------------------------------
# RLHFConfig
# ---------------------------------------------------------------------------

class TestRLHFConfig:
    def test_defaults(self):
        c = RLHFConfig()
        assert c.ppo_epochs == 4
        assert c.num_mini_batches == 4
        assert c.clip_epsilon == 0.2
        assert c.value_loss_coef == 0.5
        assert c.entropy_coef == 0.01
        assert c.max_grad_norm == 1.0
        assert c.gamma == 1.0
        assert c.lam == 0.95
        assert c.reward_model_path is None
        assert c.ref_model_path is None
        assert c.use_ref_model is True
        assert c.gen_max_length == 512
        assert c.gen_temperature == 1.0
        assert c.gen_top_p == 0.9
        assert c.target_kl == 0.02
        assert c.kl_coef == 0.1

    def test_custom(self):
        c = RLHFConfig(ppo_epochs=2, clip_epsilon=0.3, use_ref_model=False,
                       gen_max_length=128, gen_temperature=0.7, gen_top_p=0.5)
        assert c.ppo_epochs == 2
        assert c.clip_epsilon == 0.3
        assert c.use_ref_model is False
        assert c.gen_max_length == 128
        assert c.gen_temperature == 0.7
        assert c.gen_top_p == 0.5


# ---------------------------------------------------------------------------
# _as_array
# ---------------------------------------------------------------------------

class TestAsArray:
    def test_tensor_returns_data(self):
        t = make_tensor((3, 4))
        out = _as_array(t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 4)

    def test_ndarray_passthrough(self):
        a = np.zeros((2, 2), dtype=np.float32)
        assert _as_array(a) is a

    def test_list_converted(self):
        out = _as_array([1, 2, 3])
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [1, 2, 3]

    def test_scalar_converted(self):
        out = _as_array(7)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 0


# ---------------------------------------------------------------------------
# _compute_gae
# ---------------------------------------------------------------------------

class TestComputeGAE:
    def test_basic(self):
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert adv.shape == (3,)
        assert ret.shape == (3,)
        # After done, no further accumulation
        assert adv[2] == pytest.approx(rewards[2] + 0.0 - values[2], abs=0.01)

    def test_no_discount(self):
        rewards = np.array([1.0, 2.0], dtype=np.float32)
        values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        dones = np.array([0.0, 0.0], dtype=np.float32)
        adv, ret = _compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        # With gamma=1, lam=1: GAE = sum of future rewards - value
        assert adv[0] == pytest.approx(3.0, abs=0.01)
        assert adv[1] == pytest.approx(2.0, abs=0.01)

    def test_all_done(self):
        rewards = np.array([1.0], dtype=np.float32)
        values = np.array([0.0, 0.0], dtype=np.float32)
        dones = np.array([1.0], dtype=np.float32)
        adv, ret = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert adv[0] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _get_logprobs
# ---------------------------------------------------------------------------

class TestGetLogprobs:
    def test_tuple_output(self):
        model = FakeBaseModel((Tensor(np.random.randn(1, 4, 8).astype(np.float32)), None))
        lp = _get_logprobs(model, np.array([[1, 2, 3, 4]], dtype=np.int64))
        assert lp.shape == (1, 4, 8)
        # Should be valid log-probs (sums close to 0 in log space)
        assert np.all(lp <= 0)

    def test_raw_output(self):
        model = FakeBaseModel(Tensor(np.random.randn(1, 4, 8).astype(np.float32)))
        lp = _get_logprobs(model, np.array([[1, 2, 3, 4]], dtype=np.int64))
        assert lp.shape == (1, 4, 8)

    def test_batch(self):
        model = FakeBaseModel(Tensor(np.random.randn(2, 3, 10).astype(np.float32)))
        lp = _get_logprobs(model, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
        assert lp.shape == (2, 3, 10)


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class TestRewardModelInit:
    def test_stores_attributes(self):
        base = FakeBaseModel(make_tensor((1, 2, 3)))
        rm = RewardModel(base, hidden_size=512)
        assert rm.base_model is base
        assert rm.hidden_size == 512
        assert rm.reward_head is None
        assert rm._feature_dim is None


class TestEnsureHead:
    def test_builds_linear_head(self):
        base = FakeBaseModel(make_tensor((1, 2, 3)))
        rm = RewardModel(base)
        rm._ensure_head(8)
        assert isinstance(rm.reward_head, SloLinear)
        assert rm.reward_head.in_features == 8
        assert rm.reward_head.out_features == 1
        assert rm._feature_dim == 8

    def test_same_dim_is_idempotent(self):
        base = FakeBaseModel(make_tensor((1, 2, 3)))
        rm = RewardModel(base)
        rm._ensure_head(8)
        first = rm.reward_head
        rm._ensure_head(8)
        assert rm.reward_head is first

    def test_changes_dim_rebuilds(self):
        base = FakeBaseModel(make_tensor((1, 2, 3)))
        rm = RewardModel(base)
        rm._ensure_head(8)
        rm._ensure_head(16)
        assert rm.reward_head.in_features == 16
        assert rm._feature_dim == 16


class TestRewardForward:
    def test_3d_tensor_output(self):
        base = FakeBaseModel(make_tensor((2, 4, 8)))
        rm = RewardModel(base)
        reward = rm(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32))
        assert isinstance(reward, Tensor)
        assert reward.data.shape == (2,)
        assert reward.data.dtype == np.float32

    def test_tuple_output_uses_first(self):
        base = FakeBaseModel((make_tensor((1, 4, 8)), "other"))
        rm = RewardModel(base)
        reward = rm(np.array([[1, 2, 3, 4]], dtype=np.int32))
        assert reward.data.shape == (1,)

    def test_2d_output_single_token(self):
        base = FakeBaseModel(make_tensor((3, 8)))
        rm = RewardModel(base)
        reward = rm(np.zeros((3, 1), dtype=np.int32))
        assert reward.data.shape == (3,)

    def test_1d_output_single_sample(self):
        base = FakeBaseModel(make_tensor((8,)))
        rm = RewardModel(base)
        reward = rm(np.zeros((1, 1), dtype=np.int32))
        assert reward.data.shape == (1,)

    def test_numpy_output(self):
        base = FakeBaseModel(np.random.randn(2, 4, 8).astype(np.float32))
        rm = RewardModel(base)
        reward = rm(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32))
        assert reward.data.shape == (2,)

    def test_call_delegates(self):
        base = FakeBaseModel(make_tensor((1, 3, 8)))
        rm = RewardModel(base)
        assert rm.forward(np.array([[1, 2, 3]], dtype=np.int32)).data.shape == (1,)


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------

class TestValueHead:
    def test_output_shape(self):
        base = FakeBaseModel(make_tensor((2, 4, 8)))
        vh = ValueHead(base)
        val = vh(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32))
        assert val.data.shape == (2,)

    def test_lazy_head(self):
        base = FakeBaseModel(make_tensor((1, 3, 8)))
        vh = ValueHead(base)
        assert vh.head is None
        vh(np.array([[1, 2, 3]], dtype=np.int32))
        assert vh.head is not None

    def test_call_delegates(self):
        base = FakeBaseModel(make_tensor((1, 3, 8)))
        vh = ValueHead(base)
        val = vh(np.array([[1, 2, 3]], dtype=np.int32))
        assert val.data.shape == (1,)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class TestPPOTrainerInit:
    def test_default_config(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm)
        assert trainer.policy is base
        assert trainer.reward_model is rm
        assert isinstance(trainer.config, RLHFConfig)

    def test_custom_config(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        rm = RewardModel(base)
        cfg = RLHFConfig(ppo_epochs=8)
        trainer = PPOTrainer(base, rm, config=cfg)
        assert trainer.config.ppo_epochs == 8

    def test_ref_model(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        ref = FakeBaseModel(make_tensor((1, 4, 8)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm, ref_model=ref)
        assert trainer.ref_model is ref


class TestCollectRollout:
    def test_output_keys(self):
        base = FakeBaseModel(Tensor(np.random.randn(2, 4, 8).astype(np.float32)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm)
        obs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        rollout = trainer.collect_rollout(obs, obs)
        assert "obs" in rollout
        assert "logprobs" in rollout
        assert "values" in rollout
        assert "advantages" in rollout
        assert "returns" in rollout
        assert "rewards" in rollout

    def test_shapes(self):
        base = FakeBaseModel(Tensor(np.random.randn(2, 4, 8).astype(np.float32)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm)
        obs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        rollout = trainer.collect_rollout(obs, obs)
        assert rollout["logprobs"].shape == (2, 4, 8)
        assert rollout["rewards"].shape == (2,)


class TestPPOUpdate:
    def test_returns_metrics(self):
        base = FakeBaseModel(Tensor(np.random.randn(2, 4, 8).astype(np.float32)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm, config=RLHFConfig(ppo_epochs=1, num_mini_batches=1))
        obs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        rollout = trainer.collect_rollout(obs, obs)
        metrics = trainer.update(rollout)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "kl_divergence" in metrics
        assert "mean_reward" in metrics
        assert "epochs_run" in metrics

    def test_metrics_are_finite(self):
        base = FakeBaseModel(Tensor(np.random.randn(2, 4, 8).astype(np.float32)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm, config=RLHFConfig(ppo_epochs=1, num_mini_batches=1))
        obs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        rollout = trainer.collect_rollout(obs, obs)
        metrics = trainer.update(rollout)
        for k, v in metrics.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"metric {k} is not finite: {v}"

    def test_multi_epoch(self):
        base = FakeBaseModel(Tensor(np.random.randn(4, 4, 8).astype(np.float32)))
        rm = RewardModel(base)
        trainer = PPOTrainer(base, rm, config=RLHFConfig(ppo_epochs=3, num_mini_batches=2, target_kl=0.0))
        obs = np.array([[1, 2, 3, 4]] * 4, dtype=np.int64)
        rollout = trainer.collect_rollout(obs, obs)
        metrics = trainer.update(rollout)
        assert metrics["epochs_run"] == 3

    def test_early_stopping(self):
        base = FakeBaseModel(Tensor(np.random.randn(4, 4, 8).astype(np.float32)))
        rm = RewardModel(base)
        # Set target_kl very low to trigger early stopping
        trainer = PPOTrainer(base, rm, config=RLHFConfig(ppo_epochs=10, num_mini_batches=1, target_kl=0.001))
        obs = np.array([[1, 2, 3, 4]] * 4, dtype=np.int64)
        rollout = trainer.collect_rollout(obs, obs)
        metrics = trainer.update(rollout)
        # May or may not early stop depending on KL, but should complete
        assert "early_stopped" in metrics


# ---------------------------------------------------------------------------
# create_rlhf_trainer
# ---------------------------------------------------------------------------

class TestCreateRlhfTrainer:
    def test_returns_trainer(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        trainer = create_rlhf_trainer(policy_model=base)
        assert isinstance(trainer, PPOTrainer)

    def test_default_config(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        trainer = create_rlhf_trainer(policy_model=base)
        assert isinstance(trainer.config, RLHFConfig)

    def test_custom_config(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        cfg = RLHFConfig(ppo_epochs=7)
        trainer = create_rlhf_trainer(policy_model=base, config=cfg)
        assert trainer.config.ppo_epochs == 7

    def test_accepts_models(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        ref = FakeBaseModel(make_tensor((1, 4, 8)))
        trainer = create_rlhf_trainer(policy_model=base, value_model=base, ref_model=ref)
        assert trainer.policy is base
        assert trainer.ref_model is ref

    def test_reward_model_created(self):
        base = FakeBaseModel(make_tensor((1, 4, 8)))
        trainer = create_rlhf_trainer(policy_model=base)
        assert isinstance(trainer.reward_model, RewardModel)
