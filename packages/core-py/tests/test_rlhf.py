"""Tests for domains/training/rlhf.py (RLHFConfig, RLHFMetric, RewardModel, _as_array, create_rlhf_trainer)."""

import numpy as np
import pytest

from domains.training.slonet import SloLinear, Tensor
from domains.training.rlhf import (
    RLHFConfig,
    RLHFMetric,
    RewardModel,
    _as_array,
    create_rlhf_trainer,
)


class FakeBaseModel:
    def __init__(self, output):
        self.output = output

    def __call__(self, input_ids):
        return self.output


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
# create_rlhf_trainer
# ---------------------------------------------------------------------------

class TestCreateRlhfTrainer:
    def test_default_config(self):
        c = create_rlhf_trainer()
        assert isinstance(c, RLHFConfig)

    def test_returns_provided_config(self):
        config = RLHFConfig(ppo_epochs=7)
        assert create_rlhf_trainer(config=config) is config

    def test_accepts_models(self):
        c = create_rlhf_trainer(policy_model=object(), value_model=object(),
                                ref_model=object(), device="cuda")
        assert isinstance(c, RLHFConfig)
