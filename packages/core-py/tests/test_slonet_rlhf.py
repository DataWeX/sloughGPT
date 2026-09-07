"""Tests for training.rlhf — RLHF components."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.rlhf import (
    RLHFMetric,
    RLHFConfig,
    _as_array,
    _get_logprobs,
    _compute_gae,
    RewardModel,
    ValueHead,
)
from domains.training.slonet import Tensor


# ── RLHFMetric ──────────────────────────────────────────────────────────────


class TestRLHFMetric:

    def test_values(self):
        assert RLHFMetric.REWARD.value == "reward"
        assert RLHFMetric.KL_DIVERGENCE.value == "kl_divergence"
        assert RLHFMetric.ENTROPY.value == "entropy"


# ── RLHFConfig ──────────────────────────────────────────────────────────────


class TestRLHFConfig:

    def test_default(self):
        config = RLHFConfig()
        assert config.ppo_epochs == 4
        assert config.clip_epsilon == 0.2
        assert config.gamma == 1.0


# ── _as_array ───────────────────────────────────────────────────────────────


class TestAsArray:

    def test_numpy(self):
        arr = np.array([1.0, 2.0])
        result = _as_array(arr)
        assert isinstance(result, np.ndarray)

    def test_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _as_array(t)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0])


# ── _get_logprobs ──────────────────────────────────────────────────────────


class TestGetLogprobs:

    def test_tuple_output(self):
        class MockModel:
            def __call__(self, x):
                return (np.random.randn(1, 3, 10), None)
        logprobs = _get_logprobs(MockModel(), np.array([[1, 2, 3]]))
        assert logprobs.shape == (1, 3, 10)

    def test_raw_logits(self):
        class MockModel:
            def __call__(self, x):
                return np.random.randn(1, 3, 10)
        logprobs = _get_logprobs(MockModel(), np.array([[1, 2, 3]]))
        assert logprobs.shape == (1, 3, 10)


# ── _compute_gae ────────────────────────────────────────────────────────────


class TestComputeGae:

    def test_basic(self):
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.0, 0.0, 0.0])
        dones = np.array([False, False, False])
        advantages, returns = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

    def test_with_done(self):
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.0, 0.0, 0.0])
        dones = np.array([False, True, False])
        advantages, returns = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert advantages.shape == rewards.shape

    def test_done_at_end(self):
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.0, 0.0, 0.0])
        dones = np.array([False, False, True])
        advantages, returns = _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert advantages.shape == rewards.shape


# ── RewardModel ─────────────────────────────────────────────────────────────


class TestRewardModel:

    def test_init(self):
        class MockModel:
            def __call__(self, x):
                return np.random.randn(1, 3, 10)
        model = RewardModel(MockModel(), hidden_size=10)
        assert model.hidden_size == 10

    def test_forward(self):
        class MockModel:
            def __call__(self, x):
                return np.random.randn(1, 3, 10)
        model = RewardModel(MockModel(), hidden_size=10)
        x = np.array([[1, 2, 3]])
        reward = model.forward(x)
        assert reward.shape == (1,)


# ── ValueHead ───────────────────────────────────────────────────────────────


class TestValueHead:

    def test_init(self):
        class MockModel:
            def __call__(self, x):
                return np.random.randn(1, 3, 10)
        head = ValueHead(MockModel())
        assert head.base_model is not None

    def test_forward(self):
        class MockModel:
            def __call__(self, x):
                return np.random.randn(1, 3, 10)
        head = ValueHead(MockModel())
        x = np.array([[1, 2, 3]])
        value = head.forward(x)
        assert value.shape == (1,)
