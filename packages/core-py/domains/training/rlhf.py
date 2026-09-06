"""
RLHF (Reinforcement Learning from Human Feedback) Module for SloughGPT

Implements PPO (Proximal Policy Optimization) for model alignment.
Includes:
- PPO Trainer
- Reward Model
- Reference Model (for KL divergence)
- Advantage estimation (GAE)

Runs entirely on numpy/SloNet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
from enum import Enum

import numpy as np
from domains.training.slonet import SloLinear, Tensor

logger = logging.getLogger("slo.rlhf")


class RLHFMetric(Enum):
    """RLHF training metrics."""

    REWARD = "reward"
    KL_DIVERGENCE = "kl_divergence"
    VALUE_LOSS = "value_loss"
    POLICY_LOSS = "policy_loss"
    ENTROPY = "entropy"
    ADVANTAGE = "advantage"


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""

    # PPO parameters
    ppo_epochs: int = 4
    num_mini_batches: int = 4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda

    # Model parameters
    reward_model_path: Optional[str] = None
    ref_model_path: Optional[str] = None
    use_ref_model: bool = True

    # Generation
    gen_max_length: int = 512
    gen_temperature: float = 1.0
    gen_top_p: float = 0.9


def _as_array(x) -> np.ndarray:
    """Coerce a SloNet Tensor (or ndarray) into its underlying numpy buffer."""
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(x)


class RewardModel:
    """
    Reward Model for RLHF.

    Takes a prompt-response pair and outputs a scalar reward.
    """

    def __init__(self, base_model, hidden_size: int = 512):
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.reward_head = None
        self._feature_dim = None

    def _ensure_head(self, feature_dim: int):
        """Lazily build the reward projection to match the base model output."""
        if self._feature_dim == feature_dim and self.reward_head is not None:
            return
        self.reward_head = SloLinear(feature_dim, 1)
        self._feature_dim = feature_dim

    def forward(self, input_ids) -> Tensor:
        """
        Compute reward for input.

        Args:
            input_ids: Input token IDs [batch_size, seq_len] (numpy or Tensor)

        Returns:
            rewards: Scalar rewards [batch_size]
        """
        hidden = self.base_model(input_ids)
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # Get logits only
        arr = _as_array(hidden)
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        # Use last token's representation
        last_hidden = arr[:, -1, :]
        self._ensure_head(last_hidden.shape[-1])
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

    def __call__(self, input_ids):
        return self.forward(input_ids)


def create_rlhf_trainer(
    policy_model=None,
    value_model: Optional[object] = None,
    ref_model: Optional[object] = None,
    config: Optional[RLHFConfig] = None,
    device: str = "cpu",
):
    """
    Create an RLHF trainer.

    Args:
        policy_model: The model to train (optional)
        value_model: Value function (can be same as policy)
        ref_model: Reference model for KL penalty
        config: RLHF configuration
        device: Device to use

    Returns:
        RLHFConfig with the provided models attached
    """
    config = config or RLHFConfig()
    # Attach models to config so callers can access them downstream.
    # TODO: Implement actual PPO training loop (currently a stub).
    config.policy_model = policy_model
    config.value_model = value_model
    config.ref_model = ref_model
    config.device = device
    return config


__all__ = [
    "RLHFConfig",
    "RLHFMetric",
    "RewardModel",
    "create_rlhf_trainer",
]
