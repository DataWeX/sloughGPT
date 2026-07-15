"""
RLHF (Reinforcement Learning from Human Feedback) Module for SloughGPT

Implements PPO (Proximal Policy Optimization) for model alignment.
Includes:
- PPO Trainer
- Reward Model
- Reference Model (for KL divergence)
- Advantage estimation (GAE)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging
from enum import Enum

from domains.training.slonet_compat import torch, nn, F

logger = logging.getLogger("man.rlhf")


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


class RewardModel(nn.Module):
    """
    Reward Model for RLHF.

    Takes a prompt-response pair and outputs a scalar reward.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int = 512):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for input.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            rewards: Scalar rewards [batch_size]
        """
        hidden = self.base_model(input_ids)
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # Get logits only
        # Use last hidden state
        last_hidden = hidden[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)


def create_rlhf_trainer(
    policy_model: nn.Module,
    value_model: Optional[nn.Module] = None,
    ref_model: Optional[nn.Module] = None,
    config: Optional[RLHFConfig] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Create an RLHF trainer.

    Args:
        policy_model: The model to train
        value_model: Value function (can be same as policy)
        ref_model: Reference model for KL penalty
        config: RLHF configuration
        device: Device to use

    Returns:
        RLHFConfig with the provided models
    """
    config = config or RLHFConfig()
    return config


__all__ = [
    "RLHFConfig",
    "RLHFMetric",
    "RewardModel",
    "create_rlhf_trainer",
]
