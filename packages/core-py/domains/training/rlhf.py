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


class PPOTrainer:
    """
    PPO Trainer for RLHF.

    Implements:
    - Generalized Advantage Estimation (GAE)
    - PPO clipping objective
    - Value function clipping
    - KL divergence penalty
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        config: RLHFConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy = policy_model
        self.value = value_model
        self.config = config
        self.device = device

        self.policy_old = None  # For computing importance sampling ratio
        self.ref_model = None  # For KL divergence

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=1e-5,
        )

    def set_ref_model(self, ref_model: nn.Module):
        """Set reference model for KL divergence."""
        self.ref_model = ref_model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        Args:
            rewards: Reward tensor [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            next_values: Next state values [batch_size]

        Returns:
            advantages: GAE advantages
            returns: Returns (advantages + values)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(rewards.size(1))):
            if t == rewards.size(1) - 1:
                next_value = next_values
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages[:, t] = gae

        returns = advantages + values
        return advantages, returns

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PPO loss.

        Args:
            log_probs: New log probabilities [batch_size, seq_len]
            old_log_probs: Old log probabilities [batch_size, seq_len]
            advantages: Advantages [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            returns: Returns [batch_size, seq_len]

        Returns:
            policy_loss: PPO policy loss
            value_loss: Value function loss
        """
        # Policy loss (PPO clipping)
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
            * advantages
        )

        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        value_pred_clipped = values + torch.clamp(
            values - returns, -self.config.clip_epsilon, self.config.clip_epsilon
        )
        value_loss1 = F.mse_loss(values, returns)
        value_loss2 = F.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss1, value_loss2).mean()

        return policy_loss, value_loss

    def kl_penalty(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence penalty.

        Args:
            log_probs: Current log probabilities
            ref_log_probs: Reference model log probabilities

        Returns:
            KL divergence
        """
        return F.kl_div(log_probs, ref_log_probs, reduction="batchmean")

    def step(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one PPO update step.

        Args:
            prompts: Prompt tokens [batch_size, prompt_len]
            responses: Response tokens [batch_size, response_len]
            rewards: Rewards for each token [batch_size, response_len]

        Returns:
            Dictionary of loss metrics
        """
        # Get old log probs and values (before update)
        with torch.no_grad():
            old_log_probs = self._get_log_probs(prompts, responses)
            old_values = self._get_values(responses)

        # Compute advantages
        next_values = torch.zeros(responses.size(0)).to(self.device)
        advantages, returns = self.compute_advantages(rewards, old_values, next_values)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates
        for _ in range(self.config.ppo_epochs):
            # Get new log probs and values
            log_probs = self._get_log_probs(prompts, responses)
            values = self._get_values(responses)

            # Compute losses
            policy_loss, value_loss = self.ppo_loss(
                log_probs, old_log_probs, advantages, values, returns
            )

            # KL penalty if using reference model
            kl_loss = 0.0
            if self.config.use_ref_model and self.ref_model is not None:
                with torch.no_grad():
                    ref_log_probs = self._get_ref_log_probs(prompts, responses)
                kl_loss = self.kl_penalty(log_probs, ref_log_probs)

            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * self._entropy(responses)
                + kl_loss
            )

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "reward": rewards.mean().item(),
        }

    def _get_log_probs(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probabilities for responses."""
        # Simplified - in practice would need proper masking
        inputs = torch.cat([prompts, responses], dim=1)
        outputs = self.policy(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Get logits only
        log_probs = F.log_softmax(outputs, dim=-1)

        # Get probs for actual response tokens
        response_start = prompts.size(1)
        response_log_probs = log_probs[:, response_start - 1 : -1, :]

        # Gather the actual token probabilities
        token_indices = responses.unsqueeze(-1)
        token_log_probs = response_log_probs.gather(-1, token_indices).squeeze(-1)

        return token_log_probs

    def _get_ref_log_probs(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Get reference model log probabilities."""
        inputs = torch.cat([prompts, responses], dim=1)
        with torch.no_grad():
            outputs = self.ref_model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            log_probs = F.log_softmax(outputs, dim=-1)

        response_start = prompts.size(1)
        response_log_probs = log_probs[:, response_start - 1 : -1, :]

        token_indices = responses.unsqueeze(-1)
        token_log_probs = response_log_probs.gather(-1, token_indices).squeeze(-1)

        return token_log_probs

    def _get_values(self, responses: torch.Tensor) -> torch.Tensor:
        """Get value estimates for responses."""
        values = self.value(responses)
        if isinstance(values, tuple):
            values = values[0]
        return values.squeeze(-1)

    def _entropy(self, responses: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy."""
        inputs = responses
        outputs = self.policy(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = F.softmax(outputs, dim=-1)
        entropy = -(probs * outputs).sum(dim=-1).mean()
        return entropy


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
) -> PPOTrainer:
    """
    Create an RLHF trainer.

    Args:
        policy_model: The model to train
        value_model: Value function (can be same as policy)
        ref_model: Reference model for KL penalty
        config: RLHF configuration
        device: Device to use

    Returns:
        PPOTrainer instance
    """
    if value_model is None:
        # Use policy as value model (common approach)
        value_model = policy_model

    config = config or RLHFConfig()
    trainer = PPOTrainer(policy_model, value_model, config, device)

    if ref_model is not None and config.use_ref_model:
        trainer.set_ref_model(ref_model)

    return trainer


__all__ = [
    "RLHFConfig",
    "RLHFMetric",
    "PPOTrainer",
    "RewardModel",
    "create_rlhf_trainer",
]
