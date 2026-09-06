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
    target_kl: float = 0.02  # Early stopping KL threshold
    kl_coef: float = 0.1  # KL penalty coefficient

    # Model parameters
    reward_model_path: Optional[str] = None
    ref_model_path: Optional[str] = None
    use_ref_model: bool = True

    # Generation
    gen_max_length: int = 512
    gen_temperature: float = 1.0
    gen_top_p: float = 0.9

    # Rollout
    rollout_steps: int = 2048
    learning_rate: float = 3e-4


def _as_array(x) -> np.ndarray:
    """Coerce a SloNet Tensor (or ndarray) into its underlying numpy buffer."""
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(x)


def _get_logprobs(model, input_ids: np.ndarray) -> np.ndarray:
    """Run a forward pass and return per-token log-probabilities [batch, seq_len].

    Handles models that return ``(logits, loss)`` tuples or raw logits.
    """
    out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    arr = _as_array(logits)
    # arr shape: [batch, seq_len, vocab]
    logprobs = arr - np.log(np.exp(arr).sum(axis=-1, keepdims=True) + 1e-8)
    return logprobs


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: [T] immediate rewards
        values: [T+1] value estimates (V(s_0) ... V(s_T))
        dones: [T] boolean done flags
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: [T] advantage estimates
        returns: [T] discounted returns (targets for value function)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = float(values[t + 1]) if t + 1 < len(values) else 0.0
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


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


class ValueHead:
    """Scalar value head that can be attached to a base model.

    Wraps a single ``SloLinear`` projecting the last hidden state to a scalar
    value estimate V(s).
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.head: Optional[SloLinear] = None
        self._feature_dim: Optional[int] = None

    def _ensure_head(self, feature_dim: int):
        if self._feature_dim == feature_dim and self.head is not None:
            return
        self.head = SloLinear(feature_dim, 1)
        self._feature_dim = feature_dim

    def forward(self, input_ids) -> Tensor:
        out = self.base_model(input_ids)
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        arr = _as_array(hidden)
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        last_hidden = arr[:, -1, :]
        self._ensure_head(last_hidden.shape[-1])
        value = self.head(last_hidden)
        return value.squeeze(-1)

    def __call__(self, input_ids):
        return self.forward(input_ids)


class PPOTrainer:
    """Proximal Policy Optimization trainer for SloughGPT.

    Operates entirely in numpy/SloNet autograd.  The typical lifecycle is:

    1. Collect *rollouts* by querying the policy and reward/value models.
    2. Call :meth:`update` to run PPO gradient steps on the collected data.
    3. Repeat until convergence.

    The trainer keeps a frozen copy of the policy (``ref_model``) for the KL
    divergence penalty unless ``config.use_ref_model`` is ``False``.
    """

    def __init__(
        self,
        policy_model,
        reward_model: RewardModel,
        value_model: Optional[object] = None,
        ref_model: Optional[object] = None,
        config: Optional[RLHFConfig] = None,
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.value_head = ValueHead(value_model or policy_model)
        self.ref_model = ref_model
        self.config = config or RLHFConfig()
        self._optimizer = None

    # ── Rollout helpers ────────────────────────────────────────────────

    def collect_rollout(
        self,
        prompts: np.ndarray,
        response_ids: np.ndarray,
        advantages: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
    ) -> dict:
        """Package transition data into a rollout dict for :meth:`update`.

        If *advantages* / *returns* are ``None`` they are computed from the
        reward and value signals already embedded in *response_ids* via
        :func:`_compute_gae`.
        """
        T = response_ids.shape[1]
        logprobs = _get_logprobs(self.policy, response_ids)
        values = _as_array(self.value_head(response_ids)).astype(np.float32)
        rewards = _as_array(self.reward_model(response_ids)).astype(np.float32)

        if advantages is None:
            # values shape: [B] or [B, T]. Extend with a bootstrap zero for GAE.
            if values.ndim == 1:
                values_1d = values.astype(np.float32)
            else:
                values_1d = values.mean(axis=0).astype(np.float32)
            if rewards.ndim == 1:
                rewards_1d = rewards.astype(np.float32)
            else:
                rewards_1d = rewards.mean(axis=0).astype(np.float32)
            # Ensure per-timestep arrays: if model returns per-sequence scalars,
            # place the reward at the last timestep and broadcast values.
            if len(rewards_1d) != T:
                rewards_per_t = np.zeros(T, dtype=np.float32)
                rewards_per_t[-1] = float(rewards_1d.mean())
            else:
                rewards_per_t = rewards_1d
            if len(values_1d) != T:
                values_per_t = np.full(T, float(values_1d.mean()), dtype=np.float32)
            else:
                values_per_t = values_1d
            # Append bootstrap zero for V(s_{T})
            values_ext = np.concatenate([values_per_t, [0.0]])
            dones = np.zeros(T, dtype=np.float32)
            advantages, returns = _compute_gae(
                rewards_per_t,
                values_ext,
                dones,
                self.config.gamma,
                self.config.lam,
            )

        return {
            "obs": response_ids,
            "logprobs": logprobs,
            "values": values,
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
            "rewards": rewards,
        }

    # ── PPO update ─────────────────────────────────────────────────────

    def update(self, rollout: dict) -> dict:
        """Run ``ppo_epochs`` of PPO on the collected rollout.

        Returns a dict of training metrics (mean across mini-batches).
        """
        cfg = self.config
        obs = rollout["obs"]           # [B, T]
        old_logprobs = rollout["logprobs"]  # [B, T, V]
        rollout["values"]      # [B]
        advantages = rollout["advantages"]   # [T]
        returns = rollout["returns"]         # [T]
        rewards = rollout["rewards"]         # [B]

        B, T = obs.shape
        vocab_size = old_logprobs.shape[-1]
        clip_eps = cfg.clip_epsilon
        n_mini = min(cfg.num_mini_batches, B)

        # Normalize advantages across the batch
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std

        all_policy_loss = []
        all_value_loss = []
        all_entropy = []
        all_kl = []
        all_approx_kl = []
        early_stopped = False

        for epoch in range(cfg.ppo_epochs):
            # Shuffle indices for mini-batching
            indices = np.random.permutation(B)
            mini_size = B // n_mini

            for i in range(n_mini):
                start = i * mini_size
                end = start + mini_size if i < n_mini - 1 else B
                mb_idx = indices[start:end]
                mb_obs = obs[mb_idx]               # [mb, T]
                mb_old_lp = old_logprobs[mb_idx]   # [mb, T, V]
                # Advantages are per-timestep [T], broadcast across batch
                mb_adv = np.broadcast_to(advantages_norm, (mb_obs.shape[0], T))

                # ── Forward pass through current policy ──
                new_logprobs = _get_logprobs(self.policy, mb_obs)  # [mb, T, V]
                new_values = _as_array(self.value_head(mb_obs)).astype(np.float32)  # [mb]

                # Pick log-prob of the actual tokens taken
                vocab_size = new_logprobs.shape[-1]
                tokens = np.clip(mb_obs.astype(np.int64), 0, vocab_size - 1)
                B_mb, T_mb = tokens.shape
                arange_b = np.arange(B_mb)[:, None]
                arange_t = np.arange(T_mb)[None, :]

                old_lp_selected = mb_old_lp[arange_b, arange_t, tokens]  # [mb, T]
                new_lp_selected = new_logprobs[arange_b, arange_t, tokens]  # [mb, T]

                # ── KL divergence (for monitoring / early stopping) ──
                if self.ref_model is not None and cfg.use_ref_model:
                    ref_lp = _get_logprobs(self.ref_model, mb_obs)
                    ref_lp_selected = ref_lp[arange_b, arange_t, tokens]
                    kl_div = (old_lp_selected - ref_lp_selected).mean()
                else:
                    kl_div = (old_lp_selected - new_lp_selected).mean()

                all_kl.append(float(kl_div))

                # ── PPO clipped surrogate objective ──
                ratio = np.exp(new_lp_selected - old_lp_selected)  # [mb, T]

                surr1 = ratio * mb_adv
                surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -np.minimum(surr1, surr2).mean()

                # ── Value function loss ──
                # new_values is [mb], returns is [T]. Use mean return as target.
                value_target = float(returns.mean())
                value_loss = ((new_values - value_target) ** 2).mean()

                # ── Entropy bonus ──
                probs = np.exp(new_logprobs)
                entropy = -(probs * new_logprobs).sum(axis=-1).mean()

                # ── KL penalty on the loss ──
                kl_penalty = cfg.kl_coef * kl_div

                # ── Combined loss ──
                loss_val = (
                    float(policy_loss)
                    + cfg.value_loss_coef * float(value_loss)
                    - cfg.entropy_coef * float(entropy)
                    + float(kl_penalty)
                )
                loss = Tensor(np.float32(loss_val), requires_grad=True)

                # ── Backward + parameter update ──
                if self._optimizer is None:
                    self._init_optimizer()

                self._zero_grads()
                loss.backward()
                self._clip_grads(cfg.max_grad_norm)
                self._optimizer.step(self._collect_params())

                all_policy_loss.append(float(policy_loss))
                all_value_loss.append(float(value_loss))
                all_entropy.append(float(entropy))
                all_approx_kl.append(float(kl_div))

            # ── Early stopping on KL ──
            mean_kl = np.mean(all_kl[-n_mini:])
            if cfg.target_kl > 0 and mean_kl > cfg.target_kl:
                logger.info(
                    "PPO early stop at epoch %d/%d — KL %.4f > target %.4f",
                    epoch + 1,
                    cfg.ppo_epochs,
                    mean_kl,
                    cfg.target_kl,
                )
                early_stopped = True
                break

        metrics = {
            "policy_loss": float(np.mean(all_policy_loss)) if all_policy_loss else 0.0,
            "value_loss": float(np.mean(all_value_loss)) if all_value_loss else 0.0,
            "entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
            "kl_divergence": float(np.mean(all_kl)) if all_kl else 0.0,
            "approx_kl": float(np.mean(all_approx_kl)) if all_approx_kl else 0.0,
            "mean_reward": float(rewards.mean()),
            "epochs_run": epoch + 1 if all_policy_loss else 0,
            "early_stopped": early_stopped,
        }
        logger.info(
            "PPO update — policy_loss=%.4f  value_loss=%.4f  entropy=%.4f  kl=%.4f  reward=%.4f",
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
            metrics["kl_divergence"],
            metrics["mean_reward"],
        )
        return metrics

    # ── Optimizer plumbing ─────────────────────────────────────────────

    def _collect_params(self) -> list:
        """Collect all trainable parameters from the policy and value head."""
        params = []
        for module in (self.policy, self.value_head):
            for attr_name in dir(module):
                attr = getattr(module, attr_name, None)
                if attr is None:
                    continue
                if hasattr(attr, "data") and hasattr(attr, "grad"):
                    params.append(attr)
                # Check one level deeper (e.g. model.layers[i].weight)
                if hasattr(attr, "__iter__") and not isinstance(attr, (str, bytes)):
                    try:
                        for sub in attr:
                            if hasattr(sub, "data") and hasattr(sub, "grad"):
                                params.append(sub)
                            if hasattr(sub, "weight") and hasattr(sub.weight, "data"):
                                params.append(sub.weight)
                            if hasattr(sub, "bias") and sub.bias is not None and hasattr(sub.bias, "data"):
                                params.append(sub.bias)
                    except TypeError:
                        pass
                if hasattr(attr, "weight") and hasattr(attr.weight, "data"):
                    params.append(attr.weight)
                if hasattr(attr, "bias") and attr.bias is not None and hasattr(attr.bias, "data"):
                    params.append(attr.bias)
        # Deduplicate by identity
        seen = set()
        unique = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    def _init_optimizer(self):
        from domains.training.slonet import SloAdam
        self._optimizer = SloAdam(lr=self.config.learning_rate)

    def _zero_grads(self):
        for p in self._collect_params():
            p.grad = None

    def _clip_grads(self, max_norm: float):
        params = self._collect_params()
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += float(np.sum(p.grad.data ** 2))
        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for p in params:
                if p.grad is not None:
                    p.grad = Tensor(p.grad.data * scale, _copy=False)


def create_rlhf_trainer(
    policy_model=None,
    value_model: Optional[object] = None,
    ref_model: Optional[object] = None,
    config: Optional[RLHFConfig] = None,
    device: str = "cpu",
):
    """
    Create an RLHF PPO trainer.

    Args:
        policy_model: The policy model to train (SloTransformer or compatible)
        value_model: Value function model (defaults to sharing policy weights)
        ref_model: Frozen reference model for KL penalty
        config: RLHF configuration
        device: Device to use (reserved for future GPU support)

    Returns:
        PPOTrainer ready for rollout collection and updates
    """
    config = config or RLHFConfig()
    reward_model = RewardModel(policy_model)
    trainer = PPOTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        value_model=value_model or policy_model,
        ref_model=ref_model,
        config=config,
    )
    return trainer


__all__ = [
    "RLHFConfig",
    "RLHFMetric",
    "RewardModel",
    "ValueHead",
    "PPOTrainer",
    "create_rlhf_trainer",
]
