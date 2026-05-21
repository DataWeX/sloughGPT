"""
Unified Training Pipeline

Integrates:
1. Deep Learning - SloughGPTModel pre-training
2. Federated Learning - Privacy-preserving distributed fine-tuning
3. RLHF/PPO - Alignment with human preferences

These are STAGES, not competing approaches:
  Pre-training → Federated Fine-tune → RLHF Alignment
"""

import asyncio
from domains.training.slonet_compat import torch
nn = torch.nn
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("sloughgpt.unified_training")


class TrainingStage(Enum):
    PRETRAINING = "pretraining"
    FEDERATED = "federated"
    RLHF = "rlhf"
    COMPLETE = "complete"


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified training pipeline."""

    # Stage 1: Pre-training
    pretrain_epochs: int = 10
    pretrain_lr: float = 1e-4
    pretrain_batch_size: int = 32

    # Stage 2: Federated Learning
    federated_rounds: int = 5
    federated_clients: int = 3
    federated_fraction: float = 0.5  # Fraction of clients per round
    federated_lr: float = 5e-5

    # Stage 3: RLHF Alignment
    rlhf_epochs: int = 4
    rlhf_lr: float = 1e-5
    ppo_clip_epsilon: float = 0.2
    kl_penalty_coef: float = 0.1

    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    save_checkpoint_every: int = 1000


@dataclass
class TrainingProgress:
    """Tracks progress through training stages."""
    stage: TrainingStage = TrainingStage.PRETRAINING
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    stage_completed: bool = False


class UnifiedTrainingPipeline:
    """
    Unified training pipeline combining:
    - Deep Learning (pre-training)
    - Federated Learning (privacy-preserving fine-tuning)
    - RLHF/PPO (alignment)

    The key: Each stage builds on the previous, preserving learned knowledge.
    """

    def __init__(
        self,
        model: nn.Module,
        config: UnifiedTrainingConfig,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        self.progress = TrainingProgress()

        # Sub-components
        self.federated_trainer = None
        self.rlhf_trainer = None

        # Optimizers
        self.optimizer = None
        self.scheduler = None

    def setup(self):
        """Initialize training components."""
        logger.info("Setting up unified training pipeline...")

        # Initialize optimizer for pre-training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=0.01,
        )

        # Initialize federated trainer
        self._setup_federated()

        # Initialize RLHF trainer
        self._setup_rlhf()

        logger.info("Unified training pipeline ready")
        return self

    def _setup_federated(self):
        """Setup federated learning components."""
        try:
            from domains.training.federated_learning import FederatedTrainer
            self.federated_trainer = FederatedTrainer(
                model=self.model,
                num_clients=self.config.federated_clients,
                device=self.device,
            )
            logger.info("Federated learning enabled")
        except Exception as e:
            logger.warning(f"Federated learning not available: {e}")
            self.federated_trainer = None

    def _setup_rlhf(self):
        """Setup RLHF components."""
        try:
            from domains.training.rlhf import PPOTrainer, RewardModel, RLHFConfig

            # Create reward model (clone of base model + reward head)
            self.reward_model = RewardModel(
                base_model=self.model,
                hidden_size=self._get_hidden_size(),
            ).to(self.device)

            # Create PPO trainer
            rlhf_config = RLHFConfig(
                ppo_epochs=self.config.rlhf_epochs,
                clip_epsilon=self.config.ppo_clip_epsilon,
                use_ref_model=True,
            )

            self.rlhf_trainer = PPOTrainer(
                policy_model=self.model,
                value_model=self.reward_model,
                config=rlhf_config,
                device=self.device,
            )

            # Set reference model (copy of original policy)
            self.rlhf_trainer.set_ref_model(self._create_ref_model())

            logger.info("RLHF/PPO enabled")
        except Exception as e:
            logger.warning(f"RLHF not available: {e}")
            self.rlhf_trainer = None

    def _get_hidden_size(self) -> int:
        """Get model's hidden size."""
        if hasattr(self.model, 'n_embed'):
            return self.model.n_embed
        elif hasattr(self.model, 'config'):
            return getattr(self.model.config, 'hidden_size', 512)
        return 512

    def _create_ref_model(self) -> nn.Module:
        """Create reference model for KL divergence."""
        import copy
        ref = copy.deepcopy(self.model)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False
        return ref

    async def train_full_pipeline(
        self,
        train_data,
        val_data=None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline:
        1. Pre-training (Deep Learning)
        2. Federated Fine-tuning
        3. RLHF Alignment
        """
        results = {}

        # Stage 1: Pre-training
        logger.info("=" * 60)
        logger.info("STAGE 1: PRE-TRAINING (Deep Learning)")
        logger.info("=" * 60)

        self.progress.stage = TrainingStage.PRETRAINING
        self.progress.total_epochs = self.config.pretrain_epochs

        pretrain_results = await self._pretrain(train_data, val_data, progress_callback)
        results["pretraining"] = pretrain_results

        # Save checkpoint after pre-training
        self._save_checkpoint("pretrain_checkpoint.pt")

        # Stage 2: Federated Learning
        if self.federated_trainer and self.config.federated_rounds > 0:
            logger.info("=" * 60)
            logger.info("STAGE 2: FEDERATED FINE-TUNING")
            logger.info("=" * 60)

            self.progress.stage = TrainingStage.FEDERATED
            self.progress.total_epochs = self.config.federated_rounds

            federated_results = await self._federated_train(train_data, progress_callback)
            results["federated"] = federated_results

            # Save checkpoint after federated
            self._save_checkpoint("federated_checkpoint.pt")

        # Stage 3: RLHF Alignment
        if self.rlhf_trainer and self.config.rlhf_epochs > 0:
            logger.info("=" * 60)
            logger.info("STAGE 3: RLHF ALIGNMENT (PPO)")
            logger.info("=" * 60)

            self.progress.stage = TrainingStage.RLHF
            self.progress.total_epochs = self.config.rlhf_epochs

            rlhf_results = await self._rlhf_train(train_data, progress_callback)
            results["rlhf"] = rlhf_results

            # Save final model
            self._save_checkpoint("final_model.pt")

        self.progress.stage = TrainingStage.COMPLETE
        self.progress.stage_completed = True

        return results

    async def _pretrain(
        self,
        train_data,
        val_data,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Stage 1: Pre-training with standard deep learning."""
        losses = []

        for epoch in range(self.config.pretrain_epochs):
            self.progress.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            self.model.train()
            for batch in train_data:
                # Forward pass
                input_ids = batch.to(self.device)
                logits, loss = self.model(input_ids, input_ids)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            self.progress.loss = avg_loss

            logger.info(f"Pre-train Epoch {epoch+1}/{self.config.pretrain_epochs} - Loss: {avg_loss:.4f}")

            if progress_callback:
                await progress_callback(self.progress)

        return {"losses": losses, "final_loss": losses[-1] if losses else 0}

    async def _federated_train(
        self,
        train_data,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Stage 2: Federated learning for privacy-preserving fine-tuning."""
        losses = []

        for round_num in range(self.config.federated_rounds):
            self.progress.epoch = round_num

            # Simulate federated round
            round_loss = await self.federated_trainer.train_round(
                train_data,
                fraction=self.config.federated_fraction,
            )

            losses.append(round_loss)
            self.progress.loss = round_loss

            logger.info(f"Federated Round {round_num+1}/{self.config.federated_rounds} - Loss: {round_loss:.4f}")

            if progress_callback:
                await progress_callback(self.progress)

        return {"losses": losses, "final_loss": losses[-1] if losses else 0}

    async def _rlhf_train(
        self,
        preference_data,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Stage 3: RLHF/PPO alignment."""
        metrics = {
            "rewards": [],
            "kl_divergence": [],
            "policy_loss": [],
            "value_loss": [],
        }

        for epoch in range(self.config.rlhf_epochs):
            self.progress.epoch = epoch

            # PPO update step
            if hasattr(preference_data, '__iter__'):
                batch = next(iter(preference_data))
                ppo_metrics = await self._ppo_step(batch)
            else:
                ppo_metrics = await self._ppo_step(preference_data)

            for key, value in ppo_metrics.items():
                if key in metrics:
                    metrics[key].append(value)

            self.progress.metrics = ppo_metrics

            logger.info(f"RLHF Epoch {epoch+1}/{self.config.rlhf_epochs} - Reward: {ppo_metrics.get('reward', 0):.4f}")

            if progress_callback:
                await progress_callback(self.progress)

        return metrics

    async def _ppo_step(self, batch) -> Dict[str, float]:
        """Perform one PPO update."""
        try:
            # Generate responses (simplified)
            input_ids = batch.to(self.device) if hasattr(batch, 'to') else batch

            # Get log probs and values
            with torch.no_grad():
                old_log_probs = self.model(input_ids)[0]

            # Simulate rewards (in real RLHF, this comes from RewardModel)
            rewards = torch.randn(input_ids.size(0), device=self.device)

            # Get value estimates
            values = self.reward_model(input_ids)

            # Compute advantages
            advantages, returns = self.rlhf_trainer.compute_advantages(
                rewards.unsqueeze(1),
                values.unsqueeze(1),
                values[:, -1] if values.size(1) > 1 else values,
            )

            # PPO loss
            log_probs = self.model(input_ids)[0]
            policy_loss, value_loss = self.rlhf_trainer.ppo_loss(
                log_probs,
                old_log_probs,
                advantages,
                values.unsqueeze(1),
                returns,
            )

            # Update
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            return {
                "reward": rewards.mean().item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "kl_divergence": 0.0,  # Would compute from ref model
            }
        except Exception as e:
            logger.warning(f"PPO step failed: {e}")
            return {"reward": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl_divergence": 0.0}

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "progress": {
                    "stage": self.progress.stage.value,
                    "epoch": self.progress.epoch,
                }
            }
            if self.reward_model:
                checkpoint["reward_model_state_dict"] = self.reward_model.state_dict()

            torch.save(checkpoint, filename)
            logger.info(f"Checkpoint saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "reward_model_state_dict" in checkpoint and self.reward_model:
                self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])

            logger.info(f"Checkpoint loaded: {filename}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False

    def get_progress(self) -> TrainingProgress:
        """Get current training progress."""
        return self.progress


# =============================================================================
# FEDERATED RL (Privacy-Preserving RL)
# =============================================================================

class FederatedRLTrainer:
    """
    Combines Federated Learning with RLHF for privacy-preserving alignment.

    Clients train locally on their preferences, then share only gradients
    (not raw data) for aggregation.
    """

    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.global_model = model
        self.num_clients = num_clients
        self.device = device
        self.client_models = []

        # Initialize client models
        self._init_clients()

    def _init_clients(self):
        """Initialize client models."""
        import copy
        for _ in range(self.num_clients):
            client = copy.deepcopy(self.global_model)
            client.train()
            self.client_models.append(client)

    async def federated_rl_round(
        self,
        client_preferences: List[Dict],
        fraction: float = 0.5,
    ) -> Dict[str, float]:
        """
        Perform one round of federated RL.

        Args:
            client_preferences: List of preference data per client
            fraction: Fraction of clients to sample

        Returns:
            Aggregated metrics
        """
        num_sampled = max(1, int(self.num_clients * fraction))
        sampled_clients = self.client_models[:num_sampled]

        client_metrics = []

        # Local RL updates
        for i, (client, prefs) in enumerate(zip(sampled_clients, client_preferences[:num_sampled])):
            metrics = await self._local_rl_update(client, prefs)
            client_metrics.append(metrics)

        # Aggregate
        aggregated = self._aggregate_metrics(client_metrics)

        # Update global model
        self._update_global_model()

        return aggregated

    async def _local_rl_update(self, client_model, preferences) -> Dict[str, float]:
        """Local RL update on client."""
        # Simplified - would use actual PPO here
        return {"reward": 0.0, "loss": 0.0}

    def _aggregate_metrics(self, client_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics from clients."""
        aggregated = {}
        for key in client_metrics[0].keys():
            values = [m[key] for m in client_metrics]
            aggregated[key] = sum(values) / len(values)
        return aggregated

    def _update_global_model(self):
        """Update global model from client models."""
        import copy

        # Simple averaging (FedAvg)
        with torch.no_grad():
            for param, *client_params in zip(
                self.global_model.parameters(),
                *[m.parameters() for m in self.client_models]
            ):
                avg_param = sum(p.data for p in client_params) / len(client_params)
                param.data.copy_(avg_param)


__all__ = [
    "UnifiedTrainingConfig",
    "UnifiedTrainingPipeline",
    "TrainingProgress",
    "TrainingStage",
    "FederatedRLTrainer",
]
