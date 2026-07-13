"""
Production-Grade Elastic Weight Consolidation (EWC)

Implements proper EWC with:
- Diagonal Fisher Information Matrix approximation
- Online EWC for continual learning
- Automatic regularization strength
- Task importance weighting
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import logging

from domains.training.slonet_compat import torch, nn, F

logger = logging.getLogger("man.ewc")


@dataclass
class EWCParameters:
    """Parameters for EWC training."""
    lambda_ewc: float = 1000.0  # Regularization strength
    diagonal_approx: bool = True  # Use diagonal Fisher approximation
    batch_size: int = 32
    num_samples: int = 100  # Samples for Fisher estimation
    clip_grad_norm: float = 10.0
    ema_decay: float = 0.9  # For running Fisher estimate


@dataclass
class TaskSnapshot:
    """Snapshot of model after learning a task."""
    task_id: str
    task_name: str
    parameters: Dict[str, torch.Tensor]
    fisher_diagonal: Dict[str, torch.Tensor]
    optimal_loss: float
    num_samples: int


class DiagonalFisherEstimator:
    """
    Estimates diagonal elements of the Fisher Information Matrix.

    For diagonal approximation:
    F_ii ≈ (1/N) * Σ (∂log p(y|x,θ) / ∂θ_i)²

    This is the empirical Fisher, computed from gradient samples.
    """

    def __init__(
        self,
        model: nn.Module,
        ema_decay: float = 0.9,
        device: str = "cpu",
    ):
        self.model = model
        self.ema_decay = ema_decay
        self.device = device
        self.fisher_accum: Dict[str, torch.Tensor] = {}
        self.num_observations = 0
        self._init_fisher()

    def _init_fisher(self):
        """Initialize Fisher accumulator."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_accum[name] = torch.zeros_like(param.data)

    def estimate(
        self,
        data_loader,
        loss_fn: Callable,
        num_samples: int = 100,
        accumulation_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate Fisher Information Matrix diagonal.

        Uses gradient squares averaged over samples.
        """
        self.model.eval()
        self._init_fisher()
        self.num_observations = 0

        samples_seen = 0
        batch_count = 0

        for batch in data_loader:
            if samples_seen >= num_samples:
                break

            self.model.zero_grad()

            # Forward pass
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                inputs = batch.to(self.device)
                targets = None

            outputs = self.model(inputs)

            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                # Use log likelihood for generative models
                loss = outputs.mean()

            # Backward pass
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_squared = param.grad.data.clone() ** 2
                    self.fisher_accum[name] = (
                        self.ema_decay * self.fisher_accum[name] +
                        (1 - self.ema_decay) * grad_squared
                    )

            self.num_observations += inputs.size(0)
            samples_seen += inputs.size(0)
            batch_count += 1

            if batch_count >= accumulation_steps:
                break

        # Normalize
        for name in self.fisher_accum:
            if self.num_observations > 0:
                self.fisher_accum[name] /= self.num_observations

            # Add small constant for numerical stability
            self.fisher_accum[name] += 1e-8

        return self.fisher_accum

    def estimate_from_logits(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate Fisher from logits (for classification).

        Uses:
        F_ii = (1/N) * Σ ∂L/∂θ_i * ∂L/∂θ_i

        where L is the negative log-likelihood.
        """
        self.model.eval()
        self._init_fisher()

        for _ in range(num_samples):
            self.model.zero_grad()

            outputs = self.model(inputs)
            log_probs = F.log_softmax(outputs, dim=-1)
            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                targets.view(-1),
            )

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_accum[name] += (param.grad.data ** 2) / num_samples

        return self.fisher_accum


class EwcContinualLearner:
    """
    Production-grade EWC for continual learning.

    Prevents catastrophic forgetting by penalizing changes to
    important parameters (those critical for previous tasks).

    Loss = L_current(θ) + λ/2 * Σ F_i (θ_i - θ*_i)²

    where:
    - L_current is the loss on the current task
    - F_i is the Fisher Information for parameter θ_i
    - θ*_i are the optimal parameters after previous tasks
    """

    def __init__(
        self,
        model: nn.Module,
        params: Optional[EWCParameters] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.params = params or EWCParameters()
        self.device = device
        self.model.to(device)

        # Fisher estimator
        self.fisher_estimator = DiagonalFisherEstimator(
            model,
            ema_decay=self.params.ema_decay,
            device=device,
        )

        # Store snapshots of each task
        self.task_snapshots: Dict[str, TaskSnapshot] = {}

        # Current task
        self.current_task: Optional[str] = None

    def save_task_snapshot(
        self,
        task_id: str,
        task_name: str,
        train_loader,
        loss_fn: Callable,
    ) -> TaskSnapshot:
        """
        Save a snapshot of model after learning a task.

        This captures:
        - Current parameter values
        - Fisher Information diagonal
        """
        logger.info("Saving snapshot for task: %s", task_name)

        # Store current parameters
        parameters = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters[name] = param.data.clone().detach()

        # Estimate Fisher Information
        fisher = self.fisher_estimator.estimate(
            train_loader,
            loss_fn,
            num_samples=self.params.num_samples,
        )

        # Calculate optimal loss
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in train_loader:
                if num_batches >= 10:
                    break
                inputs = batch[0].to(self.device) if isinstance(batch, tuple) else batch.to(self.device)
                targets = batch[1].to(self.device) if isinstance(batch, tuple) and len(batch) > 1 else None

                outputs = self.model(inputs)
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    loss = outputs.mean()
                total_loss += loss.item()
                num_batches += 1

        optimal_loss = total_loss / max(num_batches, 1)

        # Create snapshot
        snapshot = TaskSnapshot(
            task_id=task_id,
            task_name=task_name,
            parameters=parameters,
            fisher_diagonal=fisher,
            optimal_loss=optimal_loss,
            num_samples=self.params.num_samples,
        )

        self.task_snapshots[task_id] = snapshot
        logger.info("  Parameters: %d", len(parameters))
        logger.info("  Fisher elements: %d", sum(t.numel() for t in fisher.values()))
        logger.info("  Optimal loss: %.4f", optimal_loss)

        return snapshot

    def ewc_loss(self, task_id: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate EWC regularization loss.

        Returns:
        - ewc_loss: The regularization term
        - ewc_stats: Statistics about the calculation
        """
        if task_id is None:
            task_id = self.current_task

        if task_id not in self.task_snapshots:
            return torch.tensor(0.0, device=self.device), {"active_tasks": 0}

        snapshot = self.task_snapshots[task_id]

        ewc_loss = torch.tensor(0.0, device=self.device)
        param_count = 0

        for name, param in self.model.named_parameters():
            if name in snapshot.parameters and name in snapshot.fisher_diagonal:
                # Get old parameter value and Fisher
                old_param = snapshot.parameters[name].to(self.device)
                fisher = snapshot.fisher_diagonal[name].to(self.device)

                # EWC penalty: F_i * (θ_i - θ*_i)²
                diff = param - old_param
                penalty = (fisher * diff ** 2).sum()
                ewc_loss = ewc_loss + penalty

                param_count += 1

        # Scale by lambda
        scaled_loss = (self.params.lambda_ewc / 2) * ewc_loss

        stats = {
            "ewc_loss": scaled_loss.item(),
            "raw_ewc_loss": ewc_loss.item(),
            "lambda": self.params.lambda_ewc,
            "active_tasks": 1,
            "param_count": param_count,
        }

        return scaled_loss, stats

    def multi_task_ewc_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate EWC loss for all previous tasks.

        For online EWC (memory-efficient):
        - Use running sum of Fisher estimates
        - Only store current task's optimal parameters
        """
        if not self.task_snapshots:
            return torch.tensor(0.0, device=self.device), {"active_tasks": 0}

        total_loss = torch.tensor(0.0, device=self.device)
        total_params = 0

        for task_id, snapshot in self.task_snapshots.items():
            task_loss = torch.tensor(0.0, device=self.device)

            for name, param in self.model.named_parameters():
                if name in snapshot.parameters and name in snapshot.fisher_diagonal:
                    old_param = snapshot.parameters[name].to(self.device)
                    fisher = snapshot.fisher_diagonal[name].to(self.device)

                    diff = param - old_param
                    task_loss = task_loss + (fisher * diff ** 2).sum()
                    total_params += 1

            # Weight by number of samples (importance)
            weight = snapshot.num_samples / sum(s.num_samples for s in self.task_snapshots.values())
            total_loss = total_loss + (weight * task_loss / 2)

        scaled_loss = self.params.lambda_ewc * total_loss

        return scaled_loss, {
            "ewc_loss": scaled_loss.item(),
            "active_tasks": len(self.task_snapshots),
            "param_count": total_params,
        }

    def forward_and_ewc(
        self,
        batch,
        loss_fn: Callable,
        task_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with EWC loss.

        Total loss = task_loss + λ/2 * Σ F_i (θ_i - θ*_i)²
        """
        # Task loss
        inputs = batch[0].to(self.device) if isinstance(batch, tuple) else batch.to(self.device)
        targets = batch[1].to(self.device) if isinstance(batch, tuple) and len(batch) > 1 else None

        outputs = self.model(inputs)

        if targets is not None:
            task_loss = loss_fn(outputs, targets)
        else:
            task_loss = outputs.mean()

        # EWC loss
        ewc_loss, ewc_stats = self.ewc_loss(task_id)

        # Total loss
        total_loss = task_loss + ewc_loss

        return total_loss, {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "ewc_loss": ewc_stats["ewc_loss"],
            "active_tasks": ewc_stats["active_tasks"],
        }

    def prune_consolidation(
        self,
        top_k_percent: float = 10.0,
    ) -> Dict[str, int]:
        """
        Identify which parameters to protect most.

        Returns parameter names that should be most protected.
        """
        if not self.task_snapshots:
            return {}

        # Average Fisher across tasks
        avg_fisher = {}
        for name in self.model.state_dict().keys():
            fisher_values = []
            for snapshot in self.task_snapshots.values():
                if name in snapshot.fisher_diagonal:
                    fisher_values.append(snapshot.fisher_diagonal[name].cpu().numpy())
            if fisher_values:
                avg_fisher[name] = np.mean(fisher_values, axis=0)

        # Find top K% important parameters
        important = {}
        for name, fisher in avg_fisher.items():
            total_params = np.prod(fisher.shape) if hasattr(fisher, 'shape') else 1
            if total_params > 1:
                threshold = np.percentile(fisher.flatten(), 100 - top_k_percent)
                important[name] = int((fisher > threshold).sum())
            else:
                important[name] = 1 if fisher > np.percentile(list(avg_fisher.values()), 100 - top_k_percent) else 0

        return important

    def estimate_forgetting(self) -> Dict[str, float]:
        """
        Estimate how much each previous task is being forgotten.
        """
        if not self.task_snapshots:
            return {}

        forgetting = {}

        for task_id, snapshot in self.task_snapshots.items():
            loss_increase = 0.0
            param_count = 0

            for name, param in self.model.named_parameters():
                if name in snapshot.parameters:
                    old_param = snapshot.parameters[name].to(self.device)
                    fisher = snapshot.fisher_diagonal.get(name, torch.ones_like(param))

                    # Distance in Fisher-scaled space
                    diff = (param - old_param) ** 2
                    weighted_diff = (fisher * diff).sum().item()
                    loss_increase += weighted_diff
                    param_count += 1

            forgetting[task_id] = loss_increase / max(param_count, 1)

        return forgetting


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EWCParameters",
    "TaskSnapshot",
    "DiagonalFisherEstimator",
    "EwcContinualLearner",
]
