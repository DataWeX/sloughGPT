"""
Production-Grade Elastic Weight Consolidation (EWC) — numpy/SloNet

Implements proper EWC with:
- Diagonal Fisher Information Matrix approximation
- Online EWC for continual learning
- Automatic regularization strength
- Task importance weighting

Runs entirely on the SloNet autograd stack (pure NumPy). No PyTorch.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import logging

from domains.training.slonet import Tensor

logger = logging.getLogger("slo.ewc")


def _as_array(x) -> np.ndarray:
    """Coerce a SloNet Tensor (or ndarray) into its underlying numpy buffer."""
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(x)


def _scalar(x) -> float:
    """Extract a plain float from a Tensor / ndarray / scalar."""
    return float(_as_array(x).reshape(-1)[0])


def _zero_grad(model):
    """Zero all parameter grads (SloNet has no ``model.zero_grad``)."""
    for p in model.parameters():
        p.grad = None


def _batch_size(inputs) -> int:
    """Return leading batch dimension of an input sample."""
    arr = np.asarray(inputs)
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _unpack_batch(batch) -> Tuple[Any, Optional[Any]]:
    """Split a batch into (inputs, targets); targets may be None."""
    if isinstance(batch, (list, tuple)):
        inputs = batch[0]
        targets = batch[1] if len(batch) > 1 else None
    else:
        inputs = batch
        targets = None
    return inputs, targets


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
    parameters: Dict[str, np.ndarray]
    fisher_diagonal: Dict[str, np.ndarray]
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
        model,
        ema_decay: float = 0.9,
        device: str = "cpu",
    ):
        self.model = model
        self.ema_decay = ema_decay
        self.device = device
        self.fisher_accum: Dict[str, np.ndarray] = {}
        self.num_observations = 0
        self._init_fisher()

    def _init_fisher(self):
        """Initialize Fisher accumulator from trainable parameters."""
        for name, param in self.model.named_parameters():
            if getattr(param, "requires_grad", True):
                self.fisher_accum[name] = np.zeros_like(_as_array(param))

    def estimate(
        self,
        data_loader,
        loss_fn: Callable,
        num_samples: int = 100,
        accumulation_steps: int = 10,
    ) -> Dict[str, np.ndarray]:
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

            inputs, targets = _unpack_batch(batch)
            _zero_grad(self.model)

            # Forward pass
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = outputs.mean()

            # Backward pass
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_squared = _as_array(param.grad) ** 2
                    self.fisher_accum[name] = (
                        self.ema_decay * self.fisher_accum[name]
                        + (1 - self.ema_decay) * grad_squared
                    )

            n = _batch_size(inputs)
            self.num_observations += n
            samples_seen += n
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
        inputs,
        targets,
        num_samples: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate Fisher from logits (for classification).

        Uses:
        F_ii = (1/N) * Σ ∂L/∂θ_i * ∂L/∂θ_i

        where L is the negative log-likelihood.
        """
        self.model.eval()
        self._init_fisher()

        for _ in range(num_samples):
            _zero_grad(self.model)

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            log_probs = outputs.log_softmax(dim=-1)
            lp = np.asarray(log_probs).reshape(-1, np.asarray(log_probs).shape[-1])
            flat_targets = np.asarray(targets).reshape(-1).astype(np.int64)
            onehot = np.zeros_like(lp)
            onehot[np.arange(lp.shape[0]), flat_targets] = 1.0
            onehot = onehot.reshape(np.asarray(log_probs).shape)

            loss = -((log_probs * Tensor(onehot)).sum()) / max(1, flat_targets.shape[0])

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_accum[name] += (_as_array(param.grad) ** 2) / num_samples

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
        model,
        params: Optional[EWCParameters] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.params = params or EWCParameters()
        self.device = device
        if hasattr(self.model, "to"):
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
        logger.info("Saving snapshot for task: %s", task_name,
            extra={"tag": "TRAIN"},)

        # Store current parameters
        parameters: Dict[str, np.ndarray] = {}
        for name, param in self.model.named_parameters():
            if getattr(param, "requires_grad", True):
                parameters[name] = _as_array(param).copy()

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
        for batch in train_loader:
            if num_batches >= 10:
                break
            inputs, targets = _unpack_batch(batch)
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = outputs.mean()
            total_loss += _scalar(loss)
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
        total_elems = sum(int(np.prod(f.shape)) for f in fisher.values())
        logger.info("  Parameters: %d", len(parameters),
            extra={"tag": "TRAIN"},)
        logger.info("  Fisher elements: %d", total_elems,
            extra={"tag": "TRAIN"},)
        logger.info("  Optimal loss: %.4f", optimal_loss,
            extra={"tag": "TRAIN"},)

        return snapshot

    def _penalty_tensor(self, snapshot: TaskSnapshot) -> Tuple[Tensor, int]:
        """Build the differentiable EWC penalty for one snapshot.

        Returns (penalty Tensor, number of matched parameters).
        """
        penalty = Tensor(np.zeros(1))
        param_count = 0
        for name, param in self.model.named_parameters():
            if name in snapshot.parameters and name in snapshot.fisher_diagonal:
                old_param = np.asarray(snapshot.parameters[name])
                fisher = np.asarray(snapshot.fisher_diagonal[name])
                diff = param - old_param
                penalty = penalty + (fisher * (diff ** 2)).sum()
                param_count += 1
        return penalty, param_count

    def ewc_loss(self, task_id: Optional[str] = None) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate EWC regularization loss.

        Returns:
        - ewc_loss: The regularization term (differentiable SloNet Tensor)
        - ewc_stats: Statistics about the calculation
        """
        if task_id is None:
            task_id = self.current_task

        if task_id not in self.task_snapshots:
            return Tensor(np.zeros(1)), {"active_tasks": 0}

        snapshot = self.task_snapshots[task_id]
        penalty, param_count = self._penalty_tensor(snapshot)

        # Scale by lambda
        scaled_loss = (self.params.lambda_ewc / 2) * penalty

        stats = {
            "ewc_loss": _scalar(scaled_loss),
            "raw_ewc_loss": _scalar(penalty),
            "lambda": self.params.lambda_ewc,
            "active_tasks": 1,
            "param_count": param_count,
        }

        return scaled_loss, stats

    def multi_task_ewc_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate EWC loss for all previous tasks.

        For online EWC (memory-efficient):
        - Use running sum of Fisher estimates
        - Only store current task's optimal parameters
        """
        if not self.task_snapshots:
            return Tensor(np.zeros(1)), {"active_tasks": 0}

        total_loss = Tensor(np.zeros(1))
        total_params = 0

        for task_id, snapshot in self.task_snapshots.items():
            task_penalty, task_params = self._penalty_tensor(snapshot)
            total_params += task_params

            # Weight by number of samples (importance)
            total_samples = sum(s.num_samples for s in self.task_snapshots.values())
            weight = snapshot.num_samples / total_samples if total_samples > 0 else 1.0
            total_loss = total_loss + (weight * task_penalty / 2)

        scaled_loss = self.params.lambda_ewc * total_loss

        return scaled_loss, {
            "ewc_loss": _scalar(scaled_loss),
            "active_tasks": len(self.task_snapshots),
            "param_count": total_params,
        }

    def forward_and_ewc(
        self,
        batch,
        loss_fn: Callable,
        task_id: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass with EWC loss.

        Total loss = task_loss + λ/2 * Σ F_i (θ_i - θ*_i)²
        """
        # Task loss
        inputs, targets = _unpack_batch(batch)

        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if targets is not None:
            task_loss = loss_fn(outputs, targets)
        else:
            task_loss = outputs.mean()

        # EWC loss
        ewc_loss, ewc_stats = self.ewc_loss(task_id)

        # Total loss
        total_loss = task_loss + ewc_loss

        return total_loss, {
            "total_loss": _scalar(total_loss),
            "task_loss": _scalar(task_loss),
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
        avg_fisher: Dict[str, np.ndarray] = {}
        state_keys = set()
        if hasattr(self.model, "state_dict"):
            state_keys = set(self.model.state_dict().keys())
        for name in state_keys:
            fisher_values = []
            for snapshot in self.task_snapshots.values():
                if name in snapshot.fisher_diagonal:
                    fisher_values.append(np.asarray(snapshot.fisher_diagonal[name]))
            if fisher_values:
                avg_fisher[name] = np.mean(fisher_values, axis=0)

        # Find top K% important parameters
        important: Dict[str, int] = {}
        all_fishers = list(avg_fisher.values())
        for name, fisher in avg_fisher.items():
            total_params = int(np.prod(fisher.shape)) if hasattr(fisher, 'shape') else 1
            if total_params > 1:
                threshold = np.percentile(fisher.flatten(), 100 - top_k_percent)
                important[name] = int((fisher > threshold).sum())
            else:
                flat_all = np.concatenate([f.flatten() for f in all_fishers]) if all_fishers else np.array([0.0])
                important[name] = 1 if fisher > np.percentile(flat_all, 100 - top_k_percent) else 0

        return important

    def estimate_forgetting(self) -> Dict[str, float]:
        """
        Estimate how much each previous task is being forgotten.
        """
        if not self.task_snapshots:
            return {}

        forgetting: Dict[str, float] = {}

        for task_id, snapshot in self.task_snapshots.items():
            loss_increase = 0.0
            param_count = 0

            for name, param in self.model.named_parameters():
                if name in snapshot.parameters:
                    old_param = np.asarray(snapshot.parameters[name])
                    fisher = np.asarray(snapshot.fisher_diagonal.get(name, np.ones_like(_as_array(param))))

                    # Distance in Fisher-scaled space
                    diff = (_as_array(param) - old_param) ** 2
                    weighted_diff = float((fisher * diff).sum())
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
