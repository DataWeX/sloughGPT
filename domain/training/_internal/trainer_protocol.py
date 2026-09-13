"""
Trainer protocol — standard interface for all training pipelines.

Every trainer should implement ``TrainerProtocol`` and return ``TrainResult``
from its ``train()`` method. This lets callers handle results uniformly
regardless of the underlying model type (HF, VLM, SloNet, etc.).

Usage::

    result = trainer.train()
    if result.success:
        print(f"Trained for {result.total_steps} steps, loss={result.final_loss}")
    else:
        print(f"Training failed: {result.error}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TrainerProtocol(Protocol):
    """Standard interface for all training pipelines.

    Every trainer — SloNet, HF, VLM, optimised — satisfies this protocol
    structurally (duck-typing) by implementing ``train() -> TrainResult``,
    ``is_training``, and ``stop()``.
    """

    def train(self, **kwargs: Any) -> "TrainResult":
        """Run training and return a ``TrainResult``."""
        ...

    @property
    def is_training(self) -> bool:
        """Whether training is currently in progress."""
        ...

    def stop(self) -> None:
        """Request early stopping of a running training job."""
        ...


@dataclass
class TrainResult:
    """Standard return type for all ``train()`` calls.

    Supports dict-like ``.get()`` for backward compatibility with code that
    accesses fields as ``result["final_loss"]`` or ``result.get("best_eval_loss")``.

    Attributes:
        success: Whether training completed without fatal error.
        status: Human-readable status string (``"completed"``, ``"no_data"``, etc.).
        final_loss: Loss at the final step (or None if unavailable).
        best_eval_loss: Best evaluation loss (SloughGPTTrainer compat).
        global_step: Final global step count.
        total_steps: Total training steps completed.
        epochs_completed: Number of full epochs.
        model_path: Path where the trained model was saved (if saved).
        checkpoint_name: Name of the saved checkpoint (if any).
        method: Training method used (``"hf"``, ``"slonet"``, ``"nanogpt"``, etc.).
        metrics: Additional metrics dict (perplexity, BLEU, accuracy, etc.).
        avg_quality: Average quality score of training data (0-5 scale).
        data_quality: Full data quality breakdown (repetition, diversity, language).
        error: Error message if success is False.
    """
    success: bool = True
    status: str = "completed"
    final_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None
    global_step: int = 0
    total_steps: int = 0
    epochs_completed: int = 0
    model_path: Optional[str] = None
    checkpoint_name: Optional[str] = None
    method: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    avg_quality: Optional[float] = None
    data_quality: Optional[Dict[str, float]] = None
    error: Optional[str] = None

    # Backward-compat aliases for old code that expects dict fields
    message: str = ""
    elapsed: float = 0.0
    phases: List[Any] = field(default_factory=list)

    @property
    def checkpoint(self) -> Optional[str]:
        """Alias for ``checkpoint_name``."""
        return self.checkpoint_name

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like ``.get()`` for backward compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dict-like ``result["key"]`` access."""
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Dict-like ``"key" in result`` support."""
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (includes all fields + backward-compat aliases)."""
        d = {
            "success": self.success,
            "status": self.status,
            "final_loss": self.final_loss,
            "best_eval_loss": self.best_eval_loss,
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "epochs_completed": self.epochs_completed,
            "model_path": self.model_path,
            "checkpoint_name": self.checkpoint_name,
            "method": self.method,
            "error": self.error,
            "message": self.message,
            "elapsed": self.elapsed,
            "phases": self.phases,
            "checkpoint": self.checkpoint_name,
            "avg_quality": self.avg_quality,
            "data_quality": self.data_quality,
        }
        d.update(self.metrics)
        return d
