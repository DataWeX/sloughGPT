"""
Training Callbacks - Hooks for Training Pipeline

Production-grade callback system for training loops:
- Logging callbacks
- Checkpoint callbacks
- Early stopping
- Learning rate scheduling
- Custom callbacks
"""

import logging
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from abc import ABC
from enum import Enum

logger = logging.getLogger("sloughgpt.callbacks")


class CallbackOrder(Enum):
    """Callback execution order."""

    FIRST = 1
    EARLY = 2
    NORMAL = 3
    LATE = 4
    LAST = 5


class TrainPhase(Enum):
    """Training phase."""

    INIT = "init"
    TRAIN = "train"
    VALIDATE = "validate"
    EPOCH_BEGIN = "epoch_begin"
    EPOCH_END = "epoch_end"
    BATCH_BEGIN = "batch_begin"
    BATCH_END = "batch_end"
    FIT_BEGIN = "fit_begin"
    FIT_END = "fit_end"


@dataclass
class TrainerState:
    """Current state of the trainer."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    learning_rate: float = 0.0
    phase: TrainPhase = TrainPhase.INIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": self.metrics,
            "learning_rate": self.learning_rate,
            "phase": self.phase.value,
        }


class Callback(ABC):
    """Base callback class."""

    def __init__(self, order: CallbackOrder = CallbackOrder.NORMAL):
        self.order = order
        self.trainer = None

    def set_trainer(self, trainer: "Trainer"):
        """Set reference to trainer."""
        self.trainer = trainer

    @property
    def state(self) -> TrainerState:
        """Get trainer state."""
        return self.trainer.state if self.trainer else None

    def on_init(self):
        """Called when callback is initialized."""
        pass

    def on_fit_begin(self):
        """Called at the beginning of training."""
        pass

    def on_fit_end(self):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, loss: float):
        """Called at the end of each batch."""
        pass

    def on_train_begin(self):
        """Called when training phase begins."""
        pass

    def on_train_end(self):
        """Called when training phase ends."""
        pass

    def on_validate_begin(self):
        """Called when validation phase begins."""
        pass

    def on_validate_end(self, metrics: Dict[str, float]):
        """Called when validation phase ends."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ):
        super().__init__(CallbackOrder.FIRST)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Check if training should stop."""
        current = metrics.get(self.monitor)

        if current is None:
            return

        if self.mode == "min":
            improved = current < (self.best_metric - self.min_delta)
        else:
            improved = current > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = current
            self.wait = 0
            self.state.best_metric = current
            self.state.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True

        return False


class ModelCheckpointCallback(Callback):
    """Save model checkpoints."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_frequency: int = 1,
        verbose: bool = True,
    ):
        super().__init__(CallbackOrder.EARLY)
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.verbose = verbose

        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if needed."""
        current = metrics.get(self.monitor)

        if current is None:
            return

        should_save = False

        if self.save_best_only:
            if self.mode == "min":
                should_save = current < self.best_metric
            else:
                should_save = current > self.best_metric

            if should_save:
                self.best_metric = current
        else:
            should_save = epoch % self.save_frequency == 0

        if should_save:
            filepath = self.filepath / f"checkpoint-epoch-{epoch}.pt"
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                logger.info(f"Saving checkpoint to {filepath}")

            return {"checkpoint_path": str(filepath), "epoch": epoch, "metrics": metrics}


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduling callback."""

    def __init__(
        self,
        scheduler_fn: Optional[Callable[[int, float], float]] = None,
        schedule: Optional[Dict[int, float]] = None,
        warmup_epochs: int = 0,
        mode: str = "cosine",
    ):
        super().__init__(CallbackOrder.EARLY)
        self.scheduler_fn = scheduler_fn
        self.schedule = schedule or {}
        self.warmup_epochs = warmup_epochs
        self.mode = mode

    def _get_warmup_lr(self, epoch: int, initial_lr: float) -> float:
        """Linear warmup."""
        if epoch < self.warmup_epochs:
            return initial_lr * (epoch + 1) / self.warmup_epochs
        return None

    def _get_cosine_lr(self, epoch: int, initial_lr: float, total_epochs: int) -> float:
        """Cosine annealing."""
        if epoch >= self.warmup_epochs:
            progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
            return initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        return None

    def _get_step_lr(self, epoch: int, initial_lr: float) -> float:
        """Step decay."""
        for milestone, lr in sorted(self.schedule.items(), reverse=True):
            if epoch >= milestone:
                return lr
        return initial_lr

    def on_epoch_begin(self, epoch: int):
        """Update learning rate."""
        if not self.state:
            return

        if self.scheduler_fn:
            lr = self.scheduler_fn(epoch, self.state.learning_rate)
            self.state.learning_rate = lr
            return

        initial_lr = 1e-3

        warmup_lr = self._get_warmup_lr(epoch, initial_lr)
        if warmup_lr is not None:
            self.state.learning_rate = warmup_lr
            return

        if self.mode == "cosine":
            lr = self._get_cosine_lr(epoch, initial_lr, 100)
        elif self.mode == "step":
            lr = self._get_step_lr(epoch, initial_lr)
        else:
            lr = initial_lr

        self.state.learning_rate = lr


class LoggingCallback(Callback):
    """Logging callback."""

    def __init__(
        self,
        log_frequency: int = 10,
        log_metrics: Optional[List[str]] = None,
        log_path: Optional[str] = None,
    ):
        super().__init__(CallbackOrder.NORMAL)
        self.log_frequency = log_frequency
        self.log_metrics = log_metrics or ["loss", "val_loss"]
        self.log_path = Path(log_path) if log_path else None

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, batch: int, loss: float):
        """Log batch metrics."""
        if batch % self.log_frequency == 0:
            logger.info(f"Step {batch}: loss = {loss:.4f}")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        log_str = f"Epoch {epoch}: "
        log_str += ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items() if k in self.log_metrics)
        logger.info(log_str)

        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")


class TensorBoardCallback(Callback):
    """TensorBoard-compatible logging."""

    def __init__(
        self, log_dir: str = "./logs", histogram_frequency: int = 0, write_graph: bool = True
    ):
        super().__init__(CallbackOrder.NORMAL)
        self.log_dir = Path(log_dir)
        self.histogram_frequency = histogram_frequency
        self.write_graph = write_graph

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, batch: int, loss: float):
        """Log batch scalar."""
        self._write_scalar("batch/loss", loss, batch)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch scalars."""
        for name, value in metrics.items():
            self._write_scalar(f"epoch/{name}", value, epoch)

    def _write_scalar(self, tag: str, value: float, step: int):
        """Write scalar to log file."""
        log_file = self.log_dir / f"{tag.replace('/', '_')}.log"
        with open(log_file, "a") as f:
            f.write(f"{step},{value}\n")


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = callbacks or []
        self._sort_callbacks()

    def _sort_callbacks(self):
        """Sort callbacks by order."""
        self.callbacks.sort(key=lambda c: c.order.value)

    def append(self, callback: Callback):
        """Add callback."""
        self.callbacks.append(callback)
        self._sort_callbacks()

    def set_trainer(self, trainer: "Trainer"):
        """Set trainer for all callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_init(self):
        """Call on_init for all callbacks."""
        for callback in self.callbacks:
            callback.on_init()

    def on_fit_begin(self):
        """Call on_fit_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_fit_begin()

    def on_fit_end(self):
        """Call on_fit_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_fit_end()

    def on_epoch_begin(self, epoch: int):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Call on_epoch_end for all callbacks. Returns True if should stop."""
        should_stop = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, metrics)
            if result is True:
                should_stop = True
        return should_stop

    def on_batch_begin(self, batch: int):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch: int, loss: float):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, loss)

    def on_train_begin(self):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_validate_begin(self):
        """Call on_validate_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_validate_begin()

    def on_validate_end(self, metrics: Dict[str, float]):
        """Call on_validate_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_validate_end(metrics)


import math


__all__ = [
    "Callback",
    "CallbackOrder",
    "TrainPhase",
    "TrainerState",
    "CallbackList",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LearningRateSchedulerCallback",
    "LoggingCallback",
    "TensorBoardCallback",
]
