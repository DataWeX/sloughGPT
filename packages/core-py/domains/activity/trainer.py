"""
ActivityTrainer — wraps train_classifier into the TrainerProtocol interface.

Exposes the SloNet-based activity classifier training pipeline as a standard
TrainerProtocol so it can be tracked alongside HF fine-tune and LSTM distillation
jobs in the central training_jobs system.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from domains.training.trainer_protocol import TrainerProtocol, TrainResult

logger = logging.getLogger("man.activity.trainer")

_ACTIVITY_MODEL_LOCK = threading.Lock()
_ACTIVITY_MODEL: Optional[Any] = None


@dataclass
class ActivityTrainerConfig:
    """Configuration for an ActivityTrainer run."""
    epochs: int = 30
    lr: float = 0.001
    batch_size: int = 16
    val_split: float = 0.2
    augment: bool = True
    data_dir: str = "data/activity_records"


class ActivityTrainer:
    """TrainerProtocol-compliant wrapper around train_classifier().

    Reads sensor data from the activity_records .npz files, trains the
    ActivityClassifier, and returns a TrainResult with accuracy and loss metrics.

    Usage::

        trainer = ActivityTrainer()
        result = trainer.train(epochs=30, lr=0.001)
        if result.success:
            print(f"Accuracy: {result.metrics.get('val_accuracy', 0):.2%}")
    """

    def __init__(self, config: Optional[ActivityTrainerConfig] = None):
        self.config = config or ActivityTrainerConfig()
        self._training = False
        self._stop_requested = False
        self._abort_event = threading.Event()

    @property
    def is_training(self) -> bool:
        return self._training

    def stop(self) -> None:
        self._stop_requested = True
        self._abort_event.set()

    # ── TrainerProtocol ──────────────────────────────────────────────

    def train(self, **kwargs: Any) -> TrainResult:
        """Run activity classifier training on collected data.

        Acceptable keyword overrides for ``ActivityTrainerConfig`` fields:
        ``epochs``, ``lr``, ``batch_size``, ``val_split``, ``augment``.

        Returns:
            TrainResult with ``val_accuracy``, ``val_loss``, ``num_labeled``
            stored in ``metrics``.
        """
        self._training = True
        self._stop_requested = False
        self._abort_event.clear()

        from domains.training.slonet import Tensor

        epochs = kwargs.get("epochs", self.config.epochs)
        lr = kwargs.get("lr", self.config.lr)
        batch_size = kwargs.get("batch_size", self.config.batch_size)
        val_split = kwargs.get("val_split", self.config.val_split)
        augment = kwargs.get("augment", self.config.augment)

        try:
            X, y = self._load_data()
            if len(X) < 5:
                return TrainResult(
                    success=False, status="no_data",
                    error=f"Need at least 5 recordings, have {len(X)}.",
                    method="activity",
                )

            labeled = y >= 0
            X_labeled = X[labeled]
            y_labeled = y[labeled]
            num_labeled = len(X_labeled)

            if num_labeled < 5:
                return TrainResult(
                    success=False, status="no_data",
                    error=f"Need at least 5 labeled recordings, have {num_labeled}.",
                    method="activity",
                )

            from domains.activity.classifier import train_classifier, _accuracy, _global_mean
            from domains.activity import ACTIVITIES

            num_classes = len(np.unique(y_labeled))

            on_epoch = self._make_progress_callback(epochs)

            model = train_classifier(
                X_labeled, y_labeled,
                epochs=epochs, lr=lr, batch_size=batch_size,
                val_split=val_split, augment=augment,
                verbose=False, on_epoch=on_epoch,
            )

            # Final validation accuracy on full labeled set
            xv = Tensor(X_labeled, requires_grad=False)
            yv = Tensor(y_labeled, requires_grad=False)
            logits = model.forward(xv)
            val_acc = float(_accuracy(logits, y_labeled))

            # Compute final loss
            xv2 = Tensor(X_labeled, requires_grad=False)
            yv2 = Tensor(y_labeled, requires_grad=False)
            from domains.training.slonet import cross_entropy
            val_loss_val = float(cross_entropy(model.forward(xv2), yv2).data)

            # Persist model globally
            global _ACTIVITY_MODEL
            with _ACTIVITY_MODEL_LOCK:
                _ACTIVITY_MODEL = model

            self._training = False

            return TrainResult(
                success=not self._stop_requested,
                status="stopped" if self._stop_requested else "completed",
                final_loss=val_loss_val,
                total_steps=epochs,
                epochs_completed=epochs,
                method="activity",
                metrics={
                    "val_accuracy": val_acc,
                    "val_loss": val_loss_val,
                    "num_labeled": num_labeled,
                    "num_classes": num_classes,
                    "epochs": epochs,
                    "activities": list(ACTIVITIES),
                },
            )

        except Exception as e:
            logger.exception("Activity training failed")
            self._training = False
            return TrainResult(
                success=False, status="failed",
                error=str(e), method="activity",
            )

    # ── Internals ───────────────────────────────────────────────────

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all .npz recordings from the data directory."""
        repo_root = Path(__file__).resolve().parents[2]  # packages/core-py/
        data_dir = repo_root / self.config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        samples, labels = [], []
        for f in sorted(data_dir.glob("*.npz")):
            try:
                d = np.load(f)
                label = int(d.get("label", -1))
                data = d["data"]
                samples.append(data)
                labels.append(label)
            except Exception as e:
                logger.warning("Skipping corrupted %s: %s", f, e)

        if not samples:
            return np.empty((0, 0, 6), dtype=np.float32), np.empty(0, dtype=np.int64)

        X = np.stack(samples).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        return X, y

    def _make_progress_callback(self, total_epochs: int):
        """Return a callable that stores per-epoch metrics for the callback pattern."""
        if total_epochs <= 0:
            total_epochs = 30

        def cb(epoch, epochs, loss, val_loss, val_accuracy, lr):
            if self._abort_event.is_set():
                import numpy as _np
                raise _np.linalg.LinAlgError("Training aborted by user")

        return cb


def get_activity_model() -> Optional[Any]:
    """Return the current global activity model (or None)."""
    with _ACTIVITY_MODEL_LOCK:
        return _ACTIVITY_MODEL


def set_activity_model(model: Any) -> None:
    """Replace the global activity model."""
    global _ACTIVITY_MODEL
    with _ACTIVITY_MODEL_LOCK:
        _ACTIVITY_MODEL = model
