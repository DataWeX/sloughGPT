"""
Training Status Tracking

Tracks training completion status and enables:
- Completion verification
- Training history
- Progress summaries

Checkpoint persistence and resume live in the canonical
``domains.training.train_pipeline.CheckpointManager`` (`.soul`/`.npz` via
``domains.training.slonet``); this module no longer ships a checkpoint manager.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from domains.training.slonet import load_checkpoint_npz as _load_npz
from domains.training.slonet import save_checkpoint_npz as _save_npz

logger = logging.getLogger("slo.training.status")


class TrainingStage(Enum):
    """Training pipeline stages."""
    NOT_STARTED = "not_started"
    PRETRAINING = "pretraining"
    FEDERATED = "federated"
    RLHF = "rlhf"
    COMPLETE = "complete"
    FAILED = "failed"


class CompletionStatus(Enum):
    """Training completion status."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    NOT_STARTED = "not_started"


@dataclass
class StageStatus:
    """Status of a single training stage."""
    name: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    epochs_completed: int = 0
    total_epochs: int = 0
    best_loss: float = 0.0
    final_loss: float = 0.0
    status: CompletionStatus = CompletionStatus.NOT_STARTED
    error: Optional[str] = None


@dataclass
class TrainingCompletionReport:
    """Complete report of training status."""
    model_name: str
    created_at: str
    trained_at: Optional[str] = None

    # Overall completion
    completion_status: CompletionStatus = CompletionStatus.NOT_STARTED
    completion_percentage: float = 0.0

    # Stage statuses
    pretraining: Optional[StageStatus] = None
    federated: Optional[StageStatus] = None
    rlhf: Optional[StageStatus] = None

    # Overall metrics
    total_epochs: int = 0
    total_steps: int = 0
    best_loss: float = 0.0
    final_loss: float = 0.0
    best_val_loss: float = 0.0

    # Checkpoint info
    checkpoint_path: Optional[str] = None
    last_checkpoint_step: int = 0
    checkpoint_count: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    dataset: str = ""
    batch_size: int = 0
    learning_rate: float = 0.0
    precision: str = ""

    def is_complete(self) -> bool:
        """Check if training is complete."""
        return self.completion_status == CompletionStatus.COMPLETED

    def can_resume(self) -> bool:
        """Check if training can be resumed."""
        return self.completion_status in [
            CompletionStatus.IN_PROGRESS,
            CompletionStatus.INTERRUPTED,
        ] and self.checkpoint_path is not None

    def get_progress_summary(self) -> str:
        """Get human-readable progress summary."""
        if self.completion_status == CompletionStatus.COMPLETED:
            return f"Training complete! Final loss: {self.final_loss:.4f}"
        elif self.completion_status == CompletionStatus.IN_PROGRESS:
            return f"Training in progress: {self.completion_percentage:.1f}%"
        elif self.completion_status == CompletionStatus.INTERRUPTED:
            return f"Training interrupted at {self.completion_percentage:.1f}%. Can resume."
        else:
            return "Training not started"


class TrainingStatusTracker:
    """
    Tracks training completion status and history.
    """

    def __init__(self, model_name: str = "sloughgpt"):
        self.model_name = model_name
        self.report = TrainingCompletionReport(
            model_name=model_name,
            created_at=datetime.now(timezone.utc).isoformat() + "Z",
        )
        self.checkpoints: List[Dict[str, Any]] = []

    def start_training(
        self,
        dataset: str = "",
        batch_size: int = 0,
        learning_rate: float = 0.0,
        pretrain_epochs: int = 0,
        federated_rounds: int = 0,
        rlhf_epochs: int = 0,
        precision: str = "bf16",
    ):
        """Initialize training status."""
        self.report.completion_status = CompletionStatus.IN_PROGRESS
        self.report.dataset = dataset
        self.report.batch_size = batch_size
        self.report.learning_rate = learning_rate
        self.report.precision = precision

        # Initialize stage statuses
        if pretrain_epochs > 0:
            self.report.pretraining = StageStatus(
                name="Pretraining",
                total_epochs=pretrain_epochs,
            )
        if federated_rounds > 0:
            self.report.federated = StageStatus(
                name="Federated Learning",
                total_epochs=federated_rounds,
            )
        if rlhf_epochs > 0:
            self.report.rlhf = StageStatus(
                name="RLHF Alignment",
                total_epochs=rlhf_epochs,
            )

    def start_stage(self, stage: TrainingStage):
        """Mark a stage as started."""
        stage_name_map = {
            TrainingStage.PRETRAINING: self.report.pretraining,
            TrainingStage.FEDERATED: self.report.federated,
            TrainingStage.RLHF: self.report.rlhf,
        }

        stage_status = stage_name_map.get(stage)
        if stage_status:
            stage_status.started_at = datetime.now(timezone.utc).isoformat() + "Z"
            stage_status.status = CompletionStatus.IN_PROGRESS

    def update_stage(
        self,
        stage: TrainingStage,
        epoch: int,
        loss: float,
        val_loss: Optional[float] = None,
    ):
        """Update stage progress."""
        stage_name_map = {
            TrainingStage.PRETRAINING: self.report.pretraining,
            TrainingStage.FEDERATED: self.report.federated,
            TrainingStage.RLHF: self.report.rlhf,
        }

        stage_status = stage_name_map.get(stage)
        if stage_status:
            stage_status.epochs_completed = epoch + 1
            stage_status.final_loss = loss
            if val_loss is not None and (stage_status.best_loss == 0 or val_loss < stage_status.best_loss):
                stage_status.best_loss = val_loss

            # Update overall progress
            self._update_overall_progress()

    def complete_stage(self, stage: TrainingStage):
        """Mark a stage as complete."""
        stage_name_map = {
            TrainingStage.PRETRAINING: self.report.pretraining,
            TrainingStage.FEDERATED: self.report.federated,
            TrainingStage.RLHF: self.report.rlhf,
        }

        stage_status = stage_name_map.get(stage)
        if stage_status:
            stage_status.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
            stage_status.status = CompletionStatus.COMPLETED

            if stage_status.best_loss > 0:
                self.report.best_loss = stage_status.best_loss
            self.report.final_loss = stage_status.final_loss
            self.report.total_epochs += stage_status.epochs_completed

            self._update_overall_progress()

    def fail_stage(self, stage: TrainingStage, error: str):
        """Mark a stage as failed."""
        stage_name_map = {
            TrainingStage.PRETRAINING: self.report.pretraining,
            TrainingStage.FEDERATED: self.report.federated,
            TrainingStage.RLHF: self.report.rlhf,
        }

        stage_status = stage_name_map.get(stage)
        if stage_status:
            stage_status.status = CompletionStatus.FAILED
            stage_status.error = error
            self.report.errors.append(f"{stage.value}: {error}")
            self.report.completion_status = CompletionStatus.FAILED

    def record_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        loss: float,
    ):
        """Record a checkpoint."""
        self.report.checkpoint_path = checkpoint_path
        self.report.last_checkpoint_step = step

        self.checkpoints.append({
            "path": checkpoint_path,
            "step": step,
            "loss": loss,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        })
        self.report.checkpoint_count = len(self.checkpoints)

    def _update_overall_progress(self):
        """Calculate overall training progress."""
        total_epochs = 0
        completed_epochs = 0

        for stage in [self.report.pretraining, self.report.federated, self.report.rlhf]:
            if stage:
                total_epochs += stage.total_epochs
                completed_epochs += stage.epochs_completed

        if total_epochs > 0:
            self.report.total_epochs = total_epochs
            self.report.completion_percentage = (completed_epochs / total_epochs) * 100

            # Check if all stages complete
            all_complete = all(
                s.status == CompletionStatus.COMPLETED
                for s in [self.report.pretraining, self.report.federated, self.report.rlhf]
                if s
            )

            if all_complete:
                self.report.completion_status = CompletionStatus.COMPLETED
                self.report.trained_at = datetime.now(timezone.utc).isoformat() + "Z"

    def mark_complete(self):
        """Mark training as complete."""
        self.report.completion_status = CompletionStatus.COMPLETED
        self.report.completion_percentage = 100.0
        self.report.trained_at = datetime.now(timezone.utc).isoformat() + "Z"

    def get_report(self) -> TrainingCompletionReport:
        """Get the completion report."""
        return self.report

    def save_report(self, path: str):
        """Save report to JSON (converts enums to their values)."""
        def _serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(path, 'w') as f:
            json.dump(asdict(self.report), f, indent=2, default=_serialize)

    @classmethod
    def load_report(cls, path: str) -> "TrainingStatusTracker":
        """Load report from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        def _coerce(obj):
            if isinstance(obj, dict):
                if {"name", "status"} <= set(obj) and isinstance(
                    obj.get("name"), str
                ):
                    return StageStatus(**{k: _coerce(v) for k, v in obj.items()})
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(v) for v in obj]
            if isinstance(obj, str):
                for enum_cls in (CompletionStatus, TrainingStage):
                    for member in enum_cls:
                        if member.value == obj:
                            return member
            return obj

        data = _coerce(data)
        tracker = cls(data.get("model_name", "unknown"))
        tracker.report = TrainingCompletionReport(**data)
        return tracker

    def print_summary(self):
        """Print human-readable summary."""
        logger.info("=" * 60, extra={"tag": "TRAIN"})
        logger.info("Training Status: %s", self.report.completion_status.value, extra={"tag": "TRAIN"})
        logger.info("Progress: %.1f%%", self.report.completion_percentage, extra={"tag": "TRAIN"})
        logger.info("Total Epochs: %s", self.report.total_epochs, extra={"tag": "TRAIN"})
        logger.info("Best Loss: %.4f", self.report.best_loss, extra={"tag": "TRAIN"})
        logger.info("Final Loss: %.4f", self.report.final_loss, extra={"tag": "TRAIN"})
        logger.info("-" * 60, extra={"tag": "TRAIN"})

        for stage in [self.report.pretraining, self.report.federated, self.report.rlhf]:
            if stage:
                logger.info("", extra={"tag": "TRAIN"})
                logger.info("%s:", stage.name, extra={"tag": "TRAIN"})
                logger.info("  Status: %s", stage.status.value, extra={"tag": "TRAIN"})
                logger.info("  Epochs: %s/%s", stage.epochs_completed, stage.total_epochs, extra={"tag": "TRAIN"})
                if stage.best_loss > 0:
                    logger.info("  Best Loss: %.4f", stage.best_loss, extra={"tag": "TRAIN"})
                if stage.error:
                    logger.info("  Error: %s", stage.error, extra={"tag": "TRAIN"})

        logger.info("=" * 60, extra={"tag": "TRAIN"})

# =============================================================================
# STANDALONE NPZ CHECKPOINT HELPERS (native)
# =============================================================================


def _tensors_to_numpy(state_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert tensors in a state dict to numpy arrays."""
    result = {}
    for k, v in state_dict.items():
        if hasattr(v, "cpu"):
            result[k] = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = _tensors_to_numpy(v)
        else:
            result[k] = np.array(v)
    return result


# Native implementations live in domains.training.slonet.
save_checkpoint_npz = _save_npz
load_checkpoint_npz = _load_npz


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TrainingStage",
    "CompletionStatus",
    "StageStatus",
    "TrainingCompletionReport",
    "TrainingStatusTracker",
]
