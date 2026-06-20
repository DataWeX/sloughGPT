"""
Training Status & Checkpoint Management

Tracks training completion status and enables:
- Checkpoint loading with metadata
- Training resume capability
- Completion verification
- Training history
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from domains.training.slonet_compat import torch
import logging

logger = logging.getLogger("man.training.status")


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
    Tracks training status and manages checkpoints.
    """
    
    def __init__(self, model_name: str = "sloughgpt"):
        self.model_name = model_name
        self.report = TrainingCompletionReport(
            model_name=model_name,
            created_at=datetime.utcnow().isoformat() + "Z",
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
            stage_status.started_at = datetime.utcnow().isoformat() + "Z"
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
            stage_status.completed_at = datetime.utcnow().isoformat() + "Z"
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
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
                self.report.trained_at = datetime.utcnow().isoformat() + "Z"
    
    def mark_complete(self):
        """Mark training as complete."""
        self.report.completion_status = CompletionStatus.COMPLETED
        self.report.completion_percentage = 100.0
        self.report.trained_at = datetime.utcnow().isoformat() + "Z"
    
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
        
        tracker = cls(data.get("model_name", "unknown"))
        tracker.report = TrainingCompletionReport(**data)
        return tracker
    
    def print_summary(self):
        """Print human-readable summary."""
        print("=" * 60)
        print(f"Training Status: {self.report.completion_status.value}")
        print(f"Progress: {self.report.completion_percentage:.1f}%")
        print(f"Total Epochs: {self.report.total_epochs}")
        print(f"Best Loss: {self.report.best_loss:.4f}")
        print(f"Final Loss: {self.report.final_loss:.4f}")
        print("-" * 60)
        
        for stage in [self.report.pretraining, self.report.federated, self.report.rlhf]:
            if stage:
                print(f"\n{stage.name}:")
                print(f"  Status: {stage.status.value}")
                print(f"  Epochs: {stage.epochs_completed}/{stage.total_epochs}")
                if stage.best_loss > 0:
                    print(f"  Best Loss: {stage.best_loss:.4f}")
                if stage.error:
                    print(f"  Error: {stage.error}")
        
        print("=" * 60)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Manages model checkpoints with metadata.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = TrainingStatusTracker()
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        step: int,
        epoch: int,
        loss: float,
        val_loss: Optional[float] = None,
        stage: TrainingStage = TrainingStage.PRETRAINING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint with metadata."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step{step}.pt"
        
        checkpoint = {
            # Model state
            "model_state_dict": model.state_dict() if hasattr(model, 'state_dict') else None,
            
            # Optimizer state
            "optimizer_state_dict": optimizer.state_dict() if optimizer and hasattr(optimizer, 'state_dict') else None,
            
            # Training progress
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "stage": stage.value,
            
            # Metadata
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata or {},
            
            # Training status
            "training_status": asdict(self.tracker.report),
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Update tracker
        self.tracker.record_checkpoint(str(checkpoint_path), step, loss)
        self.tracker.update_stage(stage, epoch, loss, val_loss)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load checkpoint with metadata."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        if "model_state_dict" in checkpoint and checkpoint["model_state_dict"] is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer
        if optimizer and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Restore training status
        if "training_status" in checkpoint:
            self.tracker.report = TrainingCompletionReport(**checkpoint["training_status"])
        
        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", 0),
            "val_loss": checkpoint.get("val_loss"),
            "stage": TrainingStage(checkpoint.get("stage", "pretraining")),
            "timestamp": checkpoint.get("timestamp"),
        }
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint (lowest loss)."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        best_path = None
        best_loss = float('inf')
        
        for path in checkpoints:
            try:
                ckpt = torch.load(path, map_location="cpu")
                loss = ckpt.get("loss", float('inf'))
                if loss < best_loss:
                    best_loss = loss
                    best_path = path
            except Exception:
                continue
        
        return str(best_path) if best_path else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata."""
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                ckpt = torch.load(path, map_location="cpu")
                checkpoints.append({
                    "path": str(path),
                    "step": ckpt.get("step", 0),
                    "epoch": ckpt.get("epoch", 0),
                    "loss": ckpt.get("loss", 0),
                    "val_loss": ckpt.get("val_loss"),
                    "stage": ckpt.get("stage"),
                    "timestamp": ckpt.get("timestamp"),
                })
            except Exception:
                continue
        
        return sorted(checkpoints, key=lambda x: x.get("step", 0), reverse=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TrainingStage",
    "CompletionStatus",
    "StageStatus",
    "TrainingCompletionReport",
    "TrainingStatusTracker",
    "CheckpointManager",
]
