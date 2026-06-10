"""
Training Sequence Protocol

Defines the sequential phases all training methods follow:
    GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE

Also provides TrainingSequenceState for progress tracking and
TrainingRunConfig for per-run configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class TrainingSequence(Enum):
    """Canonical training phase sequence.

    Every training run follows these phases in order. Stages may be skipped
    depending on TrainingRunConfig.
    """
    IDLE = "idle"
    GENERATE_DATA = "generate_data"
    DISTILL = "distill"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"
    COMPLETE = "complete"
    FAILED = "failed"
    EARLY_STOP = "early_stop"

    @classmethod
    def ordered_phases(cls) -> List[TrainingSequence]:
        """Return phases in execution order."""
        return [
            cls.IDLE,
            cls.GENERATE_DATA,
            cls.DISTILL,
            cls.TRAIN,
            cls.EVALUATE,
            cls.DEPLOY,
            cls.COMPLETE,
            cls.FAILED,
            cls.EARLY_STOP,
        ]


@dataclass
class PhaseResult:
    """Result of a single training phase."""
    phase: TrainingSequence
    status: str = "working"  # working | success | error | skipped
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "status": self.status,
            "message": self.message,
            "metrics": self.metrics,
        }


@dataclass
class TrainingSequenceState:
    """Tracks progress through a training sequence.

    Usage:
        state = TrainingSequenceState()
        state.start_phase(TrainingSequence.GENERATE_DATA)
        # ... do work ...
        state.complete_phase(TrainingSequence.GENERATE_DATA, metrics={"samples": 1000})
    """
    current_phase: TrainingSequence = TrainingSequence.IDLE
    phase_results: List[PhaseResult] = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    early_stop_reason: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.current_phase not in (
            TrainingSequence.IDLE,
            TrainingSequence.COMPLETE,
            TrainingSequence.FAILED,
            TrainingSequence.EARLY_STOP,
        )

    @property
    def is_done(self) -> bool:
        return self.current_phase in (
            TrainingSequence.COMPLETE,
            TrainingSequence.FAILED,
            TrainingSequence.EARLY_STOP,
        )

    def start_phase(self, phase: TrainingSequence) -> None:
        """Mark a phase as started."""
        self.current_phase = phase
        self.phase_results.append(PhaseResult(phase=phase, status="working"))

    def complete_phase(self, phase: TrainingSequence, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark a phase as completed successfully."""
        for pr in self.phase_results:
            if pr.phase == phase and pr.status == "working":
                pr.status = "success"
                if metrics:
                    pr.metrics.update(metrics)
                break

    def fail_phase(self, phase: TrainingSequence, message: str = "") -> None:
        """Mark a phase as failed."""
        for pr in self.phase_results:
            if pr.phase == phase and pr.status == "working":
                pr.status = "error"
                pr.message = message or f"Phase {phase.value} failed"
                break
        self.current_phase = TrainingSequence.FAILED
        self.error = message or f"Phase {phase.value} failed"

    def skip_phase(self, phase: TrainingSequence, reason: str = "") -> None:
        """Mark a phase as skipped."""
        self.phase_results.append(PhaseResult(phase=phase, status="skipped", message=reason))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase.value,
            "phase_results": [pr.to_dict() for pr in self.phase_results],
            "is_running": self.is_running,
            "is_done": self.is_done,
            "error": self.error,
            "early_stop_reason": self.early_stop_reason,
        }

    def to_sse_event(self, stream_name: str = "auto-train") -> Dict[str, Any]:
        """Format as standard SSE envelope event."""
        return {
            "stream": stream_name,
            "phase": self.current_phase.value,
            "status": "complete" if self.is_done else ("error" if self.current_phase == TrainingSequence.FAILED else "working"),
            "data": {
                "phase_results": [pr.to_dict() for pr in self.phase_results],
            },
            "meta": {
                "error": self.error,
            },
            "message": self.phase_results[-1].message if self.phase_results else "",
        }


@dataclass
class TrainingRunConfig:
    """Configuration for a training run, controlling which phases execute."""
    skip_generate: bool = False
    skip_distill: bool = False
    skip_train: bool = False
    skip_evaluate: bool = False
    skip_deploy: bool = False
    max_epochs: int = 10
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.01
    eval_every_n_steps: int = 100
    deploy_on_complete: bool = True

    @classmethod
    def defaults(cls) -> TrainingRunConfig:
        return cls()

    def effective_phases(self) -> List[TrainingSequence]:
        """Return the list of phases that will actually run."""
        phases = [TrainingSequence.GENERATE_DATA, TrainingSequence.DISTILL,
                  TrainingSequence.TRAIN, TrainingSequence.EVALUATE,
                  TrainingSequence.DEPLOY]
        skip_map = {
            TrainingSequence.GENERATE_DATA: self.skip_generate,
            TrainingSequence.DISTILL: self.skip_distill,
            TrainingSequence.TRAIN: self.skip_train,
            TrainingSequence.EVALUATE: self.skip_evaluate,
            TrainingSequence.DEPLOY: self.skip_deploy,
        }
        return [p for p in phases if not skip_map.get(p, False)]


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DataGenerator(Protocol):
    """Protocol for generating synthetic training data (teacher-driven)."""
    def generate(self, prompt: str, num_samples: int, max_length: int) -> List[str]:
        ...


@runtime_checkable
class StudentModel(Protocol):
    """Protocol for a student model that can be trained via distillation."""
    def train_step(self, inputs: Any, labels: Any) -> float:
        ...

    def evaluate(self, inputs: Any, labels: Any) -> Dict[str, float]:
        ...


# =============================================================================
# Checkpoint format
# =============================================================================


@dataclass
class CheckpointFormat:
    """Standard checkpoint format with metadata."""
    name: str
    step: int
    loss: float
    val_loss: Optional[float] = None
    epoch: int = 0
    stoi: Optional[Dict[str, int]] = None
    itos: Optional[Dict[int, str]] = None
    vocab: Optional[Dict[str, int]] = None
    personality_traits: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "step": self.step,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "epoch": self.epoch,
            "personality_traits": self.personality_traits,
            **self.metadata,
        }


__all__ = [
    "TrainingSequence",
    "PhaseResult",
    "TrainingSequenceState",
    "TrainingRunConfig",
    "DataGenerator",
    "StudentModel",
    "CheckpointFormat",
]
