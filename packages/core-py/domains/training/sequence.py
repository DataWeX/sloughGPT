"""
Training Sequence Protocol

Unified training stages for both auto-train and normal training.
Every training run follows the same sequence regardless of model type.

Sequence:
    GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY

Protocols:
    - TrainingProtocol: main train() entrypoint
    - DataGenerator: produces (prompt, response) pairs from teacher model
    - StudentModel: the model being trained
    - EvalProtocol: evaluates student against teacher/baseline
    - CheckpointProtocol: save/load with vocabulary embedding

Both auto-train (LSTM) and normal training (Transformer) share this sequence.
The only difference is:
    - auto-train: uses GPT2 as teacher, LSTM as student, char-level
    - normal: uses static files, Transformer as student, char-level
    - both: produce .pt checkpoints with stoi/itos/vocab for eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable
import time

from domains.training.slonet_compat import torch


class TrainingSequence(Enum):
    """The canonical training sequence. Every training run follows these stages."""
    IDLE = "idle"                        # No training started
    GENERATE_DATA = "generate_data"       # Teacher model produces training pairs
    DISTILL = "distill"                  # Student learns from teacher outputs
    TRAIN = "train"                      # Direct training on data
    EVALUATE = "evaluate"                # Eval student vs teacher/baseline
    DEPLOY = "deploy"                    # Save checkpoint, mark ready
    COMPLETE = "complete"                # Training finished successfully
    FAILED = "failed"                    # Training failed
    EARLY_STOP = "early_stop"            # Training stopped early


@dataclass
class StageResult:
    """Result from a single training stage."""
    stage: TrainingSequence
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    data_generated: int = 0            # chars or pairs generated
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    message: str = ""

    def duration(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "duration_s": round(self.duration(), 2),
            "success": self.success,
            "metrics": self.metrics,
            "data_generated": self.data_generated,
            "checkpoint": self.checkpoint_path,
            "error": self.error,
            "message": self.message,
        }


@dataclass
class TrainingSequenceState:
    """Tracks progress through the full training sequence."""
    current_stage: TrainingSequence = TrainingSequence.IDLE
    stages_completed: List[TrainingSequence] = field(default_factory=list)
    stage_results: Dict[TrainingSequence, StageResult] = field(default_factory=dict)
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    best_loss: float = float("inf")
    progress_pct: float = 0.0

    def start_stage(self, stage: TrainingSequence) -> StageResult:
        self.current_stage = stage
        result = StageResult(stage=stage, started_at=time.time())
        self.stage_results[stage] = result
        return result

    def complete_stage(self, stage: TrainingSequence, result: StageResult):
        result.completed_at = time.time()
        result.success = True
        self.stage_results[stage] = result
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)

    def fail_stage(self, stage: TrainingSequence, error: str, result: Optional[StageResult] = None):
        r = result or StageResult(stage=stage, started_at=time.time())
        r.completed_at = time.time()
        r.success = False
        r.error = error
        self.stage_results[stage] = r
        self.current_stage = TrainingSequence.FAILED

    def update_progress(self, step: int, loss: float, epoch: int):
        self.total_steps = max(self.total_steps, step)
        self.current_epoch = epoch
        self.current_loss = loss
        self.best_loss = min(self.best_loss, loss)
        if self.total_epochs > 0:
            self.progress_pct = min(100.0, (epoch / self.total_epochs) * 100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.current_stage.value,
            "stages_completed": [s.value for s in self.stages_completed],
            "total_steps": self.total_steps,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": round(self.current_loss, 4),
            "best_loss": round(self.best_loss, 4),
            "progress_pct": round(self.progress_pct, 1),
            "stage_results": {k.value: v.to_dict() for k, v in self.stage_results.items()},
        }


@dataclass
class TrainingRunConfig:
    """Configuration for a training run."""
    # Sequence settings
    stages: List[TrainingSequence] = field(default_factory=lambda: [
        TrainingSequence.GENERATE_DATA,
        TrainingSequence.DISTILL,
        TrainingSequence.TRAIN,
        TrainingSequence.EVALUATE,
        TrainingSequence.DEPLOY,
    ])
    skip_generate: bool = False           # Use static files instead of GPT generation
    skip_distill: bool = False            # Skip distillation, just train directly
    skip_evaluate: bool = False           # Skip eval stage

    # Model settings
    student_type: str = "lstm"            # "lstm" or "transformer"
    teacher_model: str = "gpt2"           # Which model generates training data
    soul_name: str = "assistant"
    system_prompt: Optional[str] = None

    # Training settings
    epochs: int = 10
    batch_size: int = 32
    block_size: int = 128
    learning_rate: float = 0.001
    temperature: float = 0.8
    max_grad_norm: float = 1.0

    # Data settings
    max_data_chars: int = 100000         # Max chars to generate/store
    max_prompts: int = 100                # Number of prompts for data generation
    data_source: str = "generated"       # "generated" or "files"

    # Checkpointing
    checkpoint_dir: str = "models/auto-training"
    checkpoint_interval: int = 500
    save_best_only: bool = False
    max_checkpoints: int = 5

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.05

    # Evaluation
    eval_prompts: List[str] = field(default_factory=lambda: [
        "Hello, how are you?",
        "What is Python?",
        "Explain machine learning.",
        "Write a short poem.",
        "What did you do today?",
    ])


@runtime_checkable
class DataGenerator(Protocol):
    """Protocol for teacher models that generate training data."""

    def generate_pair(self, prompt: str, temperature: float) -> tuple[str, str]:
        """Generate (prompt, response) pair. Returns (prompt, response)."""
        ...

    def generate_batch(self, prompts: List[str], temperature: float) -> List[tuple[str, str]]:
        """Generate multiple pairs efficiently."""
        ...


@runtime_checkable
class StudentModel(Protocol):
    """Protocol for student models being trained."""

    def train_step(self, x: Any, y: Any) -> Dict[str, float]:
        """Run one training step. Returns {'loss': float, ...}."""
        ...

    def eval_step(self, x: Any, y: Any) -> Dict[str, float]:
        """Run one eval step. Returns metrics."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get model state for checkpointing."""
        ...

    def load_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint."""
        ...


class CheckpointFormat:
    """
    Standard checkpoint format used by both LSTM and Transformer.
    Every checkpoint includes vocabulary so eval tools can run without external files.

    Format:
        checkpoint = {
            "model_state": {...},        # model.state_dict()
            "stoi": {...},              # char-to-int mapping
            "itos": {...},              # int-to-char mapping
            "vocab_size": int,
            "soul_name": str,
            "system_prompt": str,
            "train_loss": float,
            "steps": int,
            "epochs": int,
            "personality_traits": {...},  # warmth, creativity, curiosity, confidence
            "student_type": str,          # "lstm" or "transformer"
            "timestamp": str,
            "stage": str,                # which stage completed
        }
    """
    VERSION = "1.0"

    @staticmethod
    def create(
        model_state: Dict[str, Any],
        stoi: Dict[str, int],
        itos: Dict[int, str],
        soul_name: str,
        system_prompt: str,
        train_loss: float,
        steps: int,
        epochs: int,
        student_type: str = "lstm",
        personality_traits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return {
            "version": CheckpointFormat.VERSION,
            "model_state": model_state,
            "stoi": stoi,
            "itos": itos,
            "vocab_size": len(stoi),
            "soul_name": soul_name,
            "system_prompt": system_prompt,
            "train_loss": round(train_loss, 4),
            "steps": steps,
            "epochs": epochs,
            "student_type": student_type,
            "personality_traits": personality_traits or {
                "warmth": 0.5, "creativity": 0.5,
                "curiosity": 0.5, "confidence": 0.5,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def save(checkpoint: Dict[str, Any], path: str):
        torch.save(checkpoint, path)