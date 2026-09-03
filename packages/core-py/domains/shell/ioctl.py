"""
Ioctl interface — clean, type-safe, documented.

Provides:
  IoctlCommand enum for type safety
  IoctlResult for consistent responses
  IoctlError for proper error handling
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any


class IoctlCommand(Enum):
    """Ioctl commands — type-safe command identifiers."""

    # ── Common commands ───────────────────────────────────────────────────
    INFO = "INFO"
    LIST_COMMANDS = "LIST_COMMANDS"

    # ── TensorDevice commands ─────────────────────────────────────────────
    MATMUL = "MATMUL"
    DOT = "DOT"
    INV = "INV"
    SVD = "SVD"
    EIG = "EIG"

    # Activations
    RELU = "RELU"
    LEAKY_RELU = "LEAKY_RELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    SOFTMAX = "SOFTMAX"
    LOG_SOFTMAX = "LOG_SOFTMAX"
    GELU = "GELU"
    SILU = "SILU"
    ELU = "ELU"
    SELU = "SELU"

    # Arithmetic
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    NEG = "NEG"
    ABS = "ABS"
    POW = "POW"
    SQRT = "SQRT"
    EXP = "EXP"
    LOG = "LOG"

    # Reduction
    SUM = "SUM"
    MEAN = "MEAN"
    STD = "STD"
    VAR = "VAR"
    MAX = "MAX"
    MIN = "MIN"
    ARGMAX = "ARGMAX"
    ARGMIN = "ARGMIN"

    # Shape
    RESHAPE = "RESHAPE"
    TRANSPOSE = "TRANSPOSE"
    FLATTEN = "FLATTEN"
    SQUEEZE = "SQUEEZE"
    UNSQUEEZE = "UNSQUEEZE"
    CAT = "CAT"
    STACK = "STACK"

    # Convolution
    CONV1D = "CONV1D"
    CONV2D = "CONV2D"

    # Pooling
    MAX_POOL1D = "MAX_POOL1D"
    MAX_POOL2D = "MAX_POOL2D"
    AVG_POOL1D = "AVG_POOL1D"
    AVG_POOL2D = "AVG_POOL2D"

    # Normalization
    BATCH_NORM = "BATCH_NORM"
    LAYER_NORM = "LAYER_NORM"
    RMS_NORM = "RMS_NORM"

    # Attention
    ATTENTION = "ATTENTION"

    # Loss functions
    CROSS_ENTROPY = "CROSS_ENTROPY"
    MSE = "MSE"
    MAE = "MAE"

    # Optimizers
    SGD_STEP = "SGD_STEP"
    ADAM_STEP = "ADAM_STEP"

    # Utility
    CLIP_GRAD_NORM = "CLIP_GRAD_NORM"
    DROPOUT = "DROPOUT"
    EMBEDDING = "EMBEDDING"
    LINEAR = "LINEAR"

    # ── NPUDevice commands ────────────────────────────────────────────────
    LOAD = "LOAD"
    UNLOAD = "UNLOAD"
    CALL = "CALL"
    BATCH = "BATCH"
    PIPELINE = "PIPELINE"
    PROFILE = "PROFILE"
    QUANTIZE = "QUANTIZE"
    CHECKPOINT_SAVE = "CHECKPOINT_SAVE"
    CHECKPOINT_LOAD = "CHECKPOINT_LOAD"
    MEMORY = "MEMORY"
    COMPUTE = "COMPUTE"


@dataclass
class IoctlResult:
    """Consistent ioctl response."""

    success: bool
    data: Any = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any = None) -> IoctlResult:
        """Create success result."""
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> IoctlResult:
        """Create failure result."""
        return cls(success=False, error=error)

    def __repr__(self) -> str:
        if self.success:
            return f"IoctlResult(ok, {self.data})"
        return f"IoctlResult(fail, {self.error})"


class IoctlError(Exception):
    """Ioctl error with command context."""

    def __init__(self, command: str, message: str):
        self.command = command
        self.message = message
        super().__init__(f"ioctl {command}: {message}")


def validate_command(command: str | IoctlCommand) -> IoctlCommand:
    """Validate and convert command to enum."""
    if isinstance(command, IoctlCommand):
        return command
    try:
        return IoctlCommand(command)
    except ValueError:
        raise IoctlError(command, f"unknown command")


def validate_args(args: tuple, expected: int, command: str) -> None:
    """Validate argument count."""
    if len(args) < expected:
        raise IoctlError(command, f"expected {expected} args, got {len(args)}")
