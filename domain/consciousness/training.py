"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.training``."""

from domain.cognition._internal.consciousness.training import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.training import (
    ConsciousnessPair,
    ConsciousnessTrainer,
    TrainingConfig,
)

__all__ = ["ConsciousnessPair", "ConsciousnessTrainer", "TrainingConfig"]
