"""Backward-compatibility shim — see ``domain.cognition._internal.consciousness.evaluation``."""

from domain.cognition._internal.consciousness.evaluation import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.evaluation import (
    ConsciousnessEvaluator,
    EvaluationReport,
)

__all__ = ["ConsciousnessEvaluator", "EvaluationReport"]
