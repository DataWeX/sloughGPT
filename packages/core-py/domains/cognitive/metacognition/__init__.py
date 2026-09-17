"""Backward-compatibility shim — imports from the new ``domain.cognition._internal.metacognition`` package."""

from domain.cognition._internal.metacognition import *

__all__ = [
    "MetacognitiveLevel",
    "CognitiveProcess",
    "MetacognitiveAssessment",
    "ReflectionInsight",
    "CognitiveStateSnapshot",
    "MetacognitiveMonitor",
]
