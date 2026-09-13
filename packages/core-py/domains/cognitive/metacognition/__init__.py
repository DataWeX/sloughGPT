"""Backward-compatibility shim — imports from the new ``domain.cognitive._internal.metacognition`` package."""

from domain.cognitive._internal.metacognition import *

__all__ = [
    "MetacognitiveLevel",
    "CognitiveProcess",
    "MetacognitiveAssessment",
    "ReflectionInsight",
    "CognitiveStateSnapshot",
    "MetacognitiveMonitor",
]
