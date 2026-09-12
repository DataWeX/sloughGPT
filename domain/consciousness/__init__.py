"""consciousness — Self-awareness, qualia, meta-cognition, narrative.

Public API:
    ConsciousnessConfig, ConsciousnessEngine, ConsciousnessEvaluator, EvaluationReport
    MetaCognition, MetaCognitiveReport, NarrativeGenerator
    QualiaEngine, QualiaState, SelfEpisode, SelfIdentity, SelfModel
    get_consciousness, reset_consciousness
"""

from domain.consciousness._internal.config import ConsciousnessConfig
from domain.consciousness._internal.evaluation import ConsciousnessEvaluator, EvaluationReport
from domain.consciousness._internal.meta_cognition import MetaCognition, MetaCognitiveReport
from domain.consciousness._internal.narrative import NarrativeGenerator
from domain.consciousness._internal.qualia import QualiaEngine, QualiaState
from domain.consciousness._internal.self_model import SelfEpisode, SelfIdentity, SelfModel
from domain.consciousness._internal.engine import (
    ConsciousnessEngine,
    get_consciousness,
    reset_consciousness,
)

__all__ = [
    "ConsciousnessConfig",
    "ConsciousnessEngine",
    "ConsciousnessEvaluator",
    "EvaluationReport",
    "MetaCognition",
    "MetaCognitiveReport",
    "NarrativeGenerator",
    "QualiaEngine",
    "QualiaState",
    "SelfEpisode",
    "SelfIdentity",
    "SelfModel",
    "get_consciousness",
    "reset_consciousness",
]
