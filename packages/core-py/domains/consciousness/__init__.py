"""Backward-compatibility shim — imports from the new ``domain.consciousness`` package."""

from domain.consciousness import (
    ConsciousnessConfig,
    ConsciousnessEngine,
    ConsciousnessEvaluator,
    EvaluationReport,
    MetaCognition,
    MetaCognitiveReport,
    NarrativeGenerator,
    QualiaEngine,
    QualiaState,
    SelfEpisode,
    SelfIdentity,
    SelfModel,
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
