"""consciousness — Self-awareness, qualia, meta-cognition, narrative.

Backward-compatibility shim — canonical implementation lives in
``domain.cognition._internal.consciousness``.

Public API:
    ConsciousnessConfig, ConsciousnessEngine, ConsciousnessEvaluator, EvaluationReport
    MetaCognition, MetaCognitiveReport, NarrativeGenerator
    QualiaEngine, QualiaState, SelfEpisode, SelfIdentity, SelfModel
    PersonalityManager, PersonalityProfile, Reflection
    get_consciousness, reset_consciousness
"""

from domain.cognition._internal.consciousness import (  # noqa: F401
    ConsciousnessConfig,
    ConsciousnessEngine,
    ConsciousnessEvaluator,
    EvaluationReport,
    MetaCognition,
    MetaCognitiveReport,
    NarrativeGenerator,
    PersonalityManager,
    PersonalityProfile,
    QualiaEngine,
    QualiaState,
    Reflection,
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
    "PersonalityManager",
    "PersonalityProfile",
    "QualiaEngine",
    "QualiaState",
    "Reflection",
    "SelfEpisode",
    "SelfIdentity",
    "SelfModel",
    "get_consciousness",
    "reset_consciousness",
]
