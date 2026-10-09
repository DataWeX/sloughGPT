"""consciousness — Self-awareness, qualia, meta-cognition, narrative.

Canonical implementation of the consciousness subsystem. Re-exported by
``domain.cognition`` (canonical facade).
"""

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import (
    ConsciousnessEngine,
    get_consciousness,
    reset_consciousness,
)
from domain.cognition._internal.consciousness.evaluation import (
    ConsciousnessEvaluator,
    EvaluationReport,
)
from domain.cognition._internal.consciousness.meta_cognition import (
    MetaCognition,
    MetaCognitiveReport,
)
from domain.cognition._internal.consciousness.narrative import NarrativeGenerator
from domain.cognition._internal.consciousness.personality import (
    PersonalityManager,
    PersonalityProfile,
)
from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState
from domain.cognition._internal.consciousness.self_model import (
    Reflection,
    SelfEpisode,
    SelfIdentity,
    SelfModel,
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
