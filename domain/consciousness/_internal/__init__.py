"""Backward-compatibility shims — canonical modules live in
``domain.cognition._internal.consciousness``.

Each submodule re-exports the same objects (same function/class identities)
so singletons and mock patch-targets stay coherent across both import paths.
"""

from domain.cognition._internal.consciousness.config import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.engine import (
    ConsciousnessEngine,
    get_consciousness,
    reset_consciousness,
)
from domain.cognition._internal.consciousness.evaluation import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.evaluation import (
    ConsciousnessEvaluator,
    EvaluationReport,
)
from domain.cognition._internal.consciousness.meta_cognition import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.meta_cognition import (
    MetaCognition,
    MetaCognitiveReport,
)
from domain.cognition._internal.consciousness.narrative import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.narrative import NarrativeGenerator
from domain.cognition._internal.consciousness.personality import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.personality import (
    PersonalityManager,
    PersonalityProfile,
)
from domain.cognition._internal.consciousness.qualia import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState
from domain.cognition._internal.consciousness.self_model import *  # noqa: F401,F403
from domain.cognition._internal.consciousness.self_model import (
    Reflection,
    SelfEpisode,
    SelfIdentity,
    SelfModel,
)
