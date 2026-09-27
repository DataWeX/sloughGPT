"""cognition — Reasoning, thinking, creativity, RAG, and consciousness.

The model's reasoning and self-awareness engine.

Public API:
    CognitiveDomain, CognitiveException
    CognitiveCore, ThinkingMode, ReasoningType, ThoughtProcess
    CreativeIdea, ReasoningChain, CognitiveProcessor
    RAGService, KGTrainingPipeline, get_rag_service, is_rag_service_ready
    ConsciousnessConfig, ConsciousnessEngine, ConsciousnessEvaluator, EvaluationReport
    MetaCognition, MetaCognitiveReport, NarrativeGenerator
    PersonalityManager, PersonalityProfile, QualiaEngine, QualiaState
    Reflection, SelfEpisode, SelfIdentity, SelfModel
    get_consciousness, reset_consciousness
"""


def __getattr__(name):
    _lazy = {
        "ConsciousnessTrainer": "domain.cognition._internal.consciousness.training",
        "TrainingConfig": "domain.cognition._internal.consciousness.training",
    }
    if name in _lazy:
        import importlib

        return getattr(importlib.import_module(_lazy[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from domain.cognition._internal.base import (
    CognitiveDomain,
    CognitiveException,
)
from domain.cognition._internal.consciousness import (
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
from domain.cognition._internal.core import (
    CognitiveCore,
    CreativeIdea,
    ReasoningChain,
    ReasoningType,
    ThinkingMode,
    ThoughtProcess,
)
from domain.cognition._internal.processor import (
    CognitiveProcessor,
)
from domain.cognition._internal.rag_service import (
    KGTrainingPipeline,
    RAGService,
    get_rag_service,
    is_rag_service_ready,
)

__all__ = [
    "CognitiveDomain",
    "CognitiveException",
    "CognitiveCore",
    "ThinkingMode",
    "ReasoningType",
    "ThoughtProcess",
    "CreativeIdea",
    "ReasoningChain",
    "CognitiveProcessor",
    "RAGService",
    "KGTrainingPipeline",
    "get_rag_service",
    "is_rag_service_ready",
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
