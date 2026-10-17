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
    ConsciousnessTrainer, TrainingConfig (the optional model bridge)
    get_consciousness, peek_consciousness, reset_consciousness
"""

from __future__ import annotations

# Every public name is imported ON ACCESS, never on package import.
#
# This is a dependency boundary, not a micro-optimisation. Importing this
# package used to eagerly pull in `_internal.core`, `_internal.processor` and
# `_internal.rag_service`, which in turn drag in the inference stack and numpy.
# So the cost of touching ANY cognition submodule was "load the whole model
# stack" — including for the consciousness/dream engine, which needs neither
# and is written to run on the standard library alone.
#
# The boundary matters because an inner life that only runs when the entire
# model stack is importable is a demo, not an engine: it cannot be exercised
# by a CLI, a contract test, or a future renderer that has no business
# importing inference. The facade must not be the thing that breaks that.
#
# Behaviour is identical for every consumer — `from domain.cognition import X`
# resolves through __getattr__ on first access and is then cached in globals().
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # ._internal.base
    "CognitiveDomain": ("._internal.base", "CognitiveDomain"),
    "CognitiveException": ("._internal.base", "CognitiveException"),
    # ._internal.consciousness
    "ConsciousnessConfig": ("._internal.consciousness", "ConsciousnessConfig"),
    "ConsciousnessEngine": ("._internal.consciousness", "ConsciousnessEngine"),
    "ConsciousnessEvaluator": ("._internal.consciousness", "ConsciousnessEvaluator"),
    "EvaluationReport": ("._internal.consciousness", "EvaluationReport"),
    "MetaCognition": ("._internal.consciousness", "MetaCognition"),
    "MetaCognitiveReport": ("._internal.consciousness", "MetaCognitiveReport"),
    "NarrativeGenerator": ("._internal.consciousness", "NarrativeGenerator"),
    "PersonalityManager": ("._internal.consciousness", "PersonalityManager"),
    "PersonalityProfile": ("._internal.consciousness", "PersonalityProfile"),
    "QualiaEngine": ("._internal.consciousness", "QualiaEngine"),
    "QualiaState": ("._internal.consciousness", "QualiaState"),
    "Reflection": ("._internal.consciousness", "Reflection"),
    "SelfEpisode": ("._internal.consciousness", "SelfEpisode"),
    "SelfIdentity": ("._internal.consciousness", "SelfIdentity"),
    "SelfModel": ("._internal.consciousness", "SelfModel"),
    "get_consciousness": ("._internal.consciousness", "get_consciousness"),
    "peek_consciousness": ("._internal.consciousness", "peek_consciousness"),
    "reset_consciousness": ("._internal.consciousness", "reset_consciousness"),
    # ._internal.core
    "CognitiveCore": ("._internal.core", "CognitiveCore"),
    "CreativeIdea": ("._internal.core", "CreativeIdea"),
    "ReasoningChain": ("._internal.core", "ReasoningChain"),
    "ReasoningType": ("._internal.core", "ReasoningType"),
    "ThinkingMode": ("._internal.core", "ThinkingMode"),
    "ThoughtProcess": ("._internal.core", "ThoughtProcess"),
    # ._internal.processor
    "CognitiveProcessor": ("._internal.processor", "CognitiveProcessor"),
    # ._internal.rag_service
    "KGTrainingPipeline": ("._internal.rag_service", "KGTrainingPipeline"),
    "RAGService": ("._internal.rag_service", "RAGService"),
    "get_rag_service": ("._internal.rag_service", "get_rag_service"),
    "is_rag_service_ready": ("._internal.rag_service", "is_rag_service_ready"),
    # ._internal.consciousness.training — the bridge to the model, optional
    "ConsciousnessTrainer": ("._internal.consciousness.training", "ConsciousnessTrainer"),
    "TrainingConfig": ("._internal.consciousness.training", "TrainingConfig"),
}


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = target
    import importlib

    value = getattr(importlib.import_module(module_path, package=__name__), attr)
    # Cache in globals so repeated access is a plain dict hit, not an import.
    globals()[name] = value
    return value


def __dir__():
    # Keep discovery intact: dir() must still advertise the lazy names.
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


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
    "peek_consciousness",
    "reset_consciousness",
    "ConsciousnessTrainer",
    "TrainingConfig",
]
