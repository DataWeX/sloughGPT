"""core — Infrastructure, hardware, system-level operations.

The "body" of the system — handles configuration, database, events,
model loading, numpy engine, compression, NPU, kernel, VM.

Also re-exports from cognitive/generation for backward compatibility.
"""

from __future__ import annotations

import importlib as _importlib

from domain.infrastructure import (
    AppConfig,
    AppError,
    ErrorCode,
    EventBus,
    LifecycleManager,
    classify_exception,
    get_config,
    get_db,
    get_event_bus,
    get_lifecycle_manager,
    reload_config,
)
from domain.infrastructure._internal.arch_config import ArchConfig
from domain.infrastructure._internal.model_loader import ModelLoader
from domain.infrastructure._internal.numpy_engine import NumpyEngine
from domain.infrastructure._internal.pugqeep.compressor import PointCompressor
from domain.infrastructure._internal.pugqeep.point import Point
from domain.shell import (
    DaitRuntime,
    Kernel,
    NeuralKernel,
    ShellCommands,
    X86VirtualSystem,
)

__all__ = [
    "AppConfig",
    "get_config",
    "reload_config",
    "EventBus",
    "get_event_bus",
    "LifecycleManager",
    "get_lifecycle_manager",
    "AppError",
    "ErrorCode",
    "classify_exception",
    "get_db",
    "NumpyEngine",
    "ModelLoader",
    "ArchConfig",
    "PointCompressor",
    "Point",
    "Kernel",
    "NeuralKernel",
    "ShellCommands",
    "DaitRuntime",
    "X86VirtualSystem",
    # Legacy re-exports (used by routers)
    "get_rag_service",
    "is_rag_service_ready",
    "RAGService",
    "KGTrainingPipeline",
    "SloEngine",
    "ConsciousnessEngine",
    "get_consciousness",
    "reset_consciousness",
    "SelfModel",
    "QualiaEngine",
    "MetaCognition",
    "NarrativeGenerator",
    "PersonalityManager",
    "PersonalityProfile",
    "VoiceEngine",
    "get_voice_engine",
    "TTSEngine",
    "SpeechRecognizer",
    "PhonemeEngine",
    "get_phoneme_engine",
    "MultimodalManager",
    "get_multimodal_manager",
    "MultimodalCapabilities",
    "ConsciousnessConfig",
    "ConsciousnessEvaluator",
    "EvaluationReport",
    "MetaCognitiveReport",
    "QualiaState",
    "SelfEpisode",
    "SelfIdentity",
]


def __getattr__(name):
    _lazy = {
        # RAG
        "get_rag_service": "domain.core._internal.rag_service",
        "is_rag_service_ready": "domain.core._internal.rag_service",
        "RAGService": "domain.core._internal.rag_service",
        "KGTrainingPipeline": "domain.core._internal.rag_service",
        "SloEngine": "domain.core._internal.soul",
        "soul": "domain.core._internal.soul",
        # Consciousness
        "ConsciousnessEngine": "domain.consciousness",
        "get_consciousness": "domain.consciousness",
        "reset_consciousness": "domain.consciousness",
        "SelfModel": "domain.consciousness",
        "QualiaEngine": "domain.consciousness",
        "MetaCognition": "domain.consciousness",
        "NarrativeGenerator": "domain.consciousness",
        "PersonalityManager": "domain.consciousness",
        "PersonalityProfile": "domain.consciousness",
        "ConsciousnessConfig": "domain.consciousness",
        "ConsciousnessEvaluator": "domain.consciousness",
        "EvaluationReport": "domain.consciousness",
        "MetaCognitiveReport": "domain.consciousness",
        "QualiaState": "domain.consciousness",
        "SelfEpisode": "domain.consciousness",
        "SelfIdentity": "domain.consciousness",
        # Voice
        "VoiceEngine": "domain.voice",
        "get_voice_engine": "domain.voice",
        "TTSEngine": "domain.voice",
        "SpeechRecognizer": "domain.voice",
        "PhonemeEngine": "domain.voice",
        "get_phoneme_engine": "domain.voice",
        # Multimodal
        "MultimodalManager": "domain.multimodal",
        "get_multimodal_manager": "domain.multimodal",
        "MultimodalCapabilities": "domain.multimodal",
    }
    if name in _lazy:
        mod = _importlib.import_module(_lazy[name])
        if hasattr(mod, name):
            return getattr(mod, name)
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
