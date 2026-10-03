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
    "run_doctor",
    "DoctorReport",
]


def __getattr__(name):
    _lazy = {
        # Shell (heavy: loads the x86 VM lazily)
        "X86VirtualSystem": "domain.shell._internal.vm",
        # RAG (canonical module lives under domain.cognition)
        "get_rag_service": "domain.cognition._internal.rag_service",
        "is_rag_service_ready": "domain.cognition._internal.rag_service",
        "RAGService": "domain.cognition._internal.rag_service",
        "KGTrainingPipeline": "domain.cognition._internal.rag_service",
        "SloEngine": "domain.core._internal.soul",
        "soul": "domain.core._internal.soul",
        # Consciousness (canonical home: domain.cognition._internal.consciousness)
        "ConsciousnessEngine": "domain.cognition",
        "get_consciousness": "domain.cognition",
        "reset_consciousness": "domain.cognition",
        "SelfModel": "domain.cognition",
        "QualiaEngine": "domain.cognition",
        "MetaCognition": "domain.cognition",
        "NarrativeGenerator": "domain.cognition",
        "PersonalityManager": "domain.cognition",
        "PersonalityProfile": "domain.cognition",
        "ConsciousnessConfig": "domain.cognition",
        "ConsciousnessEvaluator": "domain.cognition",
        "EvaluationReport": "domain.cognition",
        "MetaCognitiveReport": "domain.cognition",
        "QualiaState": "domain.cognition",
        "SelfEpisode": "domain.cognition",
        "SelfIdentity": "domain.cognition",
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
        # Site doctor (Phase A: report-only probes)
        "run_doctor": "domain.core._internal.doctor",
        "DoctorReport": "domain.core._internal.doctor",
    }
    if name in _lazy:
        mod = _importlib.import_module(_lazy[name])
        if hasattr(mod, name):
            return getattr(mod, name)
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
