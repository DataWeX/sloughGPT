"""
SloughGPT Inference Module

.soul Soul Unit format — the living identity format for trained AI models.

Eager exports (lightweight, no torch): SloProfile, save_soul, load_soul, etc.
Lazy exports (torch-dependent): load_model, InferenceEngine, QType, etc.
"""

from .slo_format import (
    SloProfile,
    PersonalityCore,
    BehavioralTraits,
    CognitiveSignature,
    EmotionalRange,
    GenerationParams,
    ContextParams,
    SouParser,
    create_soul_profile,
    save_soul,
    load_soul,
    write_v3_sou,
    generate_sample_dialogue,
    SOU_MAGIC,
    SOU_VERSION,
    SOU_VERSION_V3,
    SOU_TRADEMARK,
)

__all__ = [
    "SloProfile",
    "PersonalityCore",
    "BehavioralTraits",
    "CognitiveSignature",
    "EmotionalRange",
    "GenerationParams",
    "ContextParams",
    "SouParser",
    "create_soul_profile",
    "save_soul",
    "load_soul",
    "write_v3_sou",
    "generate_sample_dialogue",
    "SOU_MAGIC",
    "SOU_VERSION",
    "SOU_VERSION_V3",
    "SOU_TRADEMARK",
    "QType",
    "QuantizationInfo",
    "Quantizer",
    "SouModelQuantizer",
    "QUANTIZATION_PRESETS",
    "get_quantization_preset",
    "InferenceConfig",
    "SouModelLoader",
    "SouInferenceEngine",
    "load_model",
    "generate",
    "chat",
    "InferenceEngine",
    "KVCache",
    "GenerationRequest",
    "BatchedRequest",
    "create_engine",
    "LlamaInferenceEngine",
    "LlamaInferenceConfig",
    "OllamaInferenceEngine",
    "find_gguf_models",
    "LLAMA_CPP_AVAILABLE",
    "LLAMA_CPP_ERROR",
]

# ── Lazy imports for torch-dependent submodules ──────────────────────────

import importlib

_LAZY_MODULES: dict[str, str] = {
    # .quantization
    "QType": ".quantization",
    "QuantizationInfo": ".quantization",
    "Quantizer": ".quantization",
    "SouModelQuantizer": ".quantization",
    "QUANTIZATION_PRESETS": ".quantization",
    "get_quantization_preset": ".quantization",
    # .loader
    "InferenceConfig": ".loader",
    "SouModelLoader": ".loader",
    "SouInferenceEngine": ".loader",
    "load_model": ".loader",
    "generate": ".loader",
    "chat": ".loader",
    # .engine
    "InferenceEngine": ".engine",
    "KVCache": ".engine",
    "GenerationRequest": ".engine",
    "BatchedRequest": ".engine",
    "create_engine": ".engine",
    # .llama_engine
    "LlamaInferenceEngine": ".llama_engine",
    "LlamaInferenceConfig": ".llama_engine",
    "OllamaInferenceEngine": ".llama_engine",
    "find_gguf_models": ".llama_engine",
    "LLAMA_CPP_AVAILABLE": ".llama_engine",
    "LLAMA_CPP_ERROR": ".llama_engine",
}


def __getattr__(name: str):
    """Lazy-import torch-dependent submodules on first access."""
    module_name = _LAZY_MODULES.get(name)
    if module_name:
        mod = importlib.import_module(module_name, __package__)
        attr = getattr(mod, name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
