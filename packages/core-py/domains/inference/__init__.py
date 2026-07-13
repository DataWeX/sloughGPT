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
    "InferenceEngine",
    "KVCache",
    "GenerationRequest",
    "BatchedRequest",
    "create_engine",
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
    # .engine
    "InferenceEngine": ".engine",
    "KVCache": ".engine",
    "GenerationRequest": ".engine",
    "BatchedRequest": ".engine",
    "create_engine": ".engine",
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
