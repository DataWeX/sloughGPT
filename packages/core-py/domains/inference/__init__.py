"""
SloughGPT Inference Module

.soul Soul Unit format — the living identity format for trained AI models.

All exports are lightweight.
"""

from __future__ import annotations

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
]
