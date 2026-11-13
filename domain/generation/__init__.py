"""generation — Inference, output, voice, multimodal.

The "output" of the system — handles token generation, model inference,
voice synthesis, multimodal processing, and chat.

Public API:
    # From inference
    VectorStore, InMemoryVectorStore, VectorEntry
    simple_embed

    # Generation entry (from inference) — the one language-API contract:
    LanguageClient, hosted, soul, engine, extract_json

    # From models
    ModelInterface, SloughGPTModel

    # From multimodal
    MultiModalConfig, MultimodalManager, TTSEngine

    # From voice
    VoiceEngine, SpeechRecognizer

    # From context
    TraitWeightsConfig, PersonalityManager

    # From chat
    ChatProcessor, MessageHandler
"""

from __future__ import annotations

from domain.context import (
    PersonalityManager as ContextPersonalityManager,
)
from domain.context import (
    TraitWeightsConfig,
)
from domain.inference import (
    InMemoryVectorStore,
    VectorEntry,
    VectorStore,
    simple_embed,
)
from domain.inference.generation import (
    LanguageClient,
    engine,
    extract_json,
    hosted,
    soul,
)
from domain.models import (
    ModelInterface,
    SloughGPTModel,
)
from domain.multimodal import (
    MultiModalConfig,
    MultimodalManager,
)
from domain.voice import (
    VoiceEngine,
)

__all__ = [
    # Inference
    "VectorStore",
    "InMemoryVectorStore",
    "VectorEntry",
    "simple_embed",
    # Generation entry (inference subsystem — the language API)
    "LanguageClient",
    "hosted",
    "soul",
    "engine",
    "extract_json",
    # Models
    "ModelInterface",
    "SloughGPTModel",
    # Multimodal
    "MultiModalConfig",
    "MultimodalManager",
    # Voice
    "VoiceEngine",
    # Context
    "TraitWeightsConfig",
    "ContextPersonalityManager",
]
