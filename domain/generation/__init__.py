"""generation — Inference, output, voice, multimodal.

The "output" of the system — handles token generation, model inference,
voice synthesis, multimodal processing, and chat.

Public API:
    # From inference
    VectorStore, InMemoryVectorStore, VectorEntry
    simple_embed

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
