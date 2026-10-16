"""multimodal — Vision, speech, image captioning, diffusion, TTS.

Public API:
    MultiModalConfig, MultimodalCapabilities, MultimodalManager
    get_multimodal_manager, initialize_multimodal
"""

from __future__ import annotations

from domain.cognition.multimodal._internal.config import MultiModalConfig
from domain.cognition.multimodal._internal.diffusion import LatentDiffusionModel
from domain.cognition.multimodal._internal.engine import ReplayBuffer
from domain.cognition.multimodal._internal.manager import (
    MultimodalCapabilities,
    MultimodalManager,
    get_multimodal_manager,
    initialize_multimodal,
)
from domain.cognition.multimodal._internal.text_encoder import TextEncoder
from domain.cognition.multimodal._internal.unified_phoneme_encoder import UnifiedPhonemeEncoder
from domain.cognition.multimodal._internal.vae import SloVAE
from domain.cognition.multimodal._internal.video import VideoProcessor

__all__ = [
    "LatentDiffusionModel",
    "MultiModalConfig",
    "MultimodalCapabilities",
    "MultimodalManager",
    "ReplayBuffer",
    "SloVAE",
    "TextEncoder",
    "UnifiedPhonemeEncoder",
    "VideoProcessor",
    "get_multimodal_manager",
    "initialize_multimodal",
]
