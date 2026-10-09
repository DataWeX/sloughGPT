"""multimodal — Vision, speech, image captioning, diffusion, TTS.

Public API:
    MultiModalConfig, MultimodalCapabilities, MultimodalManager
    get_multimodal_manager, initialize_multimodal
"""

from domain.multimodal._internal.config import MultiModalConfig
from domain.multimodal._internal.diffusion import LatentDiffusionModel
from domain.multimodal._internal.engine import ReplayBuffer
from domain.multimodal._internal.manager import (
    MultimodalCapabilities,
    MultimodalManager,
    get_multimodal_manager,
    initialize_multimodal,
)
from domain.multimodal._internal.text_encoder import TextEncoder
from domain.multimodal._internal.unified_phoneme_encoder import UnifiedPhonemeEncoder
from domain.multimodal._internal.vae import SloVAE
from domain.multimodal._internal.video import VideoProcessor

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
