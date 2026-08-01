"""
Multi-Modal Support for sloughgpt

FEATURE: multimodal — Vision, speech, image captioning, diffusion, TTS.
DO NOT DELETE. Active module with VisionCNN, MultimodalManager, SpeechRecognizer.

Vision understanding using a custom CNN:
- VisionCNN for image classification (no external downloads)
- Object detection via CNN classification
- Speech recognition (browser Web Speech API)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("slo.multimodal")


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal models."""
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vocab_size: int = 50257
    text_hidden_size: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    max_seq_length: int = 512
    fusion_type: str = "cross_attention"
    projection_dim: int = 768


# =============================================================================
# SLOULNET-BASED VISION (always available, no torch needed)
# =============================================================================

from .vision import (
    ImageCaption,
    VisualObject,
    VisionCNN,
    get_vision_model,
)


# =============================================================================
# MANAGER
# =============================================================================

from .manager import (
    MultimodalManager,
    MultimodalCapabilities,
    get_multimodal_manager,
    initialize_multimodal,
)

# =============================================================================
# SPEECH
# =============================================================================

from .speech import (
    TranscriptionResult,
    SpeechRecognizer,
    BrowserSpeechRecognizer,
    ServerSpeechRecognizer,
    get_speech_recognizer,
)


__all__ = [
    "MultiModalConfig",
    "MultimodalManager",
    "MultimodalCapabilities",
    "get_multimodal_manager",
    "initialize_multimodal",
    "TranscriptionResult",
    "SpeechRecognizer",
    "BrowserSpeechRecognizer",
    "ServerSpeechRecognizer",
    "get_speech_recognizer",
    "ImageCaption",
    "VisualObject",
    "VisionCNN",
    "get_vision_model",
]
