"""
Multi-Modal Support for sloughgpt

FEATURE: multimodal — Vision, speech, image captioning, diffusion, TTS.
DO NOT DELETE. Active module with VisionCNN, MultimodalManager, SpeechRecognizer.

Vision understanding using a custom CNN:
- VisionCNN for image classification (no external downloads)
- Object detection via CNN classification
- Speech recognition (browser Web Speech API)

Optional PyTorch components (only if torch is installed):
- VisionEncoder (ViT), ImageCaptionModel, CLIPModel
"""

import logging
from typing import Optional
from dataclasses import dataclass

try:
    from domains.training.slonet_compat import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None
    F = None

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
# OPTIONAL TORCH COMPONENTS (only when torch is available)
# =============================================================================

if _HAS_TORCH:
    class VisionEncoder(nn.Module):
        """Vision Transformer (ViT) encoder for images."""
        def __init__(self, config: MultiModalConfig):
            super().__init__()
            self.config = config
            self.patch_embed = nn.Conv2d(
                in_channels=3, out_channels=config.vision_hidden_size,
                kernel_size=config.patch_size, stride=config.patch_size,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_hidden_size))
            n_patches = (config.image_size // config.patch_size) ** 2
            self.position_embed = nn.Parameter(torch.zeros(1, n_patches + 1, config.vision_hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.vision_hidden_size, nhead=config.vision_num_heads,
                dim_feedforward=config.vision_hidden_size * 4, dropout=0.1,
                activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.vision_num_layers)
            self.projection = nn.Linear(config.vision_hidden_size, config.projection_dim)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.position_embed, std=0.02)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            x = self.patch_embed(images).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(images.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            x = x + self.position_embed
            x = self.transformer(x)
            return self.projection(x)


# =============================================================================
# SOUlNET-BASED VISION (always available, no torch needed)
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
