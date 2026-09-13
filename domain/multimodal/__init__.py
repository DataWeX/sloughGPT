"""multimodal — Vision, speech, image captioning, diffusion, TTS.

Public API:
    MultiModalConfig, MultimodalCapabilities, MultimodalManager
    get_multimodal_manager, initialize_multimodal
"""

from domain.multimodal._internal.config import MultiModalConfig
from domain.multimodal._internal.manager import (
    MultimodalCapabilities,
    MultimodalManager,
    get_multimodal_manager,
    initialize_multimodal,
)

__all__ = [
    "MultiModalConfig",
    "MultimodalCapabilities",
    "MultimodalManager",
    "get_multimodal_manager",
    "initialize_multimodal",
]
