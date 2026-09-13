"""Backward-compatibility shim — imports from the new ``domain.multimodal`` package."""

from domain.multimodal import (
    MultiModalConfig,
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
