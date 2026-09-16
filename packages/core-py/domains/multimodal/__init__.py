"""Backward-compatibility shim — imports from the new ``domain.multimodal`` package."""

from domain.multimodal import (
    MultimodalCapabilities,
    MultiModalConfig,
    MultimodalManager,
    get_multimodal_manager,
    initialize_multimodal,
)
from domain.multimodal._internal import engine  # noqa: F401

__all__ = [
    "MultiModalConfig",
    "MultimodalCapabilities",
    "MultimodalManager",
    "get_multimodal_manager",
    "initialize_multimodal",
]
