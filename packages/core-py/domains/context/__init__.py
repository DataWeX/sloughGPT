"""Backward-compatibility shim — imports from the new ``domain.context`` package."""

from domain.context import (
    TraitWeightsConfig,
    PersonalityManager,
    MemoryManager,
    StyleManager,
    TaskManager,
    ConsciousnessManager,
    get_trait_config,
    reset_trait_config,
)

__all__ = [
    "TraitWeightsConfig",
    "PersonalityManager",
    "MemoryManager",
    "StyleManager",
    "TaskManager",
    "ConsciousnessManager",
    "get_trait_config",
    "reset_trait_config",
]
