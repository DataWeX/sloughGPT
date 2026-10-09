"""context — Context managers for model behavior steering.

Public API:
    TraitWeightsConfig, PersonalityManager, MemoryManager, StyleManager, TaskManager
    ConsciousnessManager, get_trait_config, reset_trait_config
"""

from domain.context._internal.managers import (
    ConsciousnessManager,
    MemoryManager,
    PersonalityManager,
    StyleManager,
    TaskManager,
    TraitWeightsConfig,
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
