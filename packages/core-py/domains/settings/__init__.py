"""Backward-compatibility shim — imports from the new ``domain.settings`` package."""

from domain.settings import (
    AdaptiveSettings,
    AppSettings,
    GenerationSettings,
    PersistentSettings,
    TrainingSettings,
    UISettings,
    VoiceSettings,
    get_settings,
)

__all__ = [
    "GenerationSettings",
    "TrainingSettings",
    "AdaptiveSettings",
    "VoiceSettings",
    "UISettings",
    "AppSettings",
    "PersistentSettings",
    "get_settings",
]
