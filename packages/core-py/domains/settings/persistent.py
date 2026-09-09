"""Persistent Settings — user-facing config that survives restarts.

Stores user preferences, generation settings, and adaptive tuning
in a simple JSON file. Replaces the in-memory dict that resets on restart.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("slo.settings")

DEFAULT_SETTINGS_PATH = Path.home() / ".config" / "sloughgpt" / "settings.json"


@dataclass
class GenerationSettings:
    """Text generation parameters."""
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    max_context_length: int = 2048


@dataclass
class TrainingSettings:
    """User's preferred training defaults."""
    preferred_model: str = ""
    auto_train: bool = False
    auto_train_threshold: int = 100
    preferred_method: str = "auto"
    max_checkpoints: int = 5
    enable_tracking: bool = False


@dataclass
class AdaptiveSettings:
    """Adaptive/learning system preferences."""
    enabled: bool = True
    exploration_rate: float = 0.2  # How often to try variations
    learning_enabled: bool = True  # Whether system learns from outcomes


@dataclass
class VoiceSettings:
    """Voice/audio preferences."""
    noise_gate_db: float = -40.0
    target_level_db: float = -20.0
    vad_enabled: bool = True
    vad_min_speech_ms: int = 250
    agc_enabled: bool = True


@dataclass
class UISettings:
    """User interface preferences."""
    theme: str = "dark"
    language: str = "en"
    show_confidence: bool = True
    compact_mode: bool = False


@dataclass
class AppSettings:
    """All user settings in one place."""
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    adaptive: AdaptiveSettings = field(default_factory=AdaptiveSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    ui: UISettings = field(default_factory=UISettings)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AppSettings:
        """Build settings from dict, using defaults for missing fields."""
        settings = cls()
        if "generation" in d:
            settings.generation = GenerationSettings(**{
                k: v for k, v in d["generation"].items()
                if k in GenerationSettings.__dataclass_fields__
            })
        if "training" in d:
            settings.training = TrainingSettings(**{
                k: v for k, v in d["training"].items()
                if k in TrainingSettings.__dataclass_fields__
            })
        if "adaptive" in d:
            settings.adaptive = AdaptiveSettings(**{
                k: v for k, v in d["adaptive"].items()
                if k in AdaptiveSettings.__dataclass_fields__
            })
        if "voice" in d:
            settings.voice = VoiceSettings(**{
                k: v for k, v in d["voice"].items()
                if k in VoiceSettings.__dataclass_fields__
            })
        if "ui" in d:
            settings.ui = UISettings(**{
                k: v for k, v in d["ui"].items()
                if k in UISettings.__dataclass_fields__
            })
        if "version" in d:
            settings.version = d["version"]
        return settings


class PersistentSettings:
    """Read/write user settings to disk.

    Settings are stored as JSON and loaded on first access.
    Changes are written through immediately so they survive restarts.

    Supports per-user profiles: pass user_id to get user-specific settings.
    """

    def __init__(self, settings_path: Optional[Path] = None, user_id: Optional[str] = None):
        if settings_path:
            self._path = settings_path
        elif user_id:
            self._path = Path.home() / ".config" / "sloughgpt" / f"settings_{user_id}.json"
        else:
            self._path = DEFAULT_SETTINGS_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._settings: Optional[AppSettings] = None
        self._listeners: List[Callable[[AppSettings], None]] = []
        self._user_id = user_id

    def load(self) -> AppSettings:
        """Load settings from disk, or return defaults."""
        if self._settings is not None:
            return self._settings

        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._settings = AppSettings.from_dict(data)
                logger.debug("Loaded settings from %s", self._path)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Corrupt settings file, using defaults: %s", e)
                self._settings = AppSettings()
        else:
            self._settings = AppSettings()

        return self._settings

    def save(self) -> None:
        """Persist current settings to disk."""
        if self._settings is None:
            return
        self._path.write_text(
            json.dumps(self._settings.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        logger.debug("Saved settings to %s", self._path)

    @property
    def settings(self) -> AppSettings:
        if self._settings is None:
            self.load()
        return self._settings

    def update(self, section: str, **kwargs: Any) -> AppSettings:
        """Update a section of settings and save.

        Args:
            section: One of 'generation', 'training', 'adaptive', 'voice', 'ui'
            **kwargs: Key-value pairs to update
        """
        settings = self.settings
        current = getattr(settings, section, None)
        if current is None:
            raise ValueError(f"Unknown settings section: {section}")

        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)
            else:
                logger.warning("Unknown setting: %s.%s", section, key)

        self.save()
        self._notify_listeners()
        return settings

    def get(self, section: str, key: str) -> Any:
        """Get a single setting value."""
        settings = self.settings
        current = getattr(settings, section, None)
        if current is None:
            return None
        return getattr(current, key, None)

    def reset(self) -> AppSettings:
        """Reset all settings to defaults."""
        self._settings = AppSettings()
        self.save()
        self._notify_listeners()
        return self._settings

    def on_change(self, callback: Callable[[AppSettings], None]) -> None:
        """Register a listener for settings changes."""
        self._listeners.append(callback)

    def _notify_listeners(self) -> None:
        for cb in self._listeners:
            try:
                cb(self.settings)
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        return self.settings.to_dict()

    def from_dict(self, data: Dict[str, Any]) -> AppSettings:
        self._settings = AppSettings.from_dict(data)
        self.save()
        self._notify_listeners()
        return self._settings

    @classmethod
    def for_user(cls, user_id: str) -> "PersistentSettings":
        """Get settings instance for a specific user."""
        return cls(user_id=user_id)


# Singleton for convenience
_default_settings: Optional[PersistentSettings] = None


def get_settings(user_id: Optional[str] = None) -> PersistentSettings:
    """Get the global settings instance, or user-specific if user_id provided."""
    global _default_settings
    if user_id:
        return PersistentSettings.for_user(user_id)
    if _default_settings is None:
        _default_settings = PersistentSettings()
    return _default_settings


__all__ = [
    "AppSettings",
    "GenerationSettings",
    "TrainingSettings",
    "AdaptiveSettings",
    "VoiceSettings",
    "UISettings",
    "PersistentSettings",
    "get_settings",
]
