"""Consciousness system configuration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("slo.consciousness.config")

_CONFIG_FILENAME = "consciousness_config.json"


@dataclass
class ConsciousnessConfig:
    """Configuration for the consciousness system.

    Levels:
        0 = off (no consciousness processing)
        1 = basic ("I notice..." statements)
        2 = full (detailed self-reports with beliefs and growth)
        3 = deep (recursive reflection on own consciousness)
    """

    level: int = 0
    max_tokens: int = 150
    training_enabled: bool = False
    training_interval: int = 100
    lora_rank: int = 4
    lora_alpha: int = 8
    store_path: str = "data/consciousness"
    reflection_interval: int = 300

    def is_enabled(self) -> bool:
        return self.level > 0

    def validate(self) -> None:
        if self.level not in (0, 1, 2, 3):
            raise ValueError(f"Level must be 0-3, got {self.level}")
        if self.max_tokens < 10:
            raise ValueError(f"max_tokens must be >= 10, got {self.max_tokens}")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")

    def get_store_path(self) -> Path:
        return Path(self.store_path)

    def _config_file(self) -> Path:
        return self.get_store_path() / _CONFIG_FILENAME

    def save(self) -> None:
        """Persist config to disk."""
        try:
            path = self._config_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self)
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save consciousness config: %s", e)

    @classmethod
    def load(cls, store_path: str = "data/consciousness") -> "ConsciousnessConfig":
        """Load config from disk, falling back to defaults."""
        config_file = Path(store_path) / _CONFIG_FILENAME
        try:
            if config_file.exists():
                data = json.loads(config_file.read_text())
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.warning("Failed to load consciousness config: %s", e)
        return cls(store_path=store_path)
