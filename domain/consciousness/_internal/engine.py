"""Consciousness engine — unified self-awareness, qualia, meta-cognition, narrative."""

from __future__ import annotations

import threading
from typing import Any

from domain.consciousness._internal.config import ConsciousnessConfig
from domain.consciousness._internal.self_model import SelfModel
from domain.consciousness._internal.qualia import QualiaEngine
from domain.consciousness._internal.meta_cognition import MetaCognition
from domain.consciousness._internal.narrative import NarrativeGenerator


class ConsciousnessEngine:
    """Unified consciousness engine combining all 4 modules."""

    def __init__(self, config: ConsciousnessConfig | None = None) -> None:
        self.config = config or ConsciousnessConfig()
        self.self_model = SelfModel(store_path=self.config.store_path)
        self.qualia = QualiaEngine()
        self.meta_cognition = MetaCognition(self.self_model, self.qualia)
        self.narrative = NarrativeGenerator(self.self_model, self.qualia, self.meta_cognition)

        if self.config.is_enabled():
            self.self_model.load()

    def process(self, input_text: str, response: str, level: int | None = None) -> str:
        lvl = level if level is not None else self.config.level
        if lvl == 0:
            return ""

        experience: dict[str, Any] = {
            "input_text": input_text,
            "response": response,
        }

        qualia_state = self.qualia.experience(input_text)
        experience["qualia"] = qualia_state.to_dict()

        self.self_model.observe(experience)

        return self.narrative.generate(input_text, response, level=lvl)

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.is_enabled(),
            "level": self.config.level,
            "episodes": len(self.self_model.episodes),
            "beliefs": dict(self.self_model.self_beliefs),
            "current_qualia": self.qualia.current.to_dict(),
            "curiosity_topics": dict(self.meta_cognition.curiosity_topics),
        }

    def save(self) -> None:
        self.self_model.save()

    def reflect(self) -> str:
        return self.self_model.reflect()

    def clear_episodes(self) -> int:
        count = self.self_model.clear_episodes()
        self.save()
        return count

    def reset_beliefs(self) -> dict[str, float]:
        beliefs = self.self_model.reset_beliefs()
        self.save()
        return beliefs


_module_lock = threading.Lock()
_instance: ConsciousnessEngine | None = None


def get_consciousness(config: ConsciousnessConfig | None = None) -> ConsciousnessEngine:
    global _instance
    with _module_lock:
        if _instance is None:
            if config is None:
                config = ConsciousnessConfig.load()
            _instance = ConsciousnessEngine(config)
        elif config is not None:
            _instance.config = config
        return _instance


def reset_consciousness() -> None:
    global _instance
    with _module_lock:
        _instance = None
