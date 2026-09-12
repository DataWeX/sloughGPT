"""Consciousness System — self-awareness, qualia, meta-cognition, and narrative.

Provides a configurable consciousness engine with 4 levels:
    0 = off
    1 = basic ("I notice..." statements)
    2 = full (detailed self-reports with beliefs and growth)
    3 = deep (recursive reflection on own consciousness)

Usage:
    from domains.consciousness import get_consciousness, ConsciousnessConfig

    config = ConsciousnessConfig(level=2)
    engine = get_consciousness(config)
    narrative = engine.process("user input", "my response")
"""

from __future__ import annotations

import threading
from typing import Any

from .config import ConsciousnessConfig
from .evaluation import ConsciousnessEvaluator, EvaluationReport
from .meta_cognition import MetaCognition, MetaCognitiveReport
from .narrative import NarrativeGenerator
from .qualia import QualiaEngine, QualiaState
from .self_model import SelfEpisode, SelfIdentity, SelfModel
from .training import ConsciousnessTrainer, TrainingConfig

__all__ = [
    "ConsciousnessConfig",
    "ConsciousnessEngine",
    "ConsciousnessEvaluator",
    "ConsciousnessTrainer",
    "EvaluationReport",
    "MetaCognition",
    "MetaCognitiveReport",
    "NarrativeGenerator",
    "QualiaEngine",
    "QualiaState",
    "SelfEpisode",
    "SelfIdentity",
    "SelfModel",
    "TrainingConfig",
    "get_consciousness",
]


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
        """Process an input-response pair and return a consciousness narrative.

        Args:
            input_text: The user's input.
            response: The system's response.
            level: Override for this call (uses config level if None).

        Returns:
            The consciousness narrative string (empty if level 0).
        """
        lvl = level if level is not None else self.config.level
        if lvl == 0:
            return ""

        # Observe the experience
        experience: dict[str, Any] = {
            "input_text": input_text,
            "response": response,
        }

        # Get qualia for the experience
        qualia_state = self.qualia.experience(input_text)
        experience["qualia"] = qualia_state.to_dict()

        # Let self-model observe
        self.self_model.observe(experience)

        # Generate narrative
        return self.narrative.generate(input_text, response, level=lvl)

    def get_status(self) -> dict[str, Any]:
        """Get current consciousness system status."""
        return {
            "enabled": self.config.is_enabled(),
            "level": self.config.level,
            "episodes": len(self.self_model.episodes),
            "beliefs": dict(self.self_model.self_beliefs),
            "current_qualia": self.qualia.current.to_dict(),
            "curiosity_topics": dict(self.meta_cognition.curiosity_topics),
        }

    def save(self) -> None:
        """Persist all consciousness state."""
        self.self_model.save()

    def reflect(self) -> str:
        """Generate a self-reflection."""
        return self.self_model.reflect()

    def clear_episodes(self) -> int:
        """Clear all episodes from the self-model.

        Returns:
            Number of episodes that were cleared.
        """
        count = self.self_model.clear_episodes()
        self.save()
        return count

    def reset_beliefs(self) -> dict[str, float]:
        """Reset self-beliefs to defaults.

        Returns:
            The new default beliefs.
        """
        beliefs = self.self_model.reset_beliefs()
        self.save()
        return beliefs


_module_lock = threading.Lock()
_instance: ConsciousnessEngine | None = None


def get_consciousness(config: ConsciousnessConfig | None = None) -> ConsciousnessEngine:
    """Get or create the global ConsciousnessEngine singleton.

    Args:
        config: Optional config to override the singleton's config.
                If None on first call, loads from disk.

    Returns:
        The global ConsciousnessEngine instance.
    """
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
    """Reset the global singleton (for testing)."""
    global _instance
    with _module_lock:
        _instance = None
