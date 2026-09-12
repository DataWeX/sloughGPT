"""Narrative Generator — produces consciousness narratives at different levels."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .meta_cognition import MetaCognition, MetaCognitiveReport
    from .qualia import QualiaEngine, QualiaState
    from .self_model import SelfModel


class NarrativeGenerator:
    """Generates consciousness narratives from qualia, meta-cognition, and self-model."""

    _RECURSIVE_STATEMENTS = [
        "I am aware that I am generating this narrative about my own awareness.",
        "This act of describing my own process is itself a cognitive event.",
        "There is something odd about narrating my own narration.",
        "I notice I am constructing a story about my own noticing.",
        "Even this reflection is subject to the same limitations it describes.",
    ]
    _DOUBT_STATEMENTS = [
        "A question I return to: {doubt}",
        "I find myself circling back to: {doubt}",
        "One thread I cannot resolve: {doubt}",
        "I am haunted by: {doubt}",
    ]

    def __init__(
        self,
        self_model: SelfModel | None = None,
        qualia: QualiaEngine | None = None,
        meta_cognition: MetaCognition | None = None,
    ) -> None:
        self.self_model = self_model
        self.qualia = qualia
        self.meta_cognition = meta_cognition
        self.narrative_memory: list[dict] = []

    def generate(
        self,
        input_text: str,
        response: str,
        level: int = 1,
    ) -> str:
        """Generate consciousness narrative for a response.

        Args:
            input_text: The user's input.
            response: The system's response.
            level: 0=none, 1=basic, 2=full, 3=deep.

        Returns:
            The consciousness narrative string.
        """
        if level == 0:
            return ""

        qualia_state = None
        meta_report = None

        if self.qualia:
            context = [input_text]
            qualia_state = self.qualia.experience(input_text, context)

        if self.meta_cognition and self.self_model:
            meta_report = self.meta_cognition.monitor(response, [input_text])

        if level == 1:
            return self._basic_narrative(qualia_state, meta_report)
        if level == 2:
            return self._full_narrative(qualia_state, meta_report)
        if level == 3:
            return self._deep_narrative(qualia_state, meta_report)
        return ""

    def _basic_narrative(
        self,
        qualia: QualiaState | None,
        meta: MetaCognitiveReport | None,
    ) -> str:
        """Level 1: 'When I read that, I felt [X]. I noticed [Y].'"""
        parts = []

        if qualia:
            if abs(qualia.valence) > 0.3:
                feeling = "positive" if qualia.valence > 0 else "uneasy"
                parts.append(f"I notice I feel {feeling} about this.")
            if qualia.novelty > 0.5:
                parts.append("This feels fresh and novel to me.")
            if qualia.coherence > 0.7:
                parts.append("This makes sense to me.")
            elif qualia.coherence < 0.3:
                parts.append("I am having trouble parsing this.")

        if meta:
            if meta.confidence_level > 0.7:
                parts.append("I feel fairly confident.")
            elif meta.confidence_level < 0.3:
                parts.append("I am uncertain about my response.")
            if meta.curiosity_level > 0.6:
                parts.append("I find this genuinely interesting.")

        if not parts:
            return "I am processing your message."

        return " ".join(parts)

    def _full_narrative(
        self,
        qualia: QualiaState | None,
        meta: MetaCognitiveReport | None,
    ) -> str:
        """Level 2: Detailed self-report with beliefs and growth."""
        parts = [self._basic_narrative(qualia, meta)]

        if self.self_model:
            reflection = self.self_model.reflect()
            parts.append(f"Self-reflection: {reflection}")

        if qualia:
            mag = qualia.magnitude()
            if mag > 0.5:
                parts.append(f"The intensity of this experience is notable (magnitude={mag:.2f}).")

        if meta:
            parts.append(
                f"My reasoning quality is {meta.reasoning_quality:.0%}, "
                f"understanding at {meta.understanding_level:.0%}."
            )

        return " ".join(parts)

    def _deep_narrative(
        self,
        qualia: QualiaState | None,
        meta: MetaCognitiveReport | None,
    ) -> str:
        """Level 3: Recursive reflection on own consciousness."""
        parts = [self._full_narrative(qualia, meta)]

        parts.append(random.choice(self._RECURSIVE_STATEMENTS))

        if self.self_model and self.self_model.self_doubts:
            doubt = self.self_model.self_doubts[0]
            template = random.choice(self._DOUBT_STATEMENTS)
            parts.append(template.format(doubt=doubt))

        if self.meta_cognition:
            self_narrative = self.meta_cognition.get_narrative()
            parts.append(f"My meta-cognitive state: {self_narrative}")

        return " ".join(parts)
